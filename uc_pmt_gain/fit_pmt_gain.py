#!/usr/bin/python
#import pickle
#print pickle.Pickler.dispatch
import dill
#print pickle.Pickler.dispatch

import ROOT as root
import sys, os

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

import emcee, corner, click
import neriX_analysis, neriX_datasets, neriX_config
from rootpy.plotting import Hist2D, Hist, Legend, Canvas
from rootpy.tree import Tree
from rootpy.io import File
from rootpy import stl
import numpy as np
import tqdm, time, copy_reg, types, pickle
from root_numpy import tree2array, array2tree
import scipy.optimize as op
import scipy.special
from scipy.stats import norm, poisson
from scipy.special import erf
from math import floor

import astroML.density_estimation

import cuda_pmt_mc
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray
#import pycuda.autoinit

import atexit

from sklearn import neighbors
from sklearn import grid_search
from sklearn import preprocessing




def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, (variance)**0.5)


def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))



def array_poisson_binned_likelihood(a_model, a_data):
    return a_data*np.log(a_model) - a_model - (a_data*np.log(a_data) - a_data + 0.5*np.log(2*np.pi*a_data))
    #return -np.log(a_data**0.5) - 0.5*np.log(2*np.pi) - 0.5*(a_data - a_model)**2/a_data



def poisson_binned_likelihood(a_model, a_data):
    #print a_data*np.log(a_model) - a_model
    a_model += 1e-6

    # this form matches ROOT's form
    mask = (a_model > 0) & (a_data > 0)
    return np.sum(array_poisson_binned_likelihood(a_model[mask], a_data[mask]))






class fit_pmt_gain(object):
    def __init__(self, filename, gpu_number=0, run=16, channel_number=17, num_mc_events=1e6, num_loops=1, b_making_comparison_plots=False, b_use_cascade_model=True):
    
    

        self.run = run

        # make class methods pickleable for multithreading process
        copy_reg.pickle(types.MethodType, reduce_method)

        self.data_dir = './data/'
        self.num_hist_events = int(2e4)
        # num_electrons = num_count_samples * (1/frequency_digitizer) * (1/impedance) * (1/external_gain) * (1/charge_of_electron) * (dynamic_voltage_range/2**num_bits)
        self.conversion_to_num_electrons = 1./(250e6)/50./10./1.6e-19*2./2**12
        
        
        # start GPU
        drv.init()
        dev = drv.Device(gpu_number)
        ctx = dev.make_context()
        
        print '\nDevice Number: %d' % (gpu_number)
        print 'Device Name: %s\n' % (dev.name())
        
        atexit.register(ctx.pop)
        
        self.gpu_cascade_model = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('cascade_pmt_model')
        self.gpu_cascade_model_arrays = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('cascade_pmt_model_array')
        self.gpu_pure_cascade_spectrum = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('pure_cascade_spectrum')
        self.gpu_fixed_pe_cascade_spectrum = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('fixed_pe_cascade_spectrum')
        self.setup_kernel = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('setup_kernel')

        self.gpu_gaussian_model = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('gaussian_pmt_model')
        self.gpu_gaussian_model_arrays = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('gaussian_pmt_model_arrays')
        self.gpu_pure_gaussian_spectrum = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('pure_gaussian_spectrum')
        self.gpu_fixed_pe_gaussian_spectrum = SourceModule(cuda_pmt_mc.cuda_pmt_mc, no_extern_c=True).get_function('fixed_pe_gaussian_spectrum')
        
        
        
        self.num_mc_events = int(num_mc_events)
        self.d_gpu_scale = {}
        block_dim = 1024/2
        self.d_gpu_scale['block'] = (block_dim,1,1)
        numBlocks = floor(num_mc_events / float(block_dim))
        self.d_gpu_scale['grid'] = (int(numBlocks), 1)
        self.num_mc_events = int(numBlocks*block_dim)
        
        self.num_loops = np.asarray(num_loops, dtype=np.int32)
        
        self.b_use_cascade_model = b_use_cascade_model
        
        self.b_making_comparison_plots = b_making_comparison_plots
        
        
        seed = int(time.time())
        self.rng_states = drv.mem_alloc(self.num_mc_events*pycuda.characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
        self.setup_kernel(np.int32(self.num_mc_events), self.rng_states, np.uint64(seed), np.uint64(0), **self.d_gpu_scale)
        print 'Cuda random states setup...\n'
        
        self.filename = filename
        self.d_fit_files = {}
        
        if not self.filename[:5] == 'nerix' and not self.filename == 'fake':
            try:
                self.d_fit_files['a_integral'] = pickle.load(open('%s%s.p' % (self.data_dir, self.filename), 'r'))
            except:
                self.d_fit_files['root_file'] = File(self.data_dir + self.filename + '.root')
                h_integral = self.d_fit_files['root_file'].laser_data_integral_hist
                a_integral = np.zeros(self.num_hist_events)
                for i in tqdm.tqdm(xrange(self.num_hist_events)):
                    a_integral[i] = h_integral.GetRandom()
                # convert to num electrons
                
                #a_integral, dummy = neriX_analysis.convert_hist_into_array_of_values(h_integral, scaling_factor=0.1)
                
                a_integral *= self.conversion_to_num_electrons
                
                # dump array to file
                pickle.dump(a_integral, open('%s%s.p' % (self.data_dir, filename), 'w'))
                self.d_fit_files['a_integral'] = a_integral
            
            self.file_identifier = self.filename[-9:]
            
            self.d_bkg_fit = pickle.load(open('./bkg_results/bkg_%s.p' % (filename), 'r'))
            self.b_nerix_file = False
        
        else:
            self.d_fit_files['a_integral'] = pickle.load(open('%s%s.p' % (self.data_dir, self.filename), 'r'))
        
            self.file_identifier = self.filename
            self.b_nerix_file = True
        
        
        # get dynode chains
        if self.b_nerix_file and not self.filename == 'fake':
            self.a_dynode_chain = np.asarray([2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.212], dtype=np.float32)
        else:
            self.a_dynode_chain = np.asarray([4, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.float32)

        self.gpu_dynode_chain = pycuda.gpuarray.to_gpu(self.a_dynode_chain)
        self.num_dynodes = np.asarray(len(self.a_dynode_chain), dtype=np.int32)
        
        num_bins_uc = 100
        num_bins_nerix = 100

        if b_use_cascade_model:
            if self.file_identifier == '0062_0061':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 2e7]
                # 690.5, 627, 643
                #self.a_free_par_guesses = [8.15021255e-03, 8.89785325e-01, 1.48677929e+01, 2.63208068e+00, 2.01119635e-01, 9.42134482e-01, 8.58082164e-01, -8.74101683e+03, 2.53032354e+05, 3.20922935e+00, 3.80128883e-01, 9.97487422e-01]
                self.a_free_par_guesses = [8.15021255e-03/1.3, 8.89785325e-01, 1.48677929e+01, 2.63208068e+00, 2.01119635e-01, 9.42134482e-01, 8.58082164e-01, -8.74101683e+03, 2.53032354e+05, 4.95481176e+00, 2.37034972e-01, 9.97487422e-01]
            
            elif self.file_identifier == '0066_0065':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 1.2e7]
                # 654.2, 630, 475
                #self.a_free_par_guesses = [7.91416462e-03, 9.24608744e-01, 1.55085462e+01, 7.46668444e-01, 1.84963960e-01, 9.56690585e-01, 8.87439385e-01, -1.11319075e+04, 2.54186783e+05, 8.93565711e+00, 1.27032774e-01, 1.00727441e+00]
                self.a_free_par_guesses = [7.91416462e-03*2, 9.24608744e-01, 1.55085462e+01, 7.46668444e-01, 1.84963960e-01, 9.56690585e-01, 8.87439385e-01, -1.11319075e+04, 2.54186783e+05, 4.95481176e+00, 2.37034972e-01, 1.00727441e+00]
            
            elif self.file_identifier == '0067_0068':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 7.5e6]
                # 603.3, 451, 447.2
                self.a_free_par_guesses = [6.67047844e-03, 9.09472860e-01, 1.46730288e+01, 9.48999507e-01, 1.86821903e-01, 8.46757458e-01, 8.76305500e-01, -9.44206855e+03, 2.53770049e+05, 4.95481176e+00, 2.37034972e-01, 1.00512801e+00]
            
            elif self.file_identifier == '0071_0072' or self.file_identifier == 'fake':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 3.4e7]
                # 924.0, 642.1, 538.9
                #self.a_free_par_guesses = [9.95865571e-03, 9.27764164e-01, 1.63507208e+01, 2.22283848e+00, 1.90175341e-01, 9.43649321e-01, 8.93044254e-01, -9.44900313e+03, 2.55406771e+05, 1.13336062e+01, 1.00481481e-01, 1.00496831e+00]
                self.a_free_par_guesses = [9.95865571e-03*2, 9.27764164e-01, 1.63507208e+01, 2.22283848e+00, 1.90175341e-01, 9.43649321e-01, 8.93044254e-01, -9.44900313e+03, 2.55406771e+05, 4.95481176e+00, 2.37034972e-01, 1.00496831e+00]
            
            elif self.file_identifier == '0073_0074':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 4.2e7]
                # higher than 775, 775, 483.6
                #self.a_free_par_guesses = [9.92998617e-03, 8.58027288e-01, 1.60443995e+01, 4.63392899e+00, 1.93849276e-01, 9.01828448e-01, 8.41169139e-01, -1.00191623e+04, 2.54874225e+05, 9.52724073e+00, 2.34781348e-01, 9.94536279e-01]
                self.a_free_par_guesses = [9.92998617e-03*2, 0.93, 1.60443995e+01, 4.63392899e+00/4., 1.93849276e-01, 0.96, 0.9, -1.00191623e+04, 2.54874225e+05, 9.3, 2.34781348e-01, 9.94536279e-01]

            elif self.file_identifier == 'nerix_160418_1523':
                self.d_fit_files['settings'] = [num_bins_nerix, -5e5, 3.e6]
                # 441.4, 455, 441.0
                self.a_free_par_guesses = [9.58421447e-03, 8.43049707e-01, 7.34426895e+00*1.5, 3., 3.30671864e-01/1.5, 8.20732092e-01, 0.92, 6.62353661e+04*1.2, 2.30312280e+05, 8.74921049e+00, 1.39186822e-01/1.1, 9.91798366e-01]


            elif self.file_identifier == 'nerix_160418_1531':
                self.d_fit_files['settings'] = [num_bins_nerix, -5e5, 4.e6]
                # 446.2, 475.4, 458.3
                #self.a_free_par_guesses = [5.70593109e-03, 8.05790314e-01, 7.16339184e+00, 2.99910557e+00, 3.40470779e-01, 9.06746120e-01, 9.45281711e-01, 9.95610772e+04, 2.37837451e+05, 7.89434243e+00, 2.81476989e-01, 9.96493253e-01]
                self.a_free_par_guesses = [9.58421447e-03, 8.43049707e-01, 7.34426895e+00*1.5, 3., 3.30671864e-01/1.5, 8.20732092e-01, 0.92, 6.62353661e+04*1.2, 2.30312280e+05, 8.74921049e+00*1.9, 1.39186822e-01/1.1, 9.91798366e-01]
            
            else:
                print '\n\nSettings do not exist for given setup: %s\n\n' % (self.file_identifier)
                sys.exit()
                
        # settings for gaussian model
        else:
            if self.file_identifier == '0062_0061':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 2e7]
                # 715.1
                #
                self.a_free_par_guesses = [8.64747984e-01, 6.02664039e+06, 1.77199212e+06, 1.24751024e+06, 6.97973938e+05, -8.40312418e+03, 2.52761845e+05, 3.96308054e+05, 1.21273128e-01, 1.23619489e+00, 1.00211284e+00]
            
            elif self.file_identifier == '0066_0065':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 1.2e7]
                # 465.4
                #
                self.a_free_par_guesses = [8.77361680e-01, 3.64052408e+06, 1.05777712e+06, 7.79704903e+05, 4.48063736e+05, -9.84898497e+03, 2.54202366e+05, 9.09153177e+05, 1.51388311e-01, 1.20055657e+00, 1.00419426e+00]
                
            elif self.file_identifier == '0067_0068':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 7.5e6]
                # 420.2
                #
                self.a_free_par_guesses = [8.87016757e-01, 2.11151344e+06, 6.16342188e+05, 4.95812960e+05, 3.77603380e+05, -9.11680472e+03, 2.53740791e+05, 7.26733316e+05, 4.19097943e-01, 1.20665106e+00, 1.00072005e+00]
                
            elif self.file_identifier == '0071_0072':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 3.4e7]
                # 1099.8
                #
                self.a_free_par_guesses = [8.53866425e-01, 9.44873093e+06, 2.71702085e+06, 1.86409732e+06, 9.95134439e+05, -8.23272143e+03, 2.56266223e+05, 8.76963878e+05, 1.34307038e-01, 1.25087809e+00, 1.00419532e+00]
            
            elif self.file_identifier == '0073_0074':
                self.d_fit_files['settings'] = [num_bins_uc, -1e6, 4.2e7]
                # 612.4
                #
                self.a_free_par_guesses = [8.52535348e-01, 9.27642130e+06, 2.64612733e+06, 1.83103901e+06, 9.13860209e+05, -1.05361382e+04, 2.54526377e+05, 6.22232690e+05, 3.48473633e-01, 2.23966247e+00, 1.00323186e+00]


            elif self.file_identifier == 'nerix_160418_1523':
                self.d_fit_files['settings'] = [num_bins_nerix, -5e5, 3.e6]
                self.a_free_par_guesses = [8.52875320e-01, 8.82719771e+05, 5.60471194e+05, 3.44607199e+05, 6.98028856e+04, 6.48546405e+04, 2.16622010e+05, 1.29856464e+00, 9.98658014e-01]

            elif self.file_identifier == 'nerix_160418_1531':
                self.d_fit_files['settings'] = [num_bins_nerix, -5e5, 4.e6]
                self.a_free_par_guesses = [7.10181973e-01, 8.91060503e+05, 6.44040319e+05, 3.33974878e+05, 1.47883993e+05, 4.26489189e+04, 1.62379092e+05, 2.86908989e+00, 1.00472883e+00]
            
            
            else:
                print '\n\nSettings do not exist for given setup: %s\n\n' % (self.file_identifier)
                sys.exit()
        
        
        
        
        self.d_best_fit_pars = {}
        # this is reserved for MCMC fitting only!
        if self.file_identifier == '0062_0061':
            self.d_best_fit_pars['cascade'] = [9.66495591e-01, 5.40094282e+00, 2.64956897e-01, 6.74034387e-01, -4.51191570e+04, 2.49419916e+05, 3.24546331e+05, 3.62771594e-01, 1.11559373e+00, 9.98569964e-01] # old but better chi2
            #self.d_best_fit_pars['cascade'] = [9.65740894e-01, 5.40141879e+00, 2.66216741e-01, 6.74060683e-01, -4.73983359e+04, 2.48767207e+05, 3.27797485e+05, 3.57030065e-01, 1.12838756e+00, 1.00161906e+00]
            self.d_best_fit_pars['gaussian'] = [9.27455886e-01, 6.02107083e+06, 1.78588913e+06, 2.03499540e+06, 1.53514454e+06, -3.03062163e+04, 2.54759740e+05, 3.72035117e+05, 3.00810065e-01, 1.14126208e+00, 9.99234449e-01]

        elif self.file_identifier == '0066_0065':
            self.d_best_fit_pars['cascade'] = [9.98697998e-01, 4.48510356e+00, 2.78206057e-01, 7.77850195e-01, -7.43767176e+03, 2.63083714e+05, 8.55090511e+05, 1.88586817e-01, 1.03808845e+00, 9.98859784e-01]
            self.d_best_fit_pars['gaussian'] = [9.80547588e-01, 3.67228734e+06, 1.03119880e+06, 1.08944376e+06, 6.42498405e+05, -2.46409355e+04, 2.64242483e+05, 1.16769239e+06, 1.84882188e-01, 1.07778134e+00, 9.99584994e-01]

        elif self.file_identifier == '0067_0068':
            self.d_best_fit_pars['cascade'] = [9.76261165e-01, 4.44232993e+00, 2.60179847e-01, 7.49635706e-01, -4.73605202e+03, 2.62854493e+05, 2.16243930e+05, 1.22287727e-01, 1.09768310e+00, 9.99442046e-01]
            self.d_best_fit_pars['gaussian'] = [9.49051753e-01, 2.14215698e+06, 6.17409360e+05, 5.87860566e+05, 2.01742826e+05, 7.93001099e+01, 2.61402293e+05, 1.32714317e+06, 1.19555846e-01, 1.07471275e+00, 1.00000740e+00]

        elif self.file_identifier == '0071_0072':
            self.d_best_fit_pars['cascade'] = [9.67527817e-01, 5.38973278e+00, 2.68471032e-01, 7.01936600e-01, -5.80387843e+04, 2.46783958e+05, 5.63181912e+05, 3.92807365e-01, 1.12080939e+00, 9.99694760e-01]
            self.d_best_fit_pars['gaussian'] = [9.45703228e-01, 9.49909717e+06, 2.85991303e+06, 2.41602972e+06, 2.00493046e+06, -4.18589563e+04, 2.39416597e+05, 8.49605025e+05, 3.54550632e-01, 1.14361339e+00, 1.00123126e+00]

        elif self.file_identifier == '0073_0074':
            self.d_best_fit_pars['cascade'] = [9.64079808e-01, 4.67173126e+00, 7.34207820e-01, 8.15681566e-01, -7.61481628e+04, 2.49755333e+05, 6.64250744e+05, 5.75134022e-01, 1.97999336e+00, 9.98884226e-01]
            self.d_best_fit_pars['gaussian'] = [9.39843324e-01, 9.61129222e+06, 2.85479094e+06, 2.66788862e+06, 2.63479342e+06, -7.81465326e+04, 2.39556430e+05, 6.19840022e+05, 4.26982056e-01, 2.00939325e+00, 9.99757442e-01]

        elif self.file_identifier == 'nerix_160418_1523':
            #self.d_best_fit_pars['cascade'] =
            self.d_best_fit_pars['gaussian'] = [7.21372889e-01, 9.85172045e+05, 4.88673583e+05, 3.05606474e+05, 2.08713927e+05, 6.82181423e+04, 2.19043515e+05, 1.31150122e+00, 9.94787066e-01]

        elif self.file_identifier == 'nerix_160418_1531':
            #self.d_best_fit_pars['cascade'] =
            #self.d_best_fit_pars['gaussian'] =
            pass
        
        
        
        self.d_fit_files['bin_edges'] = np.linspace(self.d_fit_files['settings'][1], self.d_fit_files['settings'][2], self.d_fit_files['settings'][0]+1) # need +1 for bin edges
        self.d_fit_files['bin_width'] = self.d_fit_files['bin_edges'][1] - self.d_fit_files['bin_edges'][0]
        self.d_fit_files['bin_centers'] = np.linspace(self.d_fit_files['settings'][1]+self.d_fit_files['bin_width']/2., self.d_fit_files['settings'][2]-self.d_fit_files['bin_width']/2., self.d_fit_files['settings'][0])
        if self.b_making_comparison_plots:
            num_bins_plots = 5*self.d_fit_files['settings'][0]
            self.d_fit_files['bin_edges_plots'] = np.linspace(self.d_fit_files['settings'][1], self.d_fit_files['settings'][2], num_bins_plots+1) # need +1 for bin edges
            self.d_fit_files['bin_width_plots'] = self.d_fit_files['bin_edges_plots'][1] - self.d_fit_files['bin_edges_plots'][0]
            self.d_fit_files['bin_centers_plots'] = np.linspace(self.d_fit_files['settings'][1]+self.d_fit_files['bin_width_plots']/2., self.d_fit_files['settings'][2]-self.d_fit_files['bin_width_plots']/2., num_bins_plots)
        
        
       
        
        #self.d_fit_files['bin_edges'] = astroML.density_estimation.bayesian_blocks(self.d_fit_files['a_integral'])
        self.d_fit_files['hist'], dummy = np.histogram(self.d_fit_files['a_integral'], bins=self.d_fit_files['bin_edges'])

        
        
        
        # set save and load paths
        if b_use_cascade_model:
            self.s_base_save_name = 'cascade_model_fit'
            self.dict_filename = 'sampler_dictionary.p'
            self.acceptance_filename = 'acceptance_fraction.p'
        else:
            self.s_base_save_name = 'gaussian_model_fit'
            self.dict_filename = 'sampler_dictionary_gm.p'
            self.acceptance_filename = 'acceptance_fraction_gm.p'
        self.s_directory_save_name = 'results/%s/' % (self.file_identifier)
        self.s_directory_save_plots_name = 'plots/%s/' % (self.file_identifier)
        
    
    
        self.b_suppress_likelihood = False
    
    
        #print self.d_fit_files['spe']['hist']
        
        
        
    def prior_between_0_and_1(self, parameter_to_examine):
        if 0 < parameter_to_examine < 1:
            return 0
        else:
            return -np.inf



    def prior_greater_than_0(self, parameter_to_examine):
        if parameter_to_examine > 0:
            return 0
        else:
            return -np.inf



    def prior_uc_bkg(self, bkg_mean, bkg_std):
        return (scipy.stats.norm.logpdf(bkg_mean, self.d_bkg_fit['bkg_mean'], self.d_bkg_fit['bkg_mean_unc']) + scipy.stats.norm.logpdf(bkg_std, self.d_bkg_fit['bkg_std'], self.d_bkg_fit['bkg_std_unc']))


            
    def cascade_model_ln_likelihood(self, a_parameters, b_return_chi2_ndf=False):
    
        #func_start_time = time.time()
        
        prob_hit_first, collection_efficiency, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, underamp_ionization_correction, poor_collection_ionization_correction, bkg_mean, bkg_std, mean_num_incident_photons, prob_pe_from_photocathode, scale_par = a_parameters
        bkg_exp, prob_exp_bkg = 1, 1e-20

        ln_prior = 0
        ln_likelihood = 0

        ln_prior += self.prior_between_0_and_1(prob_hit_first)
        ln_prior += self.prior_between_0_and_1(collection_efficiency)
        ln_prior += self.prior_between_0_and_1(underamp_ionization_correction)
        ln_prior += self.prior_greater_than_0(mean_num_incident_photons)
        ln_prior += self.prior_between_0_and_1(prob_pe_from_photocathode)
        ln_prior += self.prior_between_0_and_1(poor_collection_ionization_correction)
        ln_prior += self.prior_greater_than_0(mean_e_from_dynode)
        ln_prior += self.prior_greater_than_0(width_e_from_dynode)
        ln_prior += self.prior_greater_than_0(bkg_std)

        if not self.b_nerix_file:
            ln_prior += self.prior_uc_bkg(bkg_mean, bkg_std)
        
        if not self.b_nerix_file:
            if poor_collection_ionization_correction < 0.7 or underamp_ionization_correction < 0.7:
                return -np.inf
        
        approximate_spe_mean = (mean_e_from_dynode*probability_electron_ionized)**12.
        #print approximate_spe_mean
        
        if bkg_exp > approximate_spe_mean:
            return -np.inf

        if not np.isfinite(ln_prior):
            return -np.inf

        if self.b_nerix_file:
            if width_e_from_dynode > 0.3*mean_e_from_dynode:
                return -np.inf

        a_hist = np.zeros(len(self.d_fit_files['hist']), dtype=np.float32)
        
        mean_num_incident_photons = np.asarray(mean_num_incident_photons, dtype=np.float32)
        prob_pe_from_photocathode = np.asarray(prob_pe_from_photocathode, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        collection_efficiency = np.asarray(collection_efficiency, dtype=np.float32)
        mean_e_from_dynode = np.asarray(mean_e_from_dynode, dtype=np.float32)
        width_e_from_dynode = np.asarray(width_e_from_dynode, dtype=np.float32)
        probability_electron_ionized = np.asarray(probability_electron_ionized, dtype=np.float32)
        underamp_ionization_correction = np.asarray(underamp_ionization_correction, dtype=np.float32)
        poor_collection_ionization_correction = np.asarray(poor_collection_ionization_correction, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)

        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)

        num_bins = np.asarray(len(self.d_fit_files['hist']), dtype=np.int32)
        bin_edges = np.asarray(self.d_fit_files['bin_edges'], dtype=np.float32)
        
        
        l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist), drv.In(mean_num_incident_photons), drv.In(prob_pe_from_photocathode), drv.In(prob_hit_first), drv.In(collection_efficiency), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(underamp_ionization_correction), drv.In(poor_collection_ionization_correction), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges), drv.In(self.num_dynodes), self.gpu_dynode_chain]

        #gpu_start_time = time.time()
    
        #start_time_mpe1 = time.time()
        self.gpu_cascade_model(*l_args_gpu, **self.d_gpu_scale)
        #print 'Time for MPE1 call: %f s' % (time.time() - start_time_spe)
        a_model = np.asarray(a_hist, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/np.sum(a_hist)*scale_par

        #print 'gpu time: %f' % (time.time() - gpu_start_time)

        #ll_start_time = time.time()

        ln_likelihood += poisson_binned_likelihood(a_model, self.d_fit_files['hist'])

        #print 'ln(l) time: %f' % (time.time() - ll_start_time)

        total_ln_likelihood = ln_prior + ln_likelihood

        if np.isnan(total_ln_likelihood):
            return -np.inf
            
        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor

        #print total_ln_likelihood
        #print np.sum( (a_model - self.d_fit_files['hist'])**2/self.d_fit_files['hist'] )

        #print 'full time: %f' % (time.time() - func_start_time)
        
        if not b_return_chi2_ndf:
            return total_ln_likelihood
        else:
            a_data = self.d_fit_files['hist']
            mask = (a_model > 0) & (a_data > 0)
            #mask_high =  (a_model[mask] - a_data[mask])**2. / a_data[mask] > 10
            #print a_model[mask_high]
            #print a_data[mask_high]
            #return np.sum((a_model[mask] - a_data[mask])**2. / a_model[mask]), num_bins - len(a_parameters)
            return np.sum((a_model[mask] - a_data[mask])**2. / a_data[mask]), num_bins - len(a_parameters)
            #return 2*np.sum(a_data[mask]*np.log(a_data[mask] / a_model[mask]) + a_model[mask] - a_data[mask]), num_bins - len(a_parameters)
            
        
        
    def perform_ad_test_cascade(self, a_parameters, num_bootstrap_samples=1e4, num_bootstraps=200):
        prob_hit_first, collection_efficiency, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, underamp_ionization_correction, poor_collection_ionization_correction, bkg_mean, bkg_std, mean_num_incident_photons, prob_pe_from_photocathode, scale_par = a_parameters
        bkg_exp, prob_exp_bkg = 1, 1e-20


        a_integrals = np.full(self.num_mc_events, -1e10, dtype=np.float32)
        
        mean_num_incident_photons = np.asarray(mean_num_incident_photons, dtype=np.float32)
        prob_pe_from_photocathode = np.asarray(prob_pe_from_photocathode, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        collection_efficiency = np.asarray(collection_efficiency, dtype=np.float32)
        mean_e_from_dynode = np.asarray(mean_e_from_dynode, dtype=np.float32)
        width_e_from_dynode = np.asarray(width_e_from_dynode, dtype=np.float32)
        probability_electron_ionized = np.asarray(probability_electron_ionized, dtype=np.float32)
        underamp_ionization_correction = np.asarray(underamp_ionization_correction, dtype=np.float32)
        poor_collection_ionization_correction = np.asarray(poor_collection_ionization_correction, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)

        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)

        l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_integrals), drv.In(mean_num_incident_photons), drv.In(prob_pe_from_photocathode), drv.In(prob_hit_first), drv.In(collection_efficiency), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(underamp_ionization_correction), drv.In(poor_collection_ionization_correction), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(self.num_dynodes), self.gpu_dynode_chain]
    
    
        self.gpu_cascade_model_arrays(*l_args_gpu, **self.d_gpu_scale)
    
        # now that we have the arrays for data and mc we need to bootstrap!
        num_bootstrap_samples = int(num_bootstrap_samples)
        num_to_choose_mc = int(2e5)
        l_adstat_data = np.zeros(num_bootstraps)
        l_adstat_synthetic = np.zeros(num_bootstraps)
        
        a_integrals = a_integrals[(a_integrals > self.d_fit_files['settings'][1]) & (a_integrals < self.d_fit_files['settings'][2])]
        a_integrals_data = self.d_fit_files['a_integral'][(self.d_fit_files['a_integral'] > self.d_fit_files['settings'][1]) & (self.d_fit_files['a_integral'] < self.d_fit_files['settings'][2])]
        #print len(a_integrals_data), len(a_integrals)
        
        
        print '\n\nPerforming AD Test...\n\n'
        
        for i in tqdm.tqdm(xrange(num_bootstraps)):
    
            idx_mc = np.random.choice(len(a_integrals), num_to_choose_mc, replace=True)
            idx_data = np.random.choice(len(a_integrals_data), num_bootstrap_samples, replace=True)
            idx_synthetic_data = np.random.choice(len(a_integrals), num_bootstrap_samples, replace=True)
            
            #print self.d_fit_files['a_integral'][idx_data]
            #f1, ax1 = plt.subplots(1)
            #ax1.hist(self.d_fit_files['a_integral'][idx_data], bins=50)
            #plt.show()

            adstat_data, pvalue_data = scipy.stats.ks_2samp(a_integrals[idx_mc], a_integrals_data[idx_data])
            adstat_synthetic_data, pvalue_synthetic_data = scipy.stats.ks_2samp(a_integrals[idx_mc], a_integrals[idx_synthetic_data])

            l_adstat_data[i] = adstat_data
            l_adstat_synthetic[i] = adstat_synthetic_data


        #print l_adstat_data
        #print l_adstat_synthetic
        print 'median AD Stat: %f' % (np.median(l_adstat_data))
        print 'Percentage synthetic above median: %.4f\n\n\n' % (len(l_adstat_synthetic[l_adstat_synthetic > np.median(l_adstat_data)]) / float(num_bootstraps))
    


    def perform_ad_test_gaussian(self, a_parameters, num_bootstrap_samples=1e4, num_bootstraps=200):
        prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = a_parameters
        bkg_exp, prob_exp_bkg = 1, 0.0001


        a_integrals = np.full(self.num_mc_events, -1e10, dtype=np.float32)
        
        mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        spe_mean = np.asarray(spe_mean, dtype=np.float32)
        spe_std = np.asarray(spe_std, dtype=np.float32)
        under_amp_mean = np.asarray(under_amp_mean, dtype=np.float32)
        under_amp_std = np.asarray(under_amp_std, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)
        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)
        
        num_bins = np.asarray(len(self.d_fit_files['hist']), dtype=np.int32)
        bin_edges = np.asarray(self.d_fit_files['bin_edges'], dtype=np.float32)
        
        
        l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_integrals), drv.In(mean_num_pe), drv.In(prob_hit_first), drv.In(spe_mean), drv.In(spe_std), drv.In(under_amp_mean), drv.In(under_amp_std), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg)]
    
    
        self.gpu_gaussian_model_arrays(*l_args_gpu, **self.d_gpu_scale)
    
        # now that we have the arrays for data and mc we need to bootstrap!
        num_bootstrap_samples = int(num_bootstrap_samples)
        num_to_choose_mc = int(2e5)
        l_adstat_data = np.zeros(num_bootstraps)
        l_adstat_synthetic = np.zeros(num_bootstraps)
        
        a_integrals = a_integrals[(a_integrals > self.d_fit_files['settings'][1]) & (a_integrals < self.d_fit_files['settings'][2])]
        a_integrals_data = self.d_fit_files['a_integral'][(self.d_fit_files['a_integral'] > self.d_fit_files['settings'][1]) & (self.d_fit_files['a_integral'] < self.d_fit_files['settings'][2])]
        #print len(a_integrals_data), len(a_integrals)
        
        
        print '\n\nPerforming AD Test...\n\n'
        
        for i in tqdm.tqdm(xrange(num_bootstraps)):
    
            idx_mc = np.random.choice(len(a_integrals), num_to_choose_mc, replace=True)
            idx_data = np.random.choice(len(a_integrals_data), num_bootstrap_samples, replace=True)
            idx_synthetic_data = np.random.choice(len(a_integrals), num_bootstrap_samples, replace=True)
            
            #print self.d_fit_files['a_integral'][idx_data]
            #f1, ax1 = plt.subplots(1)
            #ax1.hist(self.d_fit_files['a_integral'][idx_data], bins=50)
            #plt.show()

            adstat_data, pvalue_data = scipy.stats.ks_2samp(a_integrals[idx_mc], a_integrals_data[idx_data])
            adstat_synthetic_data, pvalue_synthetic_data = scipy.stats.ks_2samp(a_integrals[idx_mc], a_integrals[idx_synthetic_data])

            l_adstat_data[i] = adstat_data
            l_adstat_synthetic[i] = adstat_synthetic_data


        #print l_adstat_data
        #print l_adstat_synthetic
        print 'median AD Stat: %f' % (np.median(l_adstat_data))
        print 'Percentage synthetic above median: %.4f\n\n\n' % (len(l_adstat_synthetic[l_adstat_synthetic > np.median(l_adstat_data)]) / float(num_bootstraps))
        
        
        
        
    def create_fake_data_cascade(self, a_parameters):
        prob_hit_first, collection_efficiency, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, underamp_ionization_correction, poor_collection_ionization_correction, bkg_mean, bkg_std, mean_num_incident_photons, prob_pe_from_photocathode, scale_par = a_parameters
        bkg_exp, prob_exp_bkg = 1, 1e-20


        a_integrals = np.zeros(self.num_mc_events, dtype=np.float32)
        
        mean_num_incident_photons = np.asarray(mean_num_incident_photons, dtype=np.float32)
        prob_pe_from_photocathode = np.asarray(prob_pe_from_photocathode, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        collection_efficiency = np.asarray(collection_efficiency, dtype=np.float32)
        mean_e_from_dynode = np.asarray(mean_e_from_dynode, dtype=np.float32)
        width_e_from_dynode = np.asarray(width_e_from_dynode, dtype=np.float32)
        probability_electron_ionized = np.asarray(probability_electron_ionized, dtype=np.float32)
        underamp_ionization_correction = np.asarray(underamp_ionization_correction, dtype=np.float32)
        poor_collection_ionization_correction = np.asarray(poor_collection_ionization_correction, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)

        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)

        l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_integrals), drv.In(mean_num_incident_photons), drv.In(prob_pe_from_photocathode), drv.In(prob_hit_first), drv.In(collection_efficiency), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(underamp_ionization_correction), drv.In(poor_collection_ionization_correction), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(self.num_dynodes), self.gpu_dynode_chain]
    
    
        self.gpu_cascade_model_arrays(*l_args_gpu, **self.d_gpu_scale)
        
        #plt.hist(a_integrals, bins=50)
        #plt.show()
    
        pickle.dump(a_integrals, open('./data/fake.p', 'w'))
    
    
    
    def run_mcmc(self, num_walkers=32, num_steps=2000, threads=1):
        
        l_value_guesses = self.a_free_par_guesses
        l_std_guesses = 0.01*np.asarray(l_value_guesses)

        
        
        
        if not os.path.exists(self.s_directory_save_name):
            os.makedirs(self.s_directory_save_name)
        num_dim = len(l_value_guesses)
        
        
        loaded_prev_sampler = False
        try:
            # two things will fail potentially
            # 1. open if file doesn't exist
            # 2. posWalkers load since initialized to None

            with open(self.s_directory_save_name + self.dict_filename, 'r') as f_prev_sampler:

                d_sampler = pickle.load(f_prev_sampler)
                prevSampler = d_sampler[num_walkers][-1]


                # need to load in weird way bc can't pickle
                # ensembler object
                a_starting_pos = prevSampler['_chain'][:,-1,:]
                random_state = prevSampler['_random']
            loaded_prev_sampler = True
            print '\nSuccessfully loaded previous chain!\n'
        except:
            print '\nCould not load previous sampler or none existed - starting new sampler.\n'

        if not loaded_prev_sampler:

            a_starting_pos = emcee.utils.sample_ball(l_value_guesses, l_std_guesses, size=num_walkers)

            random_state = None

            # create file if it doesn't exist
            if not os.path.exists(self.s_directory_save_name + self.dict_filename):
                with open(self.s_directory_save_name + self.dict_filename, 'w') as f_prev_sampler:
                    d_sampler = {}

                    d_sampler[num_walkers] = []

                    pickle.dump(d_sampler, f_prev_sampler)
            else:
                with open(self.s_directory_save_name + self.dict_filename, 'r') as f_prev_sampler:
                    d_sampler = pickle.load(f_prev_sampler)
                with open(self.s_directory_save_name + self.dict_filename, 'w') as f_prev_sampler:

                    d_sampler[num_walkers] = []

                    pickle.dump(d_sampler, f_prev_sampler)
        
        


        if self.b_use_cascade_model:
            #sampler = emcee.EnsembleSampler(num_walkers, num_dim, self.cascade_model_ln_likelihood, threads=threads)
            sampler = emcee.DESampler(num_walkers, num_dim, self.cascade_model_ln_likelihood, threads=threads, autoscale_gamma=True)
        else:
            sampler = emcee.DESampler(num_walkers, num_dim, self.gaussian_model_ln_likelihood, threads=threads, autoscale_gamma=True)
        
        print '\n\nBeginning MCMC sampler\n\n'
        print '\nNumber of walkers * number of steps = %d * %d = %d function calls\n' % (num_walkers, num_steps, num_walkers*num_steps)
        start_time_mcmc = time.time()

        with click.progressbar(sampler.sample(a_starting_pos, iterations=num_steps, ), length=num_steps) as mcmc_sampler:
            for pos, lnprob, state in mcmc_sampler:
                pass

        total_time_mcmc = (time.time() - start_time_mcmc) / 3600.
        print '\n\n%d function calls took %.2f hours.\n\n' % (num_walkers*num_steps, total_time_mcmc)


        dictionary_for_sampler = sampler.__dict__
        if 'lnprobfn' in dictionary_for_sampler:
            del dictionary_for_sampler['lnprobfn']
        if 'pool' in dictionary_for_sampler:
            del dictionary_for_sampler['pool']

        with open(self.s_directory_save_name + self.dict_filename, 'r') as f_prev_sampler:
            d_sampler = pickle.load(f_prev_sampler)
        #f_prev_sampler.close()

        f_prev_sampler = open(self.s_directory_save_name + self.dict_filename, 'w')

        d_sampler[num_walkers].append(sampler.__dict__)

        pickle.dump(d_sampler, f_prev_sampler)
        f_prev_sampler.close()



        #sampler.run_mcmc(posWalkers, numSteps) # shortcut of above method
        pickle.dump(sampler.acceptance_fraction, open(self.s_directory_save_name + self.acceptance_filename, 'w'))
    
    
    
    def draw_cascade_model_fit(self, a_parameters, name_for_save=''):
     
        prob_hit_first, collection_efficiency, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, underamp_ionization_correction, poor_collection_ionization_correction, bkg_mean, bkg_std, mean_num_incident_photons, prob_pe_from_photocathode, scale_par = a_parameters
        bkg_exp, prob_exp_bkg = 1, 1e-20
        
        
        
        a_hist = np.zeros(len(self.d_fit_files['hist']), dtype=np.float32)
        
        mean_num_incident_photons = np.asarray(mean_num_incident_photons, dtype=np.float32)
        prob_pe_from_photocathode = np.asarray(prob_pe_from_photocathode, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        collection_efficiency = np.asarray(collection_efficiency, dtype=np.float32)
        mean_e_from_dynode = np.asarray(mean_e_from_dynode, dtype=np.float32)
        width_e_from_dynode = np.asarray(width_e_from_dynode, dtype=np.float32)
        probability_electron_ionized = np.asarray(probability_electron_ionized, dtype=np.float32)
        underamp_ionization_correction = np.asarray(underamp_ionization_correction, dtype=np.float32)
        poor_collection_ionization_correction = np.asarray(poor_collection_ionization_correction, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)
        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)
        
        
        num_bins = np.asarray(len(self.d_fit_files['hist']), dtype=np.int32)
        bin_edges = np.asarray(self.d_fit_files['bin_edges'], dtype=np.float32)
        
        
        l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist), drv.In(mean_num_incident_photons), drv.In(prob_pe_from_photocathode), drv.In(prob_hit_first), drv.In(collection_efficiency), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(underamp_ionization_correction), drv.In(poor_collection_ionization_correction), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges), drv.In(self.num_dynodes), self.gpu_dynode_chain]
    
    
        #start_time_mpe1 = time.time()
        self.gpu_cascade_model(*l_args_gpu, **self.d_gpu_scale)
        #print 'Time for MPE1 call: %f s' % (time.time() - start_time_spe)
        a_model = np.asarray(a_hist, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/np.sum(a_hist)*scale_par
        
        
        f1, (ax1) = plt.subplots(1)
        #ax1.set_yscale('log', nonposx='clip')
    
        a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = neriX_analysis.prepare_hist_arrays_for_plotting(self.d_fit_files['hist'], self.d_fit_files['bin_edges'])
        ax1.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='b', fmt='.')
        a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = neriX_analysis.prepare_hist_arrays_for_plotting(a_model, self.d_fit_files['bin_edges'])
        ax1.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='r', fmt='.')
        
        
        if not name_for_save == '':
            pickle.dump(a_model, open('./%s.p' % (name_for_save), 'w'))
    
        #plt.show()
    
    
    
    def draw_model_with_error_bands(self, num_walkers, num_steps_to_include=20, b_save_dict_for_plot=False):
        if not self.b_making_comparison_plots:
            print 'Must instantiate class such that it is ready for making comparison plots\n.'
            sys.exit()
        
        print '\n\nAdded single PE spectrum with mean and variance output in this function too\n\n'
        
        num_dim = len(self.a_free_par_guesses)
        
        sPathToFile = self.s_directory_save_name + self.dict_filename
        
        if os.path.exists(sPathToFile):
            dSampler = pickle.load(open(sPathToFile, 'r'))
            l_chains = []
            for sampler in dSampler[num_walkers]:
                l_chains.append(sampler['_chain'])

            a_sampler = np.concatenate(l_chains, axis=1)

            print 'Successfully loaded sampler!'
        else:
            print sPathToFile
            print 'Could not find file!'
            sys.exit()
        
        
        a_sampler = a_sampler[:, -num_steps_to_include:, :num_dim].reshape((-1, num_dim))
        
        num_independent_samples = num_walkers*num_steps_to_include
        
        num_bins_plots = len(self.d_fit_files['bin_centers_plots'])
        a_histograms = np.zeros((num_independent_samples, num_bins_plots), dtype=np.float32)
        a_histograms_pure = np.zeros((num_independent_samples, num_bins_plots), dtype=np.float32)
        
        a_means = np.zeros(num_independent_samples)
        a_stds = np.zeros(num_independent_samples)
        
        
        print 'Generating spectra from draws from the posterior'
        #print 'For neriX fixing photoionization of first dynode to zero (confirm with corner plot)!\n\n'
        for i in tqdm.tqdm(xrange(num_independent_samples)):
            #i = -1
            
            #prob_hit_first, collection_efficiency, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, underamp_ionization_correction, poor_collection_ionization_correction, bkg_mean, bkg_std, mean_num_incident_photons, prob_pe_from_photocathode, scale_par = a_best_fit_pars
            
            if self.b_use_cascade_model:
                
                collection_efficiency = a_sampler[i][1]
                mean_e_from_dynode = a_sampler[i][2]
                width_e_from_dynode = a_sampler[i][3]
                probability_electron_ionized = a_sampler[i][4]
                underamp_ionization_correction = a_sampler[i][5]
                poor_collection_ionization_correction = a_sampler[i][6]
            
                if self.filename == 'fake' or not self.b_nerix_file:
                    prob_hit_first = a_sampler[i][0]
                    bkg_mean = a_sampler[i][7]
                    bkg_std = a_sampler[i][8]
                    bkg_exp = 1
                    prob_exp_bkg = 1e-20
                    mean_num_incident_photons = a_sampler[i][9]
                    prob_pe_from_photocathode = a_sampler[i][10]
                    scale_par = a_sampler[i][11]
                else:
                    #prob_hit_first = 0.
                    prob_hit_first = a_sampler[i][0]
                    bkg_mean = a_sampler[i][7]
                    bkg_std = a_sampler[i][8]
                    bkg_exp = 1
                    prob_exp_bkg = 1e-20
                    mean_num_incident_photons = a_sampler[i][9]
                    prob_pe_from_photocathode = a_sampler[i][10]
                    scale_par = a_sampler[i][11]
            
            else:
                prob_hit_first = a_sampler[i][0]
                spe_mean = a_sampler[i][1]
                spe_std = a_sampler[i][2]
                under_amp_mean = a_sampler[i][3]
                under_amp_std = a_sampler[i][4]
                
                if not self.file_identifier[:5] == 'nerix':
                    bkg_mean = a_sampler[i][5]
                    bkg_std = a_sampler[i][6]
                    bkg_exp = 1
                    prob_exp_bkg = 0.0001
                    mean_num_pe = a_sampler[i][9]
                    scale_par = a_sampler[i][10]
                else:
                    bkg_mean = a_sampler[i][5]
                    bkg_std = a_sampler[i][6]
                    bkg_exp = 1
                    prob_exp_bkg = 0.0001
                    mean_num_pe = a_sampler[i][7]
                    scale_par = a_sampler[i][8]
            
            
            
            a_hist = np.zeros(num_bins_plots, dtype=np.float32)
            
            a_hist_pure = np.zeros(num_bins_plots, dtype=np.float32)
            
            
            
            num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
            prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
            
            if self.b_use_cascade_model:
                mean_num_incident_photons = np.asarray(mean_num_incident_photons, dtype=np.float32)
                mean_e_from_dynode = np.asarray(mean_e_from_dynode, dtype=np.float32)
                width_e_from_dynode = np.asarray(width_e_from_dynode, dtype=np.float32)
                probability_electron_ionized = np.asarray(probability_electron_ionized, dtype=np.float32)
                collection_efficiency = np.asarray(collection_efficiency, dtype=np.float32)
                underamp_ionization_correction = np.asarray(underamp_ionization_correction, dtype=np.float32)
                poor_collection_ionization_correction = np.asarray(poor_collection_ionization_correction, dtype=np.float32)
                prob_pe_from_photocathode = np.asarray(prob_pe_from_photocathode, dtype=np.float32)
            else:
                mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
                spe_mean = np.asarray(spe_mean, dtype=np.float32)
                spe_std = np.asarray(spe_std, dtype=np.float32)
                under_amp_mean = np.asarray(under_amp_mean, dtype=np.float32)
                under_amp_std = np.asarray(under_amp_std, dtype=np.float32)
            
            
            bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
            bkg_std = np.asarray(bkg_std, dtype=np.float32)
            bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
            prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)
            
            num_bins = np.asarray(num_bins_plots, dtype=np.int32)
            bin_edges = np.asarray(self.d_fit_files['bin_edges_plots'], dtype=np.float32)
            
            
            if self.b_use_cascade_model:
                l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist), drv.In(mean_num_incident_photons), drv.In(prob_pe_from_photocathode), drv.In(prob_hit_first), drv.In(collection_efficiency), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(underamp_ionization_correction), drv.In(poor_collection_ionization_correction), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges), drv.In(self.num_dynodes), self.gpu_dynode_chain]
                self.gpu_cascade_model(*l_args_gpu, **self.d_gpu_scale)
            else:
                l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist), drv.In(mean_num_pe), drv.In(prob_hit_first), drv.In(spe_mean), drv.In(spe_std), drv.In(under_amp_mean), drv.In(under_amp_std), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges)]
                self.gpu_gaussian_model(*l_args_gpu, **self.d_gpu_scale)

        
            #start_time_mpe1 = time.time()
            #print 'Time for MPE1 call: %f s' % (time.time() - start_time_spe)
            a_model = np.asarray(a_hist, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/np.sum(a_hist)*self.d_fit_files['bin_width']/self.d_fit_files['bin_width_plots']*scale_par
            
            a_histograms[i] = a_model
        
        
        
            # gather inputs for pure spec
            if self.b_use_cascade_model:
                l_pure_spec = [self.rng_states, drv.In(num_trials), drv.InOut(a_hist_pure), drv.In(np.asarray(1, dtype=np.int32)), drv.In(np.asarray(prob_pe_from_photocathode/(prob_pe_from_photocathode+prob_hit_first), dtype=np.float32)), drv.In(np.asarray(prob_hit_first/(prob_pe_from_photocathode+prob_hit_first), dtype=np.float32)), drv.In(collection_efficiency), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(underamp_ionization_correction), drv.In(poor_collection_ionization_correction), drv.In(num_bins), drv.In(bin_edges), drv.In(self.num_dynodes), self.gpu_dynode_chain]
                self.gpu_pure_cascade_spectrum(*l_pure_spec, **self.d_gpu_scale)
            else:
                l_pure_spec = [self.rng_states, drv.In(num_trials), drv.InOut(a_hist_pure), drv.In(np.asarray(1, dtype=np.int32)), drv.In(prob_hit_first), drv.In(spe_mean), drv.In(spe_std), drv.In(under_amp_mean), drv.In(under_amp_std), drv.In(num_bins), drv.In(bin_edges)]
                self.gpu_pure_gaussian_spectrum(*l_pure_spec, **self.d_gpu_scale)

            
            
            try:
                a_means[i], a_stds[i] = weighted_avg_and_std(self.d_fit_files['bin_centers_plots'], a_hist_pure)
            except:
                print 'Forced to skip pure spectrum...'
            a_pure_single_spec = np.asarray(a_hist_pure, dtype=np.float32)/np.sum(a_hist_pure)
        
            a_histograms_pure[i] = a_pure_single_spec
        
        
        
        a_one_sigma_below = np.zeros(num_bins_plots, dtype=np.float32)
        a_one_sigma_above = np.zeros(num_bins_plots, dtype=np.float32)
        for bin in xrange(num_bins):
            a_one_sigma_below[bin], a_one_sigma_above[bin] = np.nanpercentile(a_histograms[:, bin], [2.5, 97.5])
            

        a_one_sigma_below_pure = np.zeros(num_bins_plots, dtype=np.float32)
        a_one_sigma_above_pure = np.zeros(num_bins_plots, dtype=np.float32)
        #print 'DOING 1 SIGMA, NOT 2'
        for bin in xrange(num_bins):
            #a_one_sigma_below_pure[bin], a_one_sigma_above_pure[bin] = np.nanpercentile(a_histograms_pure[:, bin], [16., 84.])
            a_one_sigma_below_pure[bin], a_one_sigma_above_pure[bin] = np.nanpercentile(a_histograms_pure[:, bin], [2.5, 97.5])
        
        
            
        # get the median from a_sampler for each parameter
        
        
    
        f1, (ax1) = plt.subplots(1)
        ax1.set_yscale('log', nonposx='clip')
    
        a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = neriX_analysis.prepare_hist_arrays_for_plotting(self.d_fit_files['hist'], self.d_fit_files['bin_edges'])
        ax1.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='b', fmt='.')
        ax1.fill_between(self.d_fit_files['bin_centers_plots'], a_one_sigma_below, a_one_sigma_above, facecolor='red', alpha=0.5, interpolate=True)
        ax1.set_title('Integrated Charge Spectrum - %s' % (self.file_identifier))
        ax1.set_xlabel(r'Integrated Charge [$e^{-}$]')
        ax1.set_ylabel('Counts')
        ax1.set_ylim(20, max(a_y_values)*1.1)

        #ax1.set_ylim(np.min(a_y_values)/1.5, 2*np.max(a_y_values))
        
        
        

        s_mean_gain = 'Mean = %.2e +/- %.2e' % (np.average(a_means), np.std(a_means))
        s_rms_gain = 'RMS = %.2e +/- %.2e' % (np.average(a_stds), np.std(a_stds))
        
        f3, (ax3) = plt.subplots(1)
        ax3.fill_between(self.d_fit_files['bin_centers_plots'], a_one_sigma_below_pure, a_one_sigma_above_pure, facecolor='blue', alpha=0.5, interpolate=True)
        ax3.set_title('Single PE Spectrum - Best Fit')
        ax3.set_xlabel(r'Integrated Charge [$e^{-}$]')
        ax3.set_ylabel('Normalized Counts')


        ax1.text(0.7, 0.9, '%s\n%s' % (s_mean_gain, s_rms_gain), ha='center', va='center', transform=ax1.transAxes)
        ax3.text(0.7, 0.8, '%s\n%s' % (s_mean_gain, s_rms_gain), ha='center', va='center', transform=ax3.transAxes)
        
        if not os.path.exists(self.s_directory_save_plots_name):
            os.makedirs(self.s_directory_save_plots_name)
            
            
        if b_save_dict_for_plot:
            d_unc_band = {}
            d_unc_band['data_x'] = a_x_values
            d_unc_band['data_y'] = a_y_values
            d_unc_band['data_x_err_low'] = a_x_err_low
            d_unc_band['data_x_err_high'] = a_x_err_high
            d_unc_band['data_y_err_low'] = a_y_err_low
            d_unc_band['data_y_err_high'] = a_y_err_high
        
            d_unc_band['mc_x'] = self.d_fit_files['bin_centers_plots']
            d_unc_band['mc_one_sigma_below'] = a_one_sigma_below
            d_unc_band['mc_one_sigma_above'] = a_one_sigma_above
        
            pickle.dump(d_unc_band, open('./plot_dicts/%s_unc_band.p' % (self.file_identifier), 'w'))
        
        
            d_spe_response = {}
            d_spe_response['data_x'] = a_x_values
            d_spe_response['data_y'] = a_y_values
            d_spe_response['data_x_err_low'] = a_x_err_low
            d_spe_response['data_x_err_high'] = a_x_err_high
            d_spe_response['data_y_err_low'] = a_y_err_low
            d_spe_response['data_y_err_high'] = a_y_err_high
        
            d_spe_response['mc_x'] = self.d_fit_files['bin_centers_plots']
            d_spe_response['mc_one_sigma_below'] = a_one_sigma_below_pure
            d_spe_response['mc_one_sigma_above'] = a_one_sigma_above_pure
        
            pickle.dump(d_spe_response, open('./plot_dicts/%s_spe_response.p' % (self.file_identifier), 'w'))
        
            
            

        f1.savefig(self.s_directory_save_plots_name + self.s_base_save_name + '_full_%s.png' % (self.file_identifier))
        f3.savefig(self.s_directory_save_plots_name + self.s_base_save_name + '_pure_%s.png' % (self.file_identifier))

    
        #plt.show()



    def draw_cascade_model_corner_plot(self, num_walkers, num_steps_to_include):
        
        l_labels_for_corner_plot = ['p_hit_first_dynode', 'electrons_per_dynode', 'p_e_freed', 'bkg_mean', 'bkg_std', 'bkg_exp', 'p_exp_bkg', 'mean_num_pe']
        num_dim = len(l_labels_for_corner_plot)
        
        sPathToFile = self.s_directory_save_name + self.dict_filename
        
        if os.path.exists(sPathToFile):
            dSampler = pickle.load(open(sPathToFile, 'r'))
            l_chains = []
            for sampler in dSampler[num_walkers]:
                l_chains.append(sampler['_chain'])

            a_sampler = np.concatenate(l_chains, axis=1)

            print 'Successfully loaded sampler!'
        else:
            print sPathToFile
            print 'Could not find file!'
            sys.exit()
        
        a_sampler = a_sampler[:, -num_steps_to_include:, :num_dim].reshape((-1, num_dim))
        
        print 'Starting corner plot...\n'
        start_time = time.time()
        fig = corner.corner(a_sampler, labels=l_labels_for_corner_plot, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3e', title_kwargs={"fontsize": 12})
        print 'Corner plot took %.3f minutes.\n\n' % ((time.time()-start_time)/60.)
        
        if not os.path.exists(self.s_directory_save_plots_name):
            os.makedirs(self.s_directory_save_plots_name)

        fig.savefig(self.s_directory_save_plots_name + self.s_base_save_name + '_corner_%s.png' % (self.file_identifier))
        
        try:
            print emcee.autocorr.integrated_time(np.mean(a_sampler, axis=0), axis=0,
                                        low=10, high=None, step=1, c=2,
                                        fast=False)
        except:
            print 'Chain too short to find autocorrelation time!'




    def draw_model_fit_with_peaks(self, num_walkers, num_steps_to_include, num_steps_to_pull_from=500, b_save_dict_for_plot=False):
        
        num_dim =len(self.a_free_par_guesses)
        
        sPathToFile = self.s_directory_save_name + self.dict_filename
        
        if os.path.exists(sPathToFile):
            dSampler = pickle.load(open(sPathToFile, 'r'))
            l_chains = []
            l_ln_likelihoods = []
            for sampler in dSampler[num_walkers]:
                l_chains.append(sampler['_chain'])
                l_ln_likelihoods.append(sampler['_lnprob'])

            a_sampler = np.concatenate(l_chains, axis=1)
            a_full_ln_likelihood = np.concatenate(l_ln_likelihoods, axis=1)

            print 'Successfully loaded sampler!'
        else:
            print sPathToFile
            print 'Could not find file!'
            sys.exit()
        
        
        """
        a_sampler = a_sampler[:num_walkers, -num_steps_to_include:, :].reshape((-1, num_dim))
        #a_sampler = a_sampler[:, -num_steps_to_include:, :].reshape((-1, num_dim))
        a_medians = np.median(a_sampler, axis=0)


        l_num_pe = [0, 1, 2, 3, 4, 5, 6]
        l_colors = ['r', 'b', 'g', 'c', 'y', 'm', 'brown']
        prob_hit_first, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = a_medians
        """
        
        """
        a_sampler = a_sampler[:, -num_steps_to_include:, :].reshape((-1, num_dim))
        dd_hist, l_bins = np.histogramdd(a_sampler, bins=5)
        
        
        l_max_bins = np.unravel_index(dd_hist.argmax(), dd_hist.shape)
        
        l_bin_centers = [0 for i in xrange(len(l_max_bins))]
        # find bin centers from max
        for i in xrange(len(l_max_bins)):
            l_bin_centers[i] = (l_bins[i][l_max_bins[i]+1] + l_bins[i][l_max_bins[i]]) / 2.
        
        l_num_pe = [0, 1, 2, 3, 4, 5, 6]
        l_colors = ['r', 'b', 'g', 'c', 'y', 'm', 'brown']
        prob_hit_first, mean_e_from_dynode, probability_electron_ionized, bkg_mean, bkg_std, mean_num_pe, scale_par = l_bin_centers
        """
        
        """
        max_num_events_for_kde = 5e4
        assert num_steps_to_include*num_walkers < max_num_events_for_kde, 'Using KDE to estimate maximum in full space so must use less than %d events for time constraints.\n' % (int(max_num_events_for_kde))
        
        a_sampler = a_sampler.reshape(-1, num_dim)
        total_length_sampler = a_sampler.shape[0]
        a_partial_sampler = np.zeros((num_steps_to_include*num_walkers, num_dim))
        for i in tqdm.tqdm(xrange(num_steps_to_include*num_walkers)):
            a_partial_sampler[i, :] = a_sampler[(-(np.random.randint(1, num_steps_to_pull_from*num_walkers) % total_length_sampler)), :]
        #a_partial_sampler = a_sampler[:, -num_steps_to_include:, :].reshape((-1, num_dim))
        
        scaler = preprocessing.StandardScaler()
        scaler.fit(a_partial_sampler)
        a_scaled_samples = scaler.transform(a_partial_sampler)

        #print a_sampler[:,1:3]
        #print a_scaled_samples

        # find the best fit bandwith since this allows us
        # to play with bias vs variance
        grid = grid_search.GridSearchCV(neighbors.KernelDensity(), {'bandwidth':np.linspace(0.01, 2., 20)}, cv=4, verbose=1, n_jobs=4)
        print '\nDetermining best bandwidth...\n'
        grid.fit(a_scaled_samples)
        #print grid.best_estimator_

        kde = neighbors.KernelDensity(**grid.best_params_)
        kde.fit(a_scaled_samples)
        
        def func_for_minimizing_for_plot(a_parameters):
            a_scaled_parameters = scaler.transform(a_parameters)
            return -kde.score(a_scaled_parameters.reshape(1, -1))
        
        
        #a_bounds = [(0.75, 1), (1, 25), (0, 1.0), (1e3, 1e5), (5e4, 8e5), (0.6, 3.), (0.2, 2)]
        a_bounds = [np.percentile(a_sampler[:,i], [2, 98]) for i in xrange(num_dim)]
        result = op.differential_evolution(func_for_minimizing_for_plot, a_bounds, disp=True, maxiter=100, tol=0.01, popsize=20, polish=True)
        result = op.minimize(func_for_minimizing_for_plot, result.x)
        
        print result.x
        
        l_num_pe = [0, 1, 2, 3, 4, 5, 6]
        l_colors = ['r', 'b', 'g', 'c', 'y', 'm', 'brown']
        
        if self.b_use_cascade_model:
            prob_hit_first, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = result.x
        else:
            if not self.file_identifier[:5] == 'nerix':
                prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = result.x
            else:
                prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, mean_num_pe, scale_par = result.x
                bkg_exp = 1
                prob_exp_bkg = 0.0001
        """
        
        """
        a_best_fit_pars = np.zeros(num_dim)
        
        max_num_events_for_kde = 5e4
        assert num_steps_to_include*num_walkers < max_num_events_for_kde, 'Using KDE to estimate maximum in full space so must use less than %d events for time constraints.\n' % (int(max_num_events_for_kde))
        
        a_sampler = a_sampler.reshape(-1, num_dim)
        total_length_sampler = a_sampler.shape[0]
        a_partial_sampler = np.zeros((num_steps_to_include*num_walkers, num_dim))
        for i in tqdm.tqdm(xrange(num_steps_to_include*num_walkers)):
            a_partial_sampler[i, :] = a_sampler[(-(np.random.randint(1, num_steps_to_pull_from*num_walkers) % total_length_sampler)), :]
        #a_partial_sampler = a_sampler[:, -num_steps_to_include:, :].reshape((-1, num_dim))
        
        #print a_sampler[:,1:3]
        #print a_scaled_samples
        
        #plt.hist(a_partial_sampler)
        #plt.show()

        for i in xrange(num_dim):
            scaler = preprocessing.StandardScaler()
            scaler.fit(a_partial_sampler[:, i].reshape(-1, 1))
            a_scaled_samples = scaler.transform(a_partial_sampler[:, i].reshape(-1, 1))
        
        
            # find the best fit bandwith since this allows us
            # to play with bias vs variance
            grid = grid_search.GridSearchCV(neighbors.KernelDensity(), {'bandwidth':np.linspace(0.01, 2., 20)}, cv=4, verbose=1, n_jobs=1)
            print '\nDetermining best bandwidth...\n'
            grid.fit(a_scaled_samples)
            #print grid.best_estimator_

            kde = neighbors.KernelDensity(**grid.best_params_)
            kde.fit(a_scaled_samples)
            
            def func_for_minimizing_for_plot(a_parameters):
                a_scaled_parameters = scaler.transform(a_parameters)
                return -kde.score(a_scaled_parameters)
            
            
            #a_bounds = [(0.75, 1), (1, 25), (0, 1.0), (1e3, 1e5), (5e4, 8e5), (0.6, 3.), (0.2, 2)]
            a_bounds = [(np.min(a_partial_sampler[:, i]), np.max(a_partial_sampler[:, i]))]
            result = op.differential_evolution(func_for_minimizing_for_plot, a_bounds, disp=True, maxiter=100, tol=0.01, popsize=20, polish=True)
            result = op.minimize(func_for_minimizing_for_plot, result.x)
            
            a_best_fit_pars[i] = result.x[0]
        
        """
        
        
        
        a_best_fit_indices = np.unravel_index(np.argmax(a_full_ln_likelihood), a_full_ln_likelihood.shape)
        a_best_fit_pars = a_sampler[a_best_fit_indices]
        print '\n\nUsing best fit so far - ln(L) = %.3f\n\n' % (a_full_ln_likelihood[a_best_fit_indices])
        
        
        """
        a_sampler = a_sampler[:, -num_steps_to_include:, :].reshape((-1, num_dim))
        #a_sampler = a_sampler[:, -num_steps_to_include:, :].reshape((-1, num_dim))
        a_best_fit_pars = np.median(a_sampler, axis=0)
        print '\n\nUsing median...\n\n'
        """
        
        l_num_photons = [i for i in xrange(20)]
        l_colors = plt.get_cmap('jet')(np.linspace(0, 1.0, len(l_num_photons)))
        
        if self.b_use_cascade_model:
            prob_hit_first, collection_efficiency, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, underamp_ionization_correction, poor_collection_ionization_correction, bkg_mean, bkg_std, mean_num_incident_photons, prob_pe_from_photocathode, scale_par = a_best_fit_pars
            
            #print 'Testing best fit from diff ev'
            #prob_hit_first, collection_efficiency, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, underamp_ionization_correction_max, underamp_ionization_correction_slope, poor_collection_ionization_correction, bkg_mean, bkg_std, mean_num_pe, scale_par = self.a_free_par_guesses
            
            bkg_exp = 1
            prob_exp_bkg = 1e-20
        else:
            if not self.file_identifier[:5] == 'nerix':
                prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = a_best_fit_pars
                bkg_exp = 1
                prob_exp_bkg = 0.0001
            
            else:
                prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, mean_num_pe, scale_par = a_best_fit_pars
                bkg_exp = 1
                prob_exp_bkg = 0.0001

        l_hists = [np.zeros(len(self.d_fit_files['bin_centers_plots']), dtype=np.float32) for i in xrange(len(l_num_photons))]
        sum_hist = np.zeros(len(self.d_fit_files['bin_centers_plots']), dtype=np.float32)
        
        
        
        
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)



        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        
        if self.b_use_cascade_model:
            prob_pe_from_photocathode = np.asarray(prob_pe_from_photocathode, dtype=np.float32)
            mean_num_incident_photons = np.asarray(mean_num_incident_photons, dtype=np.float32)
            mean_e_from_dynode = np.asarray(mean_e_from_dynode, dtype=np.float32)
            width_e_from_dynode = np.asarray(width_e_from_dynode, dtype=np.float32)
            probability_electron_ionized = np.asarray(probability_electron_ionized, dtype=np.float32)
            
            collection_efficiency = np.asarray(collection_efficiency, dtype=np.float32)
            poor_collection_ionization_correction = np.asarray(poor_collection_ionization_correction, dtype=np.float32)
            underamp_ionization_correction = np.asarray(underamp_ionization_correction, dtype=np.float32)
        
            #print 'Fixing underamp correction to 1 from %f' % (underamp_ionization_correction)
            #underamp_ionization_correction = np.asarray(1., dtype=np.float32)
        else:
            mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
            spe_mean = np.asarray(spe_mean, dtype=np.float32)
            spe_std = np.asarray(spe_std, dtype=np.float32)
            under_amp_mean = np.asarray(under_amp_mean, dtype=np.float32)
            under_amp_std = np.asarray(under_amp_std, dtype=np.float32)
        
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)
        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)
        
        bin_edges = np.asarray(self.d_fit_files['bin_edges_plots'], dtype=np.float32)
        num_bins = np.asarray(len(bin_edges)-1, dtype=np.int32)
        
        sum_of_hists = 0

        for i, num_photons in enumerate(l_num_photons):
            current_hist = l_hists[i]
        
        
            
            if self.b_use_cascade_model:
                num_trials = np.asarray(int(self.num_mc_events*scipy.stats.poisson.pmf(num_photons, mean_num_incident_photons)), dtype=np.int32)
                num_photons = np.asarray(num_photons, dtype=np.int32)
                
                l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(current_hist), drv.In(num_photons), drv.In(prob_pe_from_photocathode), drv.In(prob_hit_first), drv.In(collection_efficiency), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(underamp_ionization_correction), drv.In(poor_collection_ionization_correction), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges), drv.In(self.num_dynodes), self.gpu_dynode_chain]
                self.gpu_fixed_pe_cascade_spectrum(*l_args_gpu, **self.d_gpu_scale)
            else:
                # same counter
                num_pe = np.asarray(num_photons, dtype=np.int32)
                num_trials = np.asarray(int(self.num_mc_events*scipy.stats.poisson.pmf(num_pe, mean_num_pe)), dtype=np.int32)
            
                l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(current_hist), drv.In(num_pe), drv.In(prob_hit_first), drv.In(spe_mean), drv.In(spe_std), drv.In(under_amp_mean), drv.In(under_amp_std), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges)]
                self.gpu_fixed_pe_gaussian_spectrum(*l_args_gpu, **self.d_gpu_scale)
            
            
            
            
            sum_of_hists += np.sum(current_hist)
            
            l_hists[i] = current_hist





        for i, num_pe in enumerate(l_num_photons):
            current_hist = l_hists[i]
            current_hist = np.asarray(current_hist, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/sum_of_hists*self.d_fit_files['bin_width']/self.d_fit_files['bin_width_plots']*scale_par
            sum_hist += current_hist
            l_hists[i] = current_hist
            
        

        f1, (ax1) = plt.subplots(1)
        ax1.set_yscale('log', nonposx='clip')
    
        a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = neriX_analysis.prepare_hist_arrays_for_plotting(self.d_fit_files['hist'], self.d_fit_files['bin_edges'])
        ax1.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='k', fmt='.')
        for i in xrange(len(l_num_photons)):
            ax1.plot(self.d_fit_files['bin_centers_plots'], l_hists[i], color=l_colors[i])
        #ax1.plot(self.d_fit_files['bin_centers_plots'], sum_hist, color='magenta', linestyle='-')

        #ax1.set_ylim(np.min(a_y_values)/1.5, 2*np.max(a_y_values))
        ax1.set_ylim(20, np.max(a_y_values)*1.1)

        ax1.set_title('Integrated Charge Spectrum - %s' % (self.file_identifier))
        ax1.set_xlabel(r'Integrated Charge [$e^{-}$]')
        ax1.set_ylabel('Counts')


        # test
        
        num_bins_plots = len(self.d_fit_files['bin_centers_plots'])

        a_hist = np.zeros(num_bins_plots, dtype=np.float32)
            
        a_hist_pure = np.zeros(num_bins_plots, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        
        if self.b_use_cascade_model:
            l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist), drv.In(mean_num_incident_photons), drv.In(prob_pe_from_photocathode), drv.In(prob_hit_first), drv.In(collection_efficiency), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(underamp_ionization_correction), drv.In(poor_collection_ionization_correction), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges), drv.In(self.num_dynodes), self.gpu_dynode_chain]
            self.gpu_cascade_model(*l_args_gpu, **self.d_gpu_scale)
        else:
            l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist), drv.In(mean_num_pe), drv.In(prob_hit_first), drv.In(spe_mean), drv.In(spe_std), drv.In(under_amp_mean), drv.In(under_amp_std), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges)]
            self.gpu_gaussian_model(*l_args_gpu, **self.d_gpu_scale)
        
    
    
        a_model = np.asarray(a_hist, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/np.sum(a_hist)*self.d_fit_files['bin_width']/self.d_fit_files['bin_width_plots']*scale_par

        #print len(a_model), len(self.d_fit_files['bin_centers_plots'])
        #print a_model
        ax1.plot(self.d_fit_files['bin_centers_plots'], a_model, color='darkorange', linestyle='--')
        
        
        print '\nBest Fit Parameters: '
        print a_best_fit_pars
        print '\n\n'

        
        f1.savefig('%s%s_pe_specs_%s.png' % (self.s_directory_save_plots_name, self.s_base_save_name, self.file_identifier))
        

        if self.b_use_cascade_model:
            best_chi2, ndf = self.cascade_model_ln_likelihood(a_best_fit_pars, True)
            print '\n\nChi2 / NDF: %.2f / %d\n\n' % (best_chi2, ndf)

            self.perform_ad_test_cascade(a_best_fit_pars, num_bootstraps=1000)

        else:
            self.perform_ad_test_gaussian(a_best_fit_pars, num_bootstraps=1000)
        
        if b_save_dict_for_plot:
            a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high
            d_best_fit_plot_pars = {}
            d_best_fit_plot_pars['data_x'] = a_x_values
            d_best_fit_plot_pars['data_y'] = a_y_values
            d_best_fit_plot_pars['data_x_err_low'] = a_x_err_low
            d_best_fit_plot_pars['data_x_err_high'] = a_x_err_high
            d_best_fit_plot_pars['data_y_err_low'] = a_y_err_low
            d_best_fit_plot_pars['data_y_err_high'] = a_y_err_high
        
            d_best_fit_plot_pars['mc_x'] = self.d_fit_files['bin_centers_plots']
            d_best_fit_plot_pars['mc_y'] = a_model
        
            if self.b_use_cascade_model:
                pickle.dump(d_best_fit_plot_pars, open('./plot_dicts/%s_best_fit.p' % (self.file_identifier), 'w'))
            else:
                pickle.dump(d_best_fit_plot_pars, open('./plot_dicts/%s_best_fit_gm.p' % (self.file_identifier), 'w'))
        
        
        

        #plt.show()
    
    
    
    def gaussian_model_ln_likelihood(self, a_parameters):
        if not self.file_identifier[:5] == 'nerix':
            prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = a_parameters
            bkg_exp, prob_exp_bkg = 1, 0.0001
        else:
            # do not include exponential background for neriX
            prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, mean_num_pe, scale_par = a_parameters
            bkg_exp, prob_exp_bkg = 1, 0.0001
        

        ln_prior = 0
        ln_likelihood = 0

        ln_prior += self.prior_between_0_and_1(prob_hit_first)
        ln_prior += self.prior_greater_than_0(spe_mean)
        ln_prior += self.prior_greater_than_0(spe_std)
        ln_prior += self.prior_greater_than_0(under_amp_mean)
        ln_prior += self.prior_greater_than_0(under_amp_std)
        ln_prior += self.prior_greater_than_0(bkg_std)
        ln_prior += self.prior_greater_than_0(mean_num_pe)
        ln_prior += self.prior_greater_than_0(bkg_exp)
        ln_prior += self.prior_between_0_and_1(prob_exp_bkg)
        
        if not self.b_nerix_file:
            ln_prior += self.prior_uc_bkg(bkg_mean, bkg_std)
        
        
        if self.file_identifier[:5] == 'nerix':
            if not (8.5e5 < spe_mean < 1e6):
                return -np.inf
            if not (4.5e5 < spe_std < 6.5e6):
                return -np.inf
            if not (prob_hit_first > 0.45):
                return -np.inf
            if spe_std > 1.0*spe_mean:
                return -np.inf
        
        else:
            if bkg_exp > spe_mean:
                return -np.inf

        
        if mean_num_pe < 0.9:
            return -np.inf
        
        
        # under amplified peak should be
        # ~1/num_dynodes=1/12 of SPE but gave buffer
        if under_amp_mean < 0.75*spe_mean**(11./12.) or under_amp_mean > 1.25*spe_mean**(11./12.) or under_amp_std > under_amp_mean:
            return -np.inf

        if not np.isfinite(ln_prior):
            return -np.inf

        a_hist = np.zeros(len(self.d_fit_files['hist']), dtype=np.float32)
        
        mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        spe_mean = np.asarray(spe_mean, dtype=np.float32)
        spe_std = np.asarray(spe_std, dtype=np.float32)
        under_amp_mean = np.asarray(under_amp_mean, dtype=np.float32)
        under_amp_std = np.asarray(under_amp_std, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)
        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)
        
        num_bins = np.asarray(len(self.d_fit_files['hist']), dtype=np.int32)
        bin_edges = np.asarray(self.d_fit_files['bin_edges'], dtype=np.float32)
        
        
        l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist), drv.In(mean_num_pe), drv.In(prob_hit_first), drv.In(spe_mean), drv.In(spe_std), drv.In(under_amp_mean), drv.In(under_amp_std), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges)]
    
    
        #start_time_mpe1 = time.time()
        self.gpu_gaussian_model(*l_args_gpu, **self.d_gpu_scale)
        #print 'Time for MPE1 call: %f s' % (time.time() - start_time_spe)
        a_model = np.asarray(a_hist, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/np.sum(a_hist)*scale_par


        ln_likelihood += poisson_binned_likelihood(a_model, self.d_fit_files['hist'])

        total_ln_likelihood = ln_prior + ln_likelihood

        if np.isnan(total_ln_likelihood):
            return -np.inf
            
        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor

        #print total_ln_likelihood
        #print np.sum( (a_model - self.d_fit_files['hist'])**2/self.d_fit_files['hist'] )
        
        return total_ln_likelihood
    
    
    
    
    def draw_gaussian_model_fit(self, a_parameters):
        if not self.file_identifier[:5] == 'nerix':
            prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = a_parameters
        else:
            # do not include exponential background for neriX
            prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, mean_num_pe, scale_par = a_parameters
            bkg_exp, prob_exp_bkg = 1, 0.
        
        
        a_hist = np.zeros(len(self.d_fit_files['hist']), dtype=np.float32)
        
        mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
        
        mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        spe_mean = np.asarray(spe_mean, dtype=np.float32)
        spe_std = np.asarray(spe_std, dtype=np.float32)
        under_amp_mean = np.asarray(under_amp_mean, dtype=np.float32)
        under_amp_std = np.asarray(under_amp_std, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)
        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)
        
        num_bins = np.asarray(len(self.d_fit_files['hist']), dtype=np.int32)
        bin_edges = np.asarray(self.d_fit_files['bin_edges'], dtype=np.float32)
        
        
        l_args_gpu = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist), drv.In(mean_num_pe), drv.In(prob_hit_first), drv.In(spe_mean), drv.In(spe_std), drv.In(under_amp_mean), drv.In(under_amp_std), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges)]
    
    
        #start_time_mpe1 = time.time()
        self.gpu_gaussian_model(*l_args_gpu, **self.d_gpu_scale)
        #print 'Time for MPE1 call: %f s' % (time.time() - start_time_spe)
        a_model = np.asarray(a_hist, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/np.sum(a_hist)*scale_par
        
        
        f1, (ax1) = plt.subplots(1)
        #ax1.set_yscale('log', nonposx='clip')
    
        a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = neriX_analysis.prepare_hist_arrays_for_plotting(self.d_fit_files['hist'], self.d_fit_files['bin_edges'])
        ax1.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='b', fmt='.')
        a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = neriX_analysis.prepare_hist_arrays_for_plotting(a_model, self.d_fit_files['bin_edges'])
        ax1.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='r', fmt='.')
        
        
        #plt.show()




    def testing_model_significance(self, a_pars_cascade, a_pars_gaussian, b_signal_only=False):
    
        # cascade fitting
        prob_hit_first, mean_e_from_dynode, width_e_from_dynode, probability_electron_ionized, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = a_pars_cascade
        num_dim_cascade = len(a_pars_cascade)
        
        
        a_hist_cascade = np.zeros(len(self.d_fit_files['hist']), dtype=np.float32)
        
        mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        mean_e_from_dynode = np.asarray(mean_e_from_dynode, dtype=np.float32)
        width_e_from_dynode = np.asarray(width_e_from_dynode, dtype=np.float32)
        probability_electron_ionized = np.asarray(probability_electron_ionized, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)
        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)
        
        num_bins = np.asarray(len(self.d_fit_files['hist']), dtype=np.int32)
        bin_edges = np.asarray(self.d_fit_files['bin_edges'], dtype=np.float32)
        
        
        l_args_gpu_cascade = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist_cascade), drv.In(mean_num_pe), drv.In(prob_hit_first), drv.In(mean_e_from_dynode), drv.In(width_e_from_dynode), drv.In(probability_electron_ionized), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges), drv.In(self.num_dynodes), self.gpu_dynode_chain]
    
    
        self.gpu_cascade_model(*l_args_gpu_cascade, **self.d_gpu_scale)
        a_model_cascade = np.asarray(a_hist_cascade, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/np.sum(a_hist_cascade)*scale_par
    
    
    
    
    
        # gaussian fitting
        num_dim_gaussian = len(a_pars_gaussian)
        prob_hit_first, spe_mean, spe_std, under_amp_mean, under_amp_std, bkg_mean, bkg_std, bkg_exp, prob_exp_bkg, mean_num_pe, scale_par = a_pars_gaussian
        
        
        a_hist_gaussian = np.zeros(len(self.d_fit_files['hist']), dtype=np.float32)
        
        mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
        
        mean_num_pe = np.asarray(mean_num_pe, dtype=np.float32)
        
        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        prob_hit_first = np.asarray(prob_hit_first, dtype=np.float32)
        spe_mean = np.asarray(spe_mean, dtype=np.float32)
        spe_std = np.asarray(spe_std, dtype=np.float32)
        under_amp_mean = np.asarray(under_amp_mean, dtype=np.float32)
        under_amp_std = np.asarray(under_amp_std, dtype=np.float32)
        bkg_mean = np.asarray(bkg_mean, dtype=np.float32)
        bkg_std = np.asarray(bkg_std, dtype=np.float32)
        bkg_exp = np.asarray(bkg_exp, dtype=np.float32)
        prob_exp_bkg = np.asarray(prob_exp_bkg, dtype=np.float32)
        
        
        l_args_gpu_gaussian = [self.rng_states, drv.In(num_trials), drv.In(self.num_loops), drv.InOut(a_hist_gaussian), drv.In(mean_num_pe), drv.In(prob_hit_first), drv.In(spe_mean), drv.In(spe_std), drv.In(under_amp_mean), drv.In(under_amp_std), drv.In(bkg_mean), drv.In(bkg_std), drv.In(bkg_exp), drv.In(prob_exp_bkg), drv.In(num_bins), drv.In(bin_edges)]
    
        self.gpu_gaussian_model(*l_args_gpu_gaussian, **self.d_gpu_scale)
        a_model_gaussian = np.asarray(a_hist_gaussian, dtype=np.float32)*np.sum(self.d_fit_files['hist'])/np.sum(a_hist_gaussian)*scale_par
    
    
    
    
    
        if b_signal_only:
            print '\nUsing signal only!'
            cutoff_value_bkg = bkg_mean + 5*bkg_std
            for cutoff_index in xrange(num_bins):
                print cutoff_value_bkg, bin_edges[cutoff_index]
                if cutoff_value_bkg < bin_edges[cutoff_index]:
                    break
    
            print 'Using last %d bins of %d.' % (num_bins-cutoff_index, num_bins)
            
            # correct number of bins in use
            num_bins -= cutoff_index
            
        else:
            print '\nUsing full spectrum!'
            cutoff_index = 0
    
    
        f_cascade_model_ln_likelihood = poisson_binned_likelihood(a_model_cascade[cutoff_index:], self.d_fit_files['hist'][cutoff_index:])
        f_gaussian_model_ln_likelihood = poisson_binned_likelihood(a_model_gaussian[cutoff_index:], self.d_fit_files['hist'][cutoff_index:])
    
        reduced_chi2_cascade = np.sum((a_model_cascade[cutoff_index:] - self.d_fit_files['hist'][cutoff_index:])**2 / self.d_fit_files['hist'][cutoff_index:])
        reduced_chi2_gaussian = np.sum((a_model_gaussian[cutoff_index:] - self.d_fit_files['hist'][cutoff_index:])**2 / self.d_fit_files['hist'][cutoff_index:])
        
        #print (a_model_cascade[cutoff_index:] - self.d_fit_files['hist'][cutoff_index:])**2 / self.d_fit_files['hist'][cutoff_index:]
        #print (a_model_gaussian[cutoff_index:] - self.d_fit_files['hist'][cutoff_index:])**2 / self.d_fit_files['hist'][cutoff_index:]
    
        print '\nCascade Model Likelihood: %.2f' % (f_cascade_model_ln_likelihood)
        print 'Cascade Model Chi^2 / NDF: %.2f / %d' % (reduced_chi2_cascade, num_bins-num_dim_cascade)
        print 'Gaussian Model Likelihood: %.2f' % (f_gaussian_model_ln_likelihood)
        print 'Gaussian Model Chi^2 / NDF: %.2f / %d\n' % (reduced_chi2_gaussian, num_bins-num_dim_gaussian)
    
    
    
    
    
        # get summed likelihood and likelihood array for each
        # https://www.rochester.edu/college/psc/clarke/SDFT.pdf
        # cascade is f and gaussian is g
        modified_lr = f_cascade_model_ln_likelihood - f_gaussian_model_ln_likelihood
        modified_lr -= (num_dim_cascade/2. - num_dim_gaussian/2.) * np.log(num_bins)
    
        a_ln_l_cascade = array_poisson_binned_likelihood(a_model_cascade[cutoff_index:], self.d_fit_files['hist'][cutoff_index:])
        a_ln_l_gaussian = array_poisson_binned_likelihood(a_model_gaussian[cutoff_index:], self.d_fit_files['hist'][cutoff_index:])
    
        omega_squared = 1./num_bins*np.sum(np.log(a_ln_l_cascade/a_ln_l_gaussian)**2.) - (1./num_bins*np.sum(np.log(a_ln_l_cascade/a_ln_l_gaussian)))**2.
    
        #print modified_lr, omega_squared
        
        f1, (ax1) = plt.subplots(1)
        ax1.set_yscale('log', nonposx='clip')
    
        a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = neriX_analysis.prepare_hist_arrays_for_plotting(self.d_fit_files['hist'], self.d_fit_files['bin_edges'])
        ax1.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='k', fmt='.')
        
        ax1.plot(a_x_values, a_model_cascade, color='darkorange', linestyle='-')
        ax1.plot(a_x_values, a_model_gaussian, color='magenta', linestyle='-')

        ax1.set_title('Integrated Charge Spectrum - %s' % (self.file_identifier))
        ax1.set_xlabel(r'Integrated Charge [$e^{-}$]')
        ax1.set_ylabel('Counts')
        
        #plt.show()
        
        
        return modified_lr / num_bins**0.5 / omega_squared**0.5
    
    
    
    



    def differential_evolution_minimizer(self, a_bounds, maxiter=250, tol=0.05, popsize=15, polish=False):
        def neg_log_likelihood_diff_ev(a_guesses):
            if self.b_use_cascade_model:
                return -self.cascade_model_ln_likelihood(a_guesses)
            else:
                return -self.gaussian_model_ln_likelihood(a_guesses)
        print '\n\nStarting differential evolution minimizer...\n\n'
        result = op.differential_evolution(neg_log_likelihood_diff_ev, a_bounds, disp=True, maxiter=maxiter, tol=tol, popsize=popsize, polish=polish)
        print result



    def suppress_likelihood(self, iterations=200):

        a_free_par_guesses = self.a_free_par_guesses
        
        l_parameters = [a_free_par_guesses for i in xrange(iterations)]
        l_log_likelihoods = [0. for i in xrange(iterations)]
        for i in tqdm.tqdm(xrange(iterations)):
            if self.b_use_cascade_model:
                l_log_likelihoods[i] = self.cascade_model_ln_likelihood(a_free_par_guesses)
            else:
                l_log_likelihoods[i] = self.gaussian_model_ln_likelihood(a_free_par_guesses)
    

        #print l_log_likelihoods
        std_ll = np.nanstd(l_log_likelihoods, ddof=1)

        print 'Mean for %.3e MC iterations is %f' % (self.num_mc_events*self.num_loops, np.nanmean(l_log_likelihoods))
        print 'Standard deviation for %.3e MC iterations is %f' % (self.num_mc_events*self.num_loops, std_ll)
        print 'Will scale LL such that variance is 0.5'

        self.b_suppress_likelihood = True
        self.ll_suppression_factor = std_ll / 0.5

        print 'LL suppression factor: %f\n' % self.ll_suppression_factor




if __name__ == '__main__':
    
    #filename = 'nerix_160418_1523'
    #filename = 'nerix_160418_1531'
    
    #filename = 'darkbox_spectra_0062_0061'
    #filename = 'darkbox_spectra_0066_0065'
    filename = 'darkbox_spectra_0067_0068'
    #filename = 'darkbox_spectra_0071_0072'
    #filename = 'darkbox_spectra_0073_0074'
    
    #filename = 'fake'
    
    gpu_number = 0
  
    num_mc_events = 4e6
    num_loops = 1
  
    #test = fit_pmt_gain(filename, gpu_number=gpu_number, num_mc_events=num_mc_events, num_loops=num_loops, b_making_comparison_plots=True, b_use_cascade_model=True)

    #test.draw_cascade_model_fit([9.58421447e-03, 8.43049707e-01, 7.34426895e+00*1.5, 3., 3.30671864e-01/1.5, 8.20732092e-01, 0.92, 6.62353661e+04*1.2, 2.30312280e+05, 8.74921049e+00*1.9, 1.39186822e-01/1.1, 9.91798366e-01])
    #print test.cascade_model_ln_likelihood([9.58421447e-03, 8.43049707e-01, 7.34426895e+00*1.5, 3., 3.30671864e-01/1.5, 8.20732092e-01, 0.92, 6.62353661e+04*1.2, 2.30312280e+05, 8.74921049e+00, 1.39186822e-01/1.1, 9.91798366e-01])
    #print test.cascade_model_ln_likelihood(test.a_free_par_guesses)
    #test.draw_cascade_model_fit(test.a_free_par_guesses)

    
    # do not uncomment unless you want to overwrite!!
    #test.create_fake_data_cascade(test.a_free_par_guesses)
    
    #test.draw_model_with_error_bands(num_walkers=64, num_steps_to_include=10, b_save_dict_for_plot=True)
    #test.draw_model_fit_with_peaks(num_walkers=64, num_steps_to_include=30, b_save_dict_for_plot=True)
    
    
    # neriX
    #test.draw_cascade_model_fit([0.98, 0.8, 25.73, 1, 0.1, 9.81466501e-01, 9.04098762e-01, 1e5, 2.5e5, 1.05, 9.97667221e-01])
    #test.draw_cascade_model_fit(test.a_free_par_guesses)
    #print test.cascade_model_ln_likelihood([0.98, 0.8, 25.73, 1, 0.1, 9.81466501e-01, 9.04098762e-01, 1e5, 2.5e5, 1.05, 9.97667221e-01])
    
    

    # UC
    # [0.01, 8.92911974e-01, 1.52550833e+01, 1.00407309e+00, 1.79592425e-01, 8.12041534e-01, 8.76785211e-01, -8.93680855e+03, 2.53804858e+05, 1.21247546e+00/0.3, 0.31, 1.00175953e+00]
    #a_bounds = [(0.001, 0.01), (0.8, 1.0), (14.5, 16.5), (0.5, 6.5), (0.1, 0.5), (0.75, 1.0), (0.75, 1.0), (-1e5, 1e5), (5e4, 8e5), (1.5, 12.), (0.1, 0.4), (0.98, 1.02)]

    #n1: [8.15021255e-03/1.3, 8.89785325e-01, 12/2, 2.63208068e+00, 2.01119635e-01*2, 9.42134482e-01, 8.58082164e-01, 6e4, 2.4e5, 4.95481176e+00, 0.26, 9.97487422e-01]
    #n2: [8.15021255e-03/1.3, 8.89785325e-01, 12/2, 2.63208068e+00, 2.01119635e-01*2, 9.42134482e-01, 8.58082164e-01, 6e4, 2.4e5, 4.95481176e+00*1.9, 0.26, 9.97487422e-01]
    # neriX
    #a_bounds = [(0.001, 0.01), (0.8, 1.0), (5.5, 7.5), (0.5, 3.5), (0.2, 0.5), (0.75, 1.0), (0.75, 1.0), (0, 2e5), (5e4, 4e5), (1.5, 14.), (0.1, 0.4), (0.98, 1.02)]
    #test.differential_evolution_minimizer(a_bounds, maxiter=150, tol=0.05, popsize=20, polish=False)

    #test.suppress_likelihood()
    #test.run_mcmc(num_walkers=64, num_steps=75*3, threads=1)
    
    
    
    test = fit_pmt_gain(filename, gpu_number=gpu_number, num_mc_events=num_mc_events, num_loops=num_loops, b_making_comparison_plots=True, b_use_cascade_model=False)
    
    #print test.gaussian_model_ln_likelihood(test.a_free_par_guesses)
    #test.draw_gaussian_model_fit(test.a_free_par_guesses)
    
    #test.draw_model_with_error_bands(num_walkers=64, num_steps_to_include=10)
    test.draw_model_fit_with_peaks(num_walkers=64, num_steps_to_include=30, b_save_dict_for_plot=True)

    # uc
    #a_bounds = [(0.85, 1), (1e6, 1.2e7), (5e5, 5e6), (3e5, 2e6), (5e4, 1e6), (-1e5, 1e5), (5e4, 8e5), (1e4, 2e6), (0, 0.45), (1., 3.), (0.8, 1.2)]
    # nerix
    #a_bounds = [(0.5, 1), (5e5, 1.1e6), (5e5, 1e6), (1e5, 8e5), (1e3, 7e5), (-1e5, 1e5), (5e4, 8e5), (1., 3.), (0.8, 1.2)]
    #test.differential_evolution_minimizer(a_bounds, maxiter=150, tol=0.05, popsize=20, polish=False)

    #test.suppress_likelihood()
    #test.run_mcmc(num_walkers=64, num_steps=2000, threads=1)






    #print test.testing_model_significance(test.d_best_fit_pars['cascade'], test.d_best_fit_pars['gaussian'], b_signal_only=False)



