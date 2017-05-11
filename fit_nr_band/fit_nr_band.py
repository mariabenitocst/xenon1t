
import matplotlib
matplotlib.use('QT4Agg')

import ROOT as root
from ROOT import gROOT

import sys, os, root_numpy, threading, random, emcee, corner, signal, tqdm

from rootpy import stl
from rootpy.io import File
from rootpy.tree import Tree, TreeModel, TreeChain
from rootpy.plotting import Hist, Hist2D, Canvas, Graph, func

import neriX_analysis, neriX_datasets
import config_xe1t

import numpy as np
from math import exp, factorial, erf, ceil, log, pow, floor, lgamma

from scipy import optimize, misc, stats
from scipy.stats import norm, multivariate_normal, binom

import copy_reg, types, pickle, click, time
from subprocess import call
from pprint import pprint

from produce_nest_yields import nest_nr_mean_yields

import cuda_full_observables_production
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray
from Queue import Queue
import threading

from sklearn import neighbors
from sklearn import grid_search
from sklearn import preprocessing
import pandas as pd



def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))


d_cathode_voltage_to_field = {0.345:210,
                              1.054:490,
                              2.356:1000}



class gpu_pool:
    def __init__(self, l_gpus, grid_dim, block_dim, num_dim_gpu_call, d_gpu_single_copy_arrays, function_name):
        self.num_gpus = len(l_gpus)
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.num_dim_gpu_call = num_dim_gpu_call
        self.d_gpu_single_copy_arrays = d_gpu_single_copy_arrays
        self.function_name = function_name

        self.alive = True
        self.q_gpu = Queue()
        for i in l_gpus:
            self.q_gpu.put(i)

        self.q_in = Queue()
        self.q_out = Queue()
        self.l_dispatcher_threads = []
        self.dispatcher_dead_time = 0.5

        self.q_dead = Queue()


        for i in xrange(self.num_gpus):
            if self.q_gpu.empty():
                break
            print 'starting worker'
            self.l_dispatcher_threads.append(threading.Thread(target=self.dispatcher, args=[self.q_gpu.get()]))
            self.l_dispatcher_threads[-1].start()


    def dispatcher(self, device_num):

        try:
            drv.init()
        except:
            self.q_dead.put(device_num)
            sys.exit()

        try:
            dev = drv.Device(device_num)
        except:
            self.q_dead.put(device_num)
            sys.exit()

        #print device_num
        #print drv.Device.count()
        if drv.Device.count() - 1 < device_num:
            self.q_dead.put(device_num)
            sys.exit()

        ctx = dev.make_context()
        print dev.name()




        seed = int(time.time()*1000)


        # source code
        local_gpu_setup_kernel = pycuda.compiler.SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('setup_kernel')

        local_rng_states = drv.mem_alloc(np.int32(self.block_dim*self.grid_dim)*pycuda.characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
        local_gpu_setup_kernel(np.int32(int(self.block_dim*self.grid_dim)), local_rng_states, np.uint64(0), np.uint64(0), grid=(int(self.grid_dim), 1), block=(int(self.block_dim), 1, 1))

        print '\nSetup Kernel run!\n'
        sys.stdout.flush()

        gpu_observables_func = SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function(self.function_name)
        
        """
        d_gpu_single_copy_array_dictionaries = {'mc_energy':self.a_mc_energy,
                                                'mc_x':self.a_mc_x,
                                                'mc_y':self.a_mc_y,
                                                'mc_z':self.a_mc_z,
                                                'e_survival_prob':self.a_e_survival_prob,
                                                'er_band_s1':self.a_er_s1,
                                                'er_band_log':self.a_er_log,
                                                'bin_edges_r2':self.bin_edges_r2,
                                                'bin_edges_z':self.bin_edges_z,
                                                's1_correction_map':self.s1_correction_map,
                                                'bin_edges_x':self.bin_edges_x,
                                                'bin_edges_y':self.bin_edges_y,
                                                's2_correction_map':self.s2_correction_map,
                                                'bin_edges_s1':self.a_s1_bin_edges,
                                                'bin_edges_log':self.a_log_bin_edges,
                                                's1bs_s1s':self.a_s1bs_s1s,
                                                's1bs_lb_bias':self.a_s1bs_lb_bias,
                                                's1bs_ub_bias':self.a_s1bs_ub_bias,
                                                's1bs_lb_smearing':self.a_s1bs_lb_smearing,
                                                's1bs_ub_smearing':self.a_s1bs_ub_smearing,
                                                's2bs_s2s':self.a_s2bs_s2s,
                                                's2bs_lb_bias':self.a_s2bs_lb_bias,
                                                's2bs_ub_bias':self.a_s2bs_ub_bias,
                                                's2bs_lb_smearing':self.a_s2bs_lb_smearing,
                                                's2bs_ub_smearing':self.a_s2bs_ub_smearing,
                                                's1pf_s1s':self.a_s1pf_s1s,
                                                's1pf_lb_acc':self.a_s1pf_lb_acc,
                                                's1pf_mean_acc':self.a_s1pf_mean_acc,
                                                's1pf_ub_acc':self.a_s1pf_ub_acc,
                                                's1cuts_s1s':self.a_s1cuts_s1s,
                                                's1cuts_mean_acc':self.a_s1cuts_mean_acc,
                                                's2cuts_s2s':self.a_s2cuts_s2s,
                                                's2cuts_mean_acc':self.a_s2cuts_mean_acc
                                                }
        """

        print 'Putting MC input arrays on GPU...'
        gpu_energies = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['mc_energy'])
        gpu_x_positions = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['mc_x'])
        gpu_y_positions = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['mc_y'])
        gpu_z_positions = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['mc_z'])
        
        gpu_e_survival_prob = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['e_survival_prob'])
        
        gpu_er_band_s1 = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['er_band_s1'])
        gpu_er_band_log = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['er_band_log'])

        gpu_bin_edges_r2 = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_r2'])
        gpu_bin_edges_z = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_z'])
        gpu_s1_correction_map = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['s1_correction_map'])
        
        gpu_bin_edges_x = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_x'])
        gpu_bin_edges_y = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_y'])
        gpu_s2_correction_map = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['s2_correction_map'])

        print 'Putting bin edges on GPU...'
        gpu_bin_edges_s1 = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_s1'])


        print 'Putting bin edges on GPU...'
        gpu_bin_edges_log = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_log'])


        print 'Putting bias and smearing arrays on GPU...'
        gpu_s1bs_s1s = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1bs_s1s'], dtype=np.float32))
        gpu_s1bs_lb_bias = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1bs_lb_bias'], dtype=np.float32))
        gpu_s1bs_ub_bias = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1bs_ub_bias'], dtype=np.float32))
        gpu_s1bs_lb_smearing = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1bs_lb_smearing'], dtype=np.float32))
        gpu_s1bs_ub_smearing = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1bs_ub_smearing'], dtype=np.float32))

        gpu_s2bs_s2s = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s2bs_s2s'], dtype=np.float32))
        gpu_s2bs_lb_bias = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s2bs_lb_bias'], dtype=np.float32))
        gpu_s2bs_ub_bias = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s2bs_ub_bias'], dtype=np.float32))
        gpu_s2bs_lb_smearing = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s2bs_lb_smearing'], dtype=np.float32))
        gpu_s2bs_ub_smearing = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s2bs_ub_smearing'], dtype=np.float32))
        
        print 'Putting acceptance arrays on GPU...'
        gpu_s1pf_s1s = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1pf_s1s'], dtype=np.float32))
        gpu_s1pf_lb_acc = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1pf_lb_acc'], dtype=np.float32))
        gpu_s1pf_mean_acc = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1pf_mean_acc'], dtype=np.float32))
        gpu_s1pf_ub_acc = pycuda.gpuarray.to_gpu(np.asarray(self.d_gpu_single_copy_arrays['s1pf_ub_acc'], dtype=np.float32))
        
        
    
        
        
        
        
        d_gpu_local_info = {'function_to_call':gpu_observables_func,
                            'rng_states':local_rng_states,
                            'gpu_energies':gpu_energies,
                            'gpu_x_positions':gpu_x_positions,
                            'gpu_y_positions':gpu_y_positions,
                            'gpu_z_positions':gpu_z_positions,
                            'gpu_e_survival_prob':gpu_e_survival_prob,
                            'gpu_er_band_s1':gpu_er_band_s1,
                            'gpu_er_band_log':gpu_er_band_log,
                            'gpu_bin_edges_r2':gpu_bin_edges_r2,
                            'gpu_bin_edges_z':gpu_bin_edges_z,
                            'gpu_s1_correction_map':gpu_s1_correction_map,
                            'gpu_bin_edges_x':gpu_bin_edges_x,
                            'gpu_bin_edges_y':gpu_bin_edges_y,
                            'gpu_s2_correction_map':gpu_s2_correction_map,
                            'gpu_bin_edges_s1':gpu_bin_edges_s1,
                            'gpu_bin_edges_log':gpu_bin_edges_log,
                            'gpu_s1bs_s1s':gpu_s1bs_s1s,
                            'gpu_s1bs_lb_bias':gpu_s1bs_lb_bias,
                            'gpu_s1bs_ub_bias':gpu_s1bs_ub_bias,
                            'gpu_s1bs_lb_smearing':gpu_s1bs_lb_smearing,
                            'gpu_s1bs_ub_smearing':gpu_s1bs_ub_smearing,
                            'gpu_s2bs_s2s':gpu_s2bs_s2s,
                            'gpu_s2bs_lb_bias':gpu_s2bs_lb_bias,
                            'gpu_s2bs_ub_bias':gpu_s2bs_ub_bias,
                            'gpu_s2bs_lb_smearing':gpu_s2bs_lb_smearing,
                            'gpu_s2bs_ub_smearing':gpu_s2bs_ub_smearing,
                            'gpu_s1pf_s1s':gpu_s1pf_s1s,
                            'gpu_s1pf_lb_acc':gpu_s1pf_lb_acc,
                            'gpu_s1pf_mean_acc':gpu_s1pf_mean_acc,
                            'gpu_s1pf_ub_acc':gpu_s1pf_ub_acc,
                            }

        # wrap up function
        # modeled off of pycuda's autoinit

        def _finish_up(ctx):
            print '\n\nWrapping up thread %d...\n\n' % (device_num)
            sys.stdout.flush()
            ctx.pop()

            from pycuda.tools import clear_context_caches
            clear_context_caches()

            # put something in output queue to satisfy
            # parent's map
            self.q_out.put([0, device_num])
            sys.exit()

        #atexit.register(_finish_up, [ctx])
        #atexit.register(ctx.pop)



        while self.alive:
            if not self.q_in.empty():
                task, args, id_num = self.q_in.get()

                #print '\nTask ID: %d\n' % id_num
                #print args
                sys.stdout.flush()

                if task == 'exit':
                    _finish_up(ctx)
                else:
                    #print len(args), self.num_dim_gpu_call
                    if True:# or len(args) == self.num_dim_gpu_call or len(args) == 21:
                        args = np.append(args, [d_gpu_local_info])
                    #print args
                    return_value = task(args)

                #print '\nFinished Task ID: %d\n' % id_num
                #print id_num, return_value

                self.q_out.put((id_num, return_value))
            else:
                time.sleep(self.dispatcher_dead_time)


    def map(self, func, l_args):
        start_time = time.time()

        #print '\n\nMap called\n\n'
        sys.stdout.flush()

        len_input = len(l_args)

        for id_num, args in enumerate(l_args):
            self.q_in.put((func, args, id_num))

        #while not self.q_in.empty():
        while len(self.q_out.queue) != len_input:
            time.sleep(0.1)

        #print 'Time calling function: %.3e' % (time.time() - start_time)
        sys.stdout.flush()

        l_q = list(self.q_out.queue)
        self.q_out.queue.clear()
        #print len(l_q)
        l_q = sorted(l_q, key=lambda x: x[0])
        a_return_values = np.asarray([s_pair[1] for s_pair in l_q])
        #print a_return_values
        return a_return_values


    def close(self):
        self.map('exit', [[0] for i in xrange(self.num_gpus - len(self.q_dead.queue))])
        print 'Closed children'
        time.sleep(2)



import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import rootpy.compiled as C
from numpy.random import normal, binomial, seed, poisson
from rootpy.plotting import root2matplotlib as rplt

C.register_file('%s/c_log_likelihood.C' % (config_xe1t.path_to_this_module), ['smart_log_likelihood'])
c_log_likelihood = C.smart_log_likelihood

#import gc
#gc.disable()

"""
import warnings
#warnings.filterwarnings('error')
# this turns runtime warnings into errors to
# catch unexpected behavior
warnings.simplefilter('error', RuntimeWarning)
# this turns off a deprecation warning in general but aimed
# at corners plot creation (should try to only put there)
# and move back to all warnings as errors)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)
"""

# MUST INCLUDE TYPES TO AVOID SEG FAULT
stl.vector(stl.vector('float'))
stl.vector(stl.vector('double'))
stl.vector(stl.vector('int'))


def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))



class fit_nr(object):
    def __init__(self, d_fit_data, fit_type, num_mc_events, name_notes=None, l_gpus=[0], num_loops=1):
        
        
        # total MC events = num_mc_events*num_loops
        
        self.num_loops = num_loops
        self.fit_type = fit_type
        
        if fit_type == 'sb':
            assert d_fit_data['cathode_setting'] in config_xe1t.l_allowed_cathode_settings
            assert d_coincidence_data['degree_setting'] in config_xe1t.l_allowed_degree_settings
        elif fit_type == 'sbf':
            assert d_fit_data['cathode_setting'] in config_xe1t.l_allowed_cathode_settings
            assert d_coincidence_data['degree_setting'] in config_xe1t.l_allowed_degree_settings
        elif fit_type == 'sb_ms':
            assert d_fit_data['cathode_setting'] in config_xe1t.l_allowed_cathode_settings
            assert d_coincidence_data['degree_setting'] in config_xe1t.l_allowed_degree_settings
        else:
            print '\nDo not recognize fit type: %s.\n' % (fit_type)
            sys.exit()

        copy_reg.pickle(types.MethodType, reduce_method)


        # set number of mc events
        self.d_gpu_scale = {}
        block_dim = 1024/2
        self.d_gpu_scale['block'] = (block_dim,1,1)
        numBlocks = floor(num_mc_events / float(block_dim))
        self.d_gpu_scale['grid'] = (int(numBlocks), 1)
        self.num_mc_events = int(numBlocks*block_dim)


        # ------------------------------------------------
        # ------------------------------------------------
        # Set paths and allowed values
        # ------------------------------------------------
        # ------------------------------------------------

        self.name_notes = name_notes
        
        self.d_cathode_voltages_to_field = config_xe1t.d_cathode_voltages_to_field
        


        # ------------------------------------------------
        # ------------------------------------------------
        # Get information for data
        # ------------------------------------------------
        # ------------------------------------------------
        
        self.degree_setting = d_fit_data['degree_setting']
        self.cathode_setting = d_fit_data['cathode_setting']

        self.l_energy_settings = config_xe1t.l_energy_settings
        self.l_s1_settings = config_xe1t.l_s1_settings
        self.l_s2_settings = config_xe1t.l_s2_settings
        self.l_log_settings = config_xe1t.l_log_settings

        self.l_quantiles = config_xe1t.l_quantiles
        
        # define bin edges for use in MC
        """
        self.a_s1_bin_edges = np.linspace(self.l_s1_settings[1], self.l_s1_settings[2], num=self.l_s1_settings[0]+1, dtype=np.float32)
        self.a_s2_bin_edges = np.linspace(self.l_s2_settings[1], self.l_s2_settings[2], num=self.l_s2_settings[0]+1, dtype=np.float32)
        self.a_log_bin_edges = np.linspace(self.l_log_settings[1], self.l_log_settings[2], num=self.l_log_settings[0]+1, dtype=np.float32)
        """
        
        self.a_s1_bin_edges = config_xe1t.a_s1_bin_edges
        self.a_log_bin_edges = config_xe1t.a_log_bin_edges
        

        self.d_coincidence_data_information = {}

        # add basic information to dictionary
        self.d_coincidence_data_information['d_s1_s2'] = pickle.load(open('%sambe_data.p' % (config_xe1t.path_to_fit_inputs), 'r'))
        
        
        self.d_coincidence_data_information['s1_hist'] = Hist(*self.l_s1_settings, name='h_s1')
        self.d_coincidence_data_information['s1_hist'].fill_array(self.d_coincidence_data_information['d_s1_s2']['s1'])
        
        self.d_coincidence_data_information['s2_hist'] = Hist(*self.l_s2_settings, name='h_s2')
        self.d_coincidence_data_information['s2_hist'].fill_array(self.d_coincidence_data_information['d_s1_s2']['s2'])
        
        self.d_coincidence_data_information['s1_s2_hist'] = Hist2D(*(self.l_s1_settings+self.l_s2_settings), name='h_s1_s2')
        self.d_coincidence_data_information['s1_s2_hist'].fill_array(np.asarray([self.d_coincidence_data_information['d_s1_s2']['s1'], self.d_coincidence_data_information['d_s1_s2']['s2']]).T)
        
        #self.d_coincidence_data_information['log_s2_s1_hist'] = Hist2D(*(self.l_s1_settings+self.l_log_settings), name='h_log')
        #self.d_coincidence_data_information['log_s2_s1_hist'].fill_array(np.asarray([self.d_coincidence_data_information['d_s1_s2']['s1'], np.log10(self.d_coincidence_data_information['d_s1_s2']['s2']/self.d_coincidence_data_information['d_s1_s2']['s1'])]).T)
        self.d_coincidence_data_information['a_log_s2_s1'], _, _ = np.histogram2d(self.d_coincidence_data_information['d_s1_s2']['s1'], np.log10(self.d_coincidence_data_information['d_s1_s2']['s2']/self.d_coincidence_data_information['d_s1_s2']['s1']), bins=[self.a_s1_bin_edges, self.a_log_bin_edges])
        
        self.d_coincidence_data_information['num_data_pts'] = len(self.d_coincidence_data_information['d_s1_s2']['s1'])
        
        #self.d_coincidence_data_information['a_log_s2_s1'] = neriX_analysis.convert_2D_hist_to_matrix(self.d_coincidence_data_information['log_s2_s1_hist'], dtype=np.float32)
        
        
        self.d_coincidence_data_information['a_log_s2_over_s1'] = np.log10(self.d_coincidence_data_information['d_s1_s2']['s2']/self.d_coincidence_data_information['d_s1_s2']['s1'])





        # ------------------------------------------------
        # ------------------------------------------------
        # Load MC files or create reduced file if not present
        # and load into easy_graph
        # ------------------------------------------------
        # ------------------------------------------------

        self.load_mc_data(view_corner_plot=False)

        # fill energy array
        self.fill_energy_array(view_energy_spectrum=False)
        
        #self.fill_mc_inputs()
        
        self.load_correction_maps()


        # ------------------------------------------------
        # ------------------------------------------------
        # Set default parameter values for use in MC
        # ------------------------------------------------
        # ------------------------------------------------


        neriX_analysis.warning_message('Detector parameters hard-coded')

        
        self.w_value = config_xe1t.w_value
        self.w_value_uncertainty = config_xe1t.w_value_uncertainty

        self.g1_value = config_xe1t.g1_value
        self.g1_uncertainty = config_xe1t.g1_uncertainty
        
        self.extraction_efficiency_value = config_xe1t.extraction_efficiency_value
        self.extraction_efficiency_uncertainty = config_xe1t.extraction_efficiency_uncertainty

        self.gas_gain_value = config_xe1t.gas_gain_value
        self.gas_gain_uncertainty = config_xe1t.gas_gain_uncertainty

        self.gas_gain_width = config_xe1t.gas_gain_width
        self.gas_gain_width_uncertainty = config_xe1t.gas_gain_width_uncertainty

        self.spe_res_value = config_xe1t.spe_res_value
        self.spe_res_uncertainty = config_xe1t.spe_res_uncertainty

        self.dpe_lb = config_xe1t.dpe_lb
        self.dpe_ub = config_xe1t.dpe_ub

        self.cut_acceptance_s1_intercept = config_xe1t.cut_acceptance_s1_intercept
        self.cut_acceptance_s1_intercept_uncertainty = config_xe1t.cut_acceptance_s1_intercept_uncertainty
        
        self.cut_acceptance_s1_slope = config_xe1t.cut_acceptance_s1_slope
        self.cut_acceptance_s1_slope_uncertainty = config_xe1t.cut_acceptance_s1_slope_uncertainty
        
        self.cut_acceptance_s2_intercept = config_xe1t.cut_acceptance_s2_intercept
        self.cut_acceptance_s2_intercept_uncertainty = config_xe1t.cut_acceptance_s2_intercept_uncertainty
        
        self.cut_acceptance_s2_slope = config_xe1t.cut_acceptance_s2_slope
        self.cut_acceptance_s2_slope_uncertainty = config_xe1t.cut_acceptance_s2_slope_uncertainty

        self.ms_par_0 = config_xe1t.ms_par_0
        self.ms_par_0_unc= config_xe1t.ms_par_0_unc
        
        self.ms_par_1 = config_xe1t.ms_par_1
        self.ms_par_1_unc = config_xe1t.ms_par_1_unc

        # below are the NEST values for the Lindhard model
        self.nest_lindhard_model = {}
        self.nest_lindhard_model['values'] = {'w_value':13.7,
                                              'alpha':1.240,
                                              'zeta':0.0472,
                                              'beta':239,
                                              'gamma':0.01385,
                                              'delta':0.0620,
                                              'kappa':0.1394,
                                              'eta':3.3,
                                              'lambda':1.14}

        # below are the spreads for walker initialization only
        # these are NOT uncertainties
        self.nest_lindhard_model['spreads'] = {'w_value':0.2,
                                               'alpha':0.008,
                                               'zeta':0.008,
                                               'beta':12,
                                               'gamma':0.00065,
                                               'delta':0.006,
                                               'kappa':0.003,
                                               'eta':2,
                                               'lambda':0.25}
        
        self.nest_lindhard_model['uncertainties'] = {'w_value':{'low':0.2, 'high':0.2},
                                                     'alpha':{'low':0.073, 'high':0.079},
                                                     'zeta':{'low':0.0073, 'high':0.0088},
                                                     'beta':{'low':8.8, 'high':28},
                                                     'gamma':{'low':0.00073, 'high':0.00058},
                                                     'delta':{'low':0.0064, 'high':0.0056},
                                                     'kappa':{'low':0.0026, 'high':0.0032},
                                                     'eta':{'low':0.7, 'high':5.3},
                                                     'lambda':{'low':0.09, 'high':0.45}}



        if fit_type == 'sb':
            self.ln_likelihood_function = self.ln_likelihood_full_matching_band
            self.ln_likelihood_function_wrapper = self.wrapper_ln_likelihood_full_matching_band
            self.num_dimensions = 22
            self.gpu_function_name = 'gpu_full_observables_production_with_log_hist'
            self.directory_name = 'run_0_band'

        elif fit_type == 'sbf':
            self.ln_likelihood_function = self.ln_likelihood_full_matching_band
            self.ln_likelihood_function_wrapper = self.wrapper_ln_likelihood_full_matching_band_with_free_pax_eff
            self.num_dimensions = 23
            self.gpu_function_name = 'gpu_full_observables_production_with_log_hist_with_free_pax_eff'
            self.directory_name = 'run_0_band'

        elif fit_type == 'sb_ms':
            self.ln_likelihood_function = self.ln_likelihood_full_matching_band
            self.ln_likelihood_function_wrapper = self.wrapper_ln_likelihood_full_matching_band_with_ms_scale
            self.num_dimensions = 24
            self.gpu_function_name = 'gpu_full_observables_production_with_log_hist_with_ms_scale'
            self.directory_name = 'run_0_band'

        else:
            print 'Do not know how to handle fit type %s.  Please check input and try again.\n\n' % self.fit_type
            sys.exit()


        # ------------------------------------------------
        # ------------------------------------------------
        # Pull required arrays for splines
        # ------------------------------------------------
        # ------------------------------------------------
        
        self.d_bias_smearing = pickle.load(open('%ss1_s2_bias_and_smearing.p' % (config_xe1t.path_to_fit_inputs), 'r'))
        self.a_s1bs_s1s = np.asarray(self.d_bias_smearing['s1']['points'], dtype=np.float32)
        self.a_s1bs_lb_bias = np.asarray(self.d_bias_smearing['s1']['lb_bias'], dtype=np.float32)
        self.a_s1bs_ub_bias = np.asarray(self.d_bias_smearing['s1']['ub_bias'], dtype=np.float32)
        self.a_s1bs_lb_smearing = np.asarray(self.d_bias_smearing['s1']['lb_smearing'], dtype=np.float32)
        self.a_s1bs_ub_smearing = np.asarray(self.d_bias_smearing['s1']['ub_smearing'], dtype=np.float32)
        self.a_s2bs_s2s = np.asarray(self.d_bias_smearing['s2']['points'], dtype=np.float32)
        self.a_s2bs_lb_bias = np.asarray(self.d_bias_smearing['s2']['lb_bias'], dtype=np.float32)
        self.a_s2bs_ub_bias = np.asarray(self.d_bias_smearing['s2']['ub_bias'], dtype=np.float32)
        self.a_s2bs_lb_smearing = np.asarray(self.d_bias_smearing['s2']['lb_smearing'], dtype=np.float32)
        self.a_s2bs_ub_smearing = np.asarray(self.d_bias_smearing['s2']['ub_smearing'], dtype=np.float32)
        
        
        self.d_acceptances = pickle.load(open('%sacceptances_ambe.p' % (config_xe1t.path_to_fit_inputs), 'r'))
        # PF acceptance
        self.a_s1pf_s1s = np.asarray(self.d_acceptances['pf_s1']['x_values'], dtype=np.float32)
        self.a_s1pf_lb_acc = np.asarray(self.d_acceptances['pf_s1']['y_values_lower'], dtype=np.float32)
        self.a_s1pf_mean_acc = np.asarray(self.d_acceptances['pf_s1']['y_values_mean'], dtype=np.float32)
        self.a_s1pf_ub_acc = np.asarray(self.d_acceptances['pf_s1']['y_values_upper'], dtype=np.float32)
        
        
        

       
        
        
        
        
        # ------------------------------------------------
        # ------------------------------------------------
        # Setup GPU pool and save time by copying
        # arrays over only once and referencing those
        # ------------------------------------------------
        # ------------------------------------------------

        print self.bin_edges_x[0]

        d_gpu_single_copy_array_dictionaries = {'mc_energy':self.a_mc_energy,
                                                'mc_x':self.a_mc_x,
                                                'mc_y':self.a_mc_y,
                                                'mc_z':self.a_mc_z,
                                                'e_survival_prob':self.a_e_survival_prob,
                                                'er_band_s1':self.a_er_s1,
                                                'er_band_log':self.a_er_log,
                                                'bin_edges_r2':self.bin_edges_r2,
                                                'bin_edges_z':self.bin_edges_z,
                                                's1_correction_map':self.s1_correction_map,
                                                'bin_edges_x':self.bin_edges_x,
                                                'bin_edges_y':self.bin_edges_y,
                                                's2_correction_map':self.s2_correction_map,
                                                'bin_edges_s1':self.a_s1_bin_edges,
                                                'bin_edges_log':self.a_log_bin_edges,
                                                's1bs_s1s':self.a_s1bs_s1s,
                                                's1bs_lb_bias':self.a_s1bs_lb_bias,
                                                's1bs_ub_bias':self.a_s1bs_ub_bias,
                                                's1bs_lb_smearing':self.a_s1bs_lb_smearing,
                                                's1bs_ub_smearing':self.a_s1bs_ub_smearing,
                                                's2bs_s2s':self.a_s2bs_s2s,
                                                's2bs_lb_bias':self.a_s2bs_lb_bias,
                                                's2bs_ub_bias':self.a_s2bs_ub_bias,
                                                's2bs_lb_smearing':self.a_s2bs_lb_smearing,
                                                's2bs_ub_smearing':self.a_s2bs_ub_smearing,
                                                's1pf_s1s':self.a_s1pf_s1s,
                                                's1pf_lb_acc':self.a_s1pf_lb_acc,
                                                's1pf_mean_acc':self.a_s1pf_mean_acc,
                                                's1pf_ub_acc':self.a_s1pf_ub_acc,
                                                }
        self.gpu_pool = gpu_pool(l_gpus=l_gpus, grid_dim=numBlocks, block_dim=block_dim, num_dim_gpu_call=self.num_dimensions, d_gpu_single_copy_arrays=d_gpu_single_copy_array_dictionaries, function_name=self.gpu_function_name)

        self.b_suppress_likelihood = False



    def close_workers(self):
        self.gpu_pool.close()



    def fill_energy_array(self, view_energy_spectrum=False):

        random.seed()
    
    
        # -----------------------------------------
        #  Energy
        # -----------------------------------------
    
        """
        bin_width = self.d_mc_energy['a_energy_bins'][1] - self.d_mc_energy['a_energy_bins'][0]

        cdf = np.cumsum(self.d_mc_energy['a_energy_hist'])
        cdf = cdf / cdf[-1]
        values = np.random.rand(self.num_mc_events)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = self.d_mc_energy['a_energy_bins'][value_bins]

        self.a_mc_energy = np.zeros(self.num_mc_events, dtype=np.float32)
        for i in tqdm.tqdm(xrange(self.num_mc_events)):
            current_random_num = np.random.random()*bin_width + random_from_cdf[i]
            
            # need to cover edge case of zero bin
            if current_random_num < 0:
                current_random_num = -current_random_num
            
            self.a_mc_energy[i] = current_random_num
        """
        
        num_times_to_copy_array = (len(self.d_mc_energy['a_energy'])%self.num_mc_events)+1
        self.a_mc_energy = np.concatenate([self.d_mc_energy['a_energy']]*num_times_to_copy_array)[:self.num_mc_events]
    
        if view_energy_spectrum:
            plt.hist(self.a_mc_energy, bins=100)
            plt.show()


        # filling directly from arrays for x, y, energy
        self.a_mc_x = np.concatenate([self.d_mc_energy['a_x']]*num_times_to_copy_array)[:self.num_mc_events]
        self.a_mc_y = np.concatenate([self.d_mc_energy['a_y']]*num_times_to_copy_array)[:self.num_mc_events]
        self.a_mc_z = np.concatenate([self.d_mc_energy['a_z']]*num_times_to_copy_array)[:self.num_mc_events]
        

        # -----------------------------------------
        #  Z
        # -----------------------------------------
        
        """
        bin_width = self.d_mc_positions['z_bin_edges'][1] - self.d_mc_positions['z_bin_edges'][0]

        cdf = np.cumsum(self.d_mc_positions['z_map'])
        cdf = cdf / cdf[-1]
        values = np.random.rand(self.num_mc_events)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = self.d_mc_positions['z_bin_edges'][value_bins]

        self.a_mc_z = np.zeros(self.num_mc_events, dtype=np.float32)
        for i in tqdm.tqdm(xrange(self.num_mc_events)):
            current_random_num = np.random.random()*bin_width + random_from_cdf[i]
            
            self.a_mc_z[i] = current_random_num
        """
        if view_energy_spectrum:
            plt.hist(self.a_mc_z, bins=100)
            plt.show()



        # -----------------------------------------
        #  Electron Lifetime
        # -----------------------------------------
    
        bin_width = self.d_mc_energy['a_el_bins'][1] - self.d_mc_energy['a_el_bins'][0]

        cdf = np.cumsum(self.d_mc_energy['a_el_hist'])
        cdf = cdf / cdf[-1]
        values = np.random.rand(self.num_mc_events)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = self.d_mc_energy['a_el_bins'][value_bins]

        self.a_e_survival_prob = np.zeros(self.num_mc_events, dtype=np.float32)
        for i in tqdm.tqdm(xrange(self.num_mc_events)):
            current_random_num = np.random.random()*bin_width + random_from_cdf[i]
            
            # current random number is lifetime whch we need to convert
            # draw from z array to make sure they are connected!
            self.a_e_survival_prob[i] = np.exp(-(config_xe1t.z_gate - self.a_mc_z[i]) / config_xe1t.e_drift_velocity / current_random_num)
    
        if view_energy_spectrum:
            plt.hist(self.a_e_survival_prob, bins=100)
            plt.show()





        # -----------------------------------------
        #  X, Y
        # -----------------------------------------
    
        """
        bin_width_x = self.d_mc_positions['x_bin_edges'][1] - self.d_mc_positions['x_bin_edges'][0]
        bin_width_y = self.d_mc_positions['y_bin_edges'][1] - self.d_mc_positions['y_bin_edges'][0]

        cdf = np.cumsum(self.d_mc_positions['xy_map'].ravel())
        cdf = cdf / cdf[-1]
        values = np.random.rand(self.num_mc_events)
        value_bins = np.searchsorted(cdf, values)
        x_idx, y_idx = np.unravel_index(value_bins, (len(self.d_mc_positions['x_bin_edges'])-1, len(self.d_mc_positions['y_bin_edges'])-1))

        self.a_mc_x = np.zeros(self.num_mc_events, dtype=np.float32)
        self.a_mc_y = np.zeros(self.num_mc_events, dtype=np.float32)
        for i in tqdm.tqdm(xrange(self.num_mc_events)):
            current_random_num_x = np.random.random()*bin_width_x + self.d_mc_positions['x_bin_edges'][x_idx[i]]
            current_random_num_y = np.random.random()*bin_width_y + self.d_mc_positions['y_bin_edges'][y_idx[i]]
            
            
            self.a_mc_x[i] = current_random_num_x
            self.a_mc_y[i] = current_random_num_y
        """
    
        if view_energy_spectrum:
            plt.hist2d(self.a_mc_x, self.a_mc_y, bins=100)
            plt.show()
    
    


        # -----------------------------------------
        #  Get array of ER bands S1 and S2
        # -----------------------------------------


        bin_width_s1 = self.d_er_band['er_band_s1_edges'][1] - self.d_er_band['er_band_s1_edges'][0]
        bin_width_log = self.d_er_band['er_band_log_edges'][1] - self.d_er_band['er_band_log_edges'][0]

        cdf = np.cumsum(self.d_er_band['er_band_hist'].ravel())
        cdf = cdf / cdf[-1]
        values = np.random.rand(self.num_mc_events)
        value_bins = np.searchsorted(cdf, values)
        s1_idx, log_idx = np.unravel_index(value_bins, (len(self.d_er_band['er_band_s1_edges'])-1, len(self.d_er_band['er_band_log_edges'])-1))

        self.a_er_s1 = np.zeros(self.num_mc_events, dtype=np.float32)
        self.a_er_log = np.zeros(self.num_mc_events, dtype=np.float32)
        for i in tqdm.tqdm(xrange(self.num_mc_events)):
            current_random_num_s1 = np.random.random()*bin_width_s1 + self.d_er_band['er_band_s1_edges'][s1_idx[i]]
            current_random_num_log = np.random.random()*bin_width_log + self.d_er_band['er_band_log_edges'][log_idx[i]]
            
            
            self.a_er_s1[i] = current_random_num_s1
            self.a_er_log[i] = current_random_num_log
    
        if view_energy_spectrum:
            plt.hist2d(self.a_er_s1, self.a_er_log, bins=[self.d_er_band['er_band_s1_edges'], self.d_er_band['er_band_log_edges']])
            plt.show()






    def load_mc_data(self, view_corner_plot):

        # KDE method
        """
        self.d_mc_inputs = pickle.load(open('%smc_inputs.p' % config_xe1t.path_to_fit_inputs, 'r'))

        a_random_samples = self.d_mc_inputs['kde'].sample(self.num_mc_events)
        a_random_samples = self.d_mc_inputs['scaler'].inverse_transform(a_random_samples)
        #print a_random_samples.shape
    
        d_mc_arrays = {'a_mc_energy':a_random_samples[:,0], 'a_mc_x':a_random_samples[:,1], 'a_mc_y':a_random_samples[:,2], 'a_mc_z':a_random_samples[:,3]}
        d_mc_arrays = pd.DataFrame(d_mc_arrays)
    
        # make cuts based on fiducial volume and energy values
        d_mc_arrays = d_mc_arrays[(d_mc_arrays['a_mc_energy'] > config_xe1t.l_energy_settings[1]) & (d_mc_arrays['a_mc_energy'] < config_xe1t.l_energy_settings[2]) & (np.abs(d_mc_arrays['a_mc_x']) < config_xe1t.max_r) & (np.abs(d_mc_arrays['a_mc_y']) < config_xe1t.max_r) & ((d_mc_arrays['a_mc_x']**2 + d_mc_arrays['a_mc_y']**2) < config_xe1t.max_r**2) & (d_mc_arrays['a_mc_z'] > config_xe1t.min_z) & (d_mc_arrays['a_mc_z'] < config_xe1t.max_z)]
    
        if view_corner_plot:
            s_path_to_plots = './plots/supporting/mc_inputs/'

            if not os.path.isdir(s_path_to_plots):
                os.mkdir(s_path_to_plots)

            a_samples_kde = np.asarray([d_mc_arrays['a_mc_energy'], d_mc_arrays['a_mc_x'], d_mc_arrays['a_mc_y'], d_mc_arrays['a_mc_z']])

            fig_corner_kde = corner.corner(a_samples_kde.T, labels=['Energy [keV]', 'X [cm]', 'Y [cm]', 'Z [cm]'])
            
            fig_corner_kde.savefig('%smc_kde_corner_plot.png' % (s_path_to_plots))
            
        self.a_mc_energy, self.a_mc_x, self.a_mc_y, self.a_mc_z = np.asarray(d_mc_arrays['a_mc_energy'], dtype=np.float32), np.asarray(d_mc_arrays['a_mc_x'], dtype=np.float32), np.asarray(d_mc_arrays['a_mc_y'], dtype=np.float32), np.asarray(d_mc_arrays['a_mc_z'], dtype=np.float32)
        """
    
        self.d_mc_energy = pickle.load(open('%sambe_mc.p' % config_xe1t.path_to_fit_inputs, 'r'))
        self.d_mc_positions = pickle.load(open('%smc_maps.p' % config_xe1t.path_to_fit_inputs, 'r'))
        self.d_er_band = pickle.load(open('%ser_band.p' % config_xe1t.path_to_fit_inputs, 'r'))



    def load_correction_maps(self):
        self.d_corrections = pickle.load(open('%ssignal_correction_maps.p' % config_xe1t.path_to_fit_inputs, 'r'))
    
        self.bin_edges_r2 = np.asarray(self.d_corrections['s1']['r2_bin_edges'], dtype=np.float32)
        self.bin_edges_z = np.asarray(self.d_corrections['s1']['z_bin_edges'], dtype=np.float32)
        self.s1_correction_map = np.asarray(self.d_corrections['s1']['map'], dtype=np.float32).T
        #self.s1_correction_map = np.rot90(self.s1_correction_map)
        #self.s1_correction_map = np.rot90(self.s1_correction_map)
        #self.s1_correction_map = np.flipud(self.s1_correction_map)
        self.s1_correction_map = self.s1_correction_map.flatten()

        self.bin_edges_x = np.asarray(self.d_corrections['s2']['x_bin_edges'], dtype=np.float32)
        self.bin_edges_y = np.asarray(self.d_corrections['s2']['y_bin_edges'], dtype=np.float32)
        self.s2_correction_map = np.asarray(self.d_corrections['s2']['map'], dtype=np.float32).T
        #self.s2_correction_map = np.rot90(self.s2_correction_map)
        #self.s2_correction_map = np.flipud(self.s2_correction_map)
        self.s2_correction_map = self.s2_correction_map.flatten()
    
        #plt.pcolor(self.bin_edges_x, self.bin_edges_y, self.s2_correction_map)
        #plt.show()



    def get_prior_log_likelihood_nest_parameter(self, par_value, par_name):
        # none of the parameters can be less than zero
        if par_value < 0:
            return -np.inf

        if par_value > self.nest_lindhard_model['values'][par_name]:
            return norm.logpdf(par_value, self.nest_lindhard_model['values'][par_name], self.nest_lindhard_model['uncertainties'][par_name]['high'])
        else:
            return norm.logpdf(par_value, self.nest_lindhard_model['values'][par_name], self.nest_lindhard_model['uncertainties'][par_name]['low'])
            
            

    def get_prior_log_likelihood_nuissance(self, likelihoodNuissance):
        if likelihoodNuissance > 1e-550:
            return np.log(likelihoodNuissance)
        else:
            return -np.inf



    def get_g1_default(self, g1_value):
        return norm.pdf(g1_value, self.g1_value, self.g1_uncertainty), g1_value
    
    
    def get_extraction_efficiency_default(self, extraction_efficiency_value):
        return norm.pdf(extraction_efficiency_value, self.extraction_efficiency_value, self.extraction_efficiency_uncertainty), extraction_efficiency_value


    # get likelihood and gas gain given random variable (nuissance parameter)
    # gasGainRV should be normally distributed
    def get_gas_gain_default(self, gas_gain_mean):
        return norm.pdf(gas_gain_mean, self.gas_gain_value, self.gas_gain_uncertainty), gas_gain_mean


    # get likelihood and gas gain width given random variable (nuissance parameter)
    # gasGainWidthRV should be normally distributed
    def get_gas_gain_width_default(self, gas_gain_width_value):
        return norm.pdf(gas_gain_width_value, self.gas_gain_width, self.gas_gain_width_uncertainty), gas_gain_width_value


    # get likelihood and spe res width given random variable (nuissance parameter)
    # gasGainWidthRV should be normally distributed
    def get_spe_res_default(self, spe_res):
        return norm.pdf(spe_res, self.spe_res_value, self.spe_res_uncertainty), spe_res



    def get_prior_log_likelihood_dpe_prob(self, dpe_prob):
        if 0.17 < dpe_prob < 0.24:
            return 0
        else:
            return -np.inf



    def get_prior_log_likelihood_resolution(self, intrinsicResolution):
        if intrinsicResolution < 0 or intrinsicResolution > 1.5:
            return -np.inf
        else:
            return 0



    
    def get_prior_log_likelihood_probability(self, prob):
        if prob < 0 or prob > 1:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_gaussian_prior(self, par):
        return norm.logpdf(par)



    def get_prior_log_likelihood_greater_than_zero(self, par_value):
        if par_value < 0:
            return -np.inf
        else:
            return 0.


    def get_prior_log_likelihood_ms_pars(self, ms_par_0, ms_par_1):
        if (ms_par_0 < 0) or (ms_par_1 < 0):
            return -np.inf
        else:
            return norm.logpdf(ms_par_0, self.ms_par_0, self.ms_par_0_unc) + norm.logpdf(ms_par_1, self.ms_par_1, self.ms_par_1_unc)



    # band matching
    def ln_likelihood_full_matching_band(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par, cut_acceptance_par, prob_bkg, scale_par, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------

        #start_time_full = time.time()

        prior_ln_likelihood = 0
        matching_ln_likelihood = 0


        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(w_value, 'w_value')
        
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(alpha, 'alpha')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(zeta, 'zeta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(beta, 'beta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(delta, 'delta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(eta, 'eta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(lamb, 'lambda')

        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(gamma, 'gamma')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(kappa, 'kappa')
        
        # gamma and kappa free
        #prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(gamma)
        #prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(kappa)

        #prior_ln_likelihood += self.get_prior_log_likelihood_probability(scale_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(scale_par)



        # priors of detector variables
        current_likelihood, g1_value = self.get_g1_default(g1_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        current_likelihood, extraction_efficiency = self.get_extraction_efficiency_default(extraction_efficiency_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_value = self.get_gas_gain_default(gas_gain_mean_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_width = self.get_gas_gain_width_default(gas_gain_width_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        prior_ln_likelihood += self.get_prior_log_likelihood_dpe_prob(dpe_prob)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s1_bias_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s1_smearing_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s2_bias_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s2_smearing_par)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(prob_bkg)
        
        #prior_ln_likelihood += self.get_prior_log_likelihood_probability(acceptance_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_gaussian_prior(acceptance_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_gaussian_prior(cut_acceptance_par)
        

        # if prior is -inf then don't bother with MC
        #print 'removed prior infinity catch temporarily'
        if not np.isfinite(prior_ln_likelihood) and not draw_fit:
            return -np.inf



        # -----------------------------------------------
        # -----------------------------------------------
        # run MC
        # -----------------------------------------------
        # -----------------------------------------------
        

        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        
        prob_bkg = np.asarray(prob_bkg, dtype=np.float32)

        w_value = np.asarray(w_value, dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        zeta = np.asarray(zeta, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)
        gamma = np.asarray(gamma, dtype=np.float32)
        delta = np.asarray(delta, dtype=np.float32)
        kappa = np.asarray(kappa, dtype=np.float32)
        eta = np.asarray(eta, dtype=np.float32)
        lamb = np.asarray(lamb, dtype=np.float32)

        g1_value = np.asarray(g1_value, dtype=np.float32)
        extraction_efficiency = np.asarray(extraction_efficiency, dtype=np.float32)
        gas_gain_value = np.asarray(gas_gain_value, dtype=np.float32)
        gas_gain_width = np.asarray(gas_gain_width, dtype=np.float32)

        dpe_prob = np.asarray(dpe_prob, dtype=np.float32)
        s1_bias_par = np.asarray(s1_bias_par, dtype=np.float32)
        s1_smearing_par = np.asarray(s1_smearing_par, dtype=np.float32)
        s2_bias_par = np.asarray(s2_bias_par, dtype=np.float32)
        s2_smearing_par = np.asarray(s2_smearing_par, dtype=np.float32)
        acceptance_par = np.asarray(acceptance_par, dtype=np.float32)
        
        cut_acceptance_s1_intercept = np.asarray(self.cut_acceptance_s1_intercept + cut_acceptance_par*self.cut_acceptance_s1_intercept_uncertainty, dtype=np.float32)
        cut_acceptance_s1_slope = np.asarray(self.cut_acceptance_s1_slope + cut_acceptance_par*self.cut_acceptance_s1_slope_uncertainty, dtype=np.float32)
        
        cut_acceptance_s2_intercept = np.asarray(self.cut_acceptance_s2_intercept + cut_acceptance_par*self.cut_acceptance_s2_intercept_uncertainty, dtype=np.float32)
        cut_acceptance_s2_slope = np.asarray(self.cut_acceptance_s2_slope + cut_acceptance_par*self.cut_acceptance_s2_slope_uncertainty, dtype=np.float32)
        
        num_pts_s1bs = np.asarray(len(self.a_s1bs_s1s), dtype=np.int32)
        num_pts_s2bs = np.asarray(len(self.a_s2bs_s2s), dtype=np.int32)
        num_pts_s1pf = np.asarray(len(self.a_s1pf_s1s), dtype=np.int32)
        
        num_bins_r2 = np.asarray(len(self.bin_edges_r2)-1, dtype=np.int32)
        num_bins_z = np.asarray(len(self.bin_edges_z)-1, dtype=np.int32)
        num_bins_x = np.asarray(len(self.bin_edges_x)-1, dtype=np.int32)
        num_bins_y = np.asarray(len(self.bin_edges_y)-1, dtype=np.int32)

        # for histogram binning
        num_bins_s1 = np.asarray(len(self.a_s1_bin_edges)-1, dtype=np.int32)
        num_bins_log = np.asarray(len(self.a_log_bin_edges)-1, dtype=np.int32)
        a_hist_2d = np.zeros(num_bins_log*num_bins_s1, dtype=np.float32)
        
        num_loops = np.asarray(self.num_loops, dtype=np.int32)



        mean_field = np.asarray(self.d_cathode_voltages_to_field[self.cathode_setting], dtype=np.float32)
        
        
        
        
        #start_time_mc = time.time()
        tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['gpu_energies'], d_gpu_local_info['gpu_x_positions'], d_gpu_local_info['gpu_y_positions'], d_gpu_local_info['gpu_z_positions'], d_gpu_local_info['gpu_e_survival_prob'], drv.In(prob_bkg), d_gpu_local_info['gpu_er_band_s1'], d_gpu_local_info['gpu_er_band_log'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(dpe_prob), drv.In(s1_bias_par), drv.In(s1_smearing_par), drv.In(s2_bias_par), drv.In(s2_smearing_par), drv.In(acceptance_par), drv.In(num_pts_s1bs), d_gpu_local_info['gpu_s1bs_s1s'], d_gpu_local_info['gpu_s1bs_lb_bias'], d_gpu_local_info['gpu_s1bs_ub_bias'], d_gpu_local_info['gpu_s1bs_lb_smearing'], d_gpu_local_info['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_gpu_local_info['gpu_s2bs_s2s'], d_gpu_local_info['gpu_s2bs_lb_bias'], d_gpu_local_info['gpu_s2bs_ub_bias'], d_gpu_local_info['gpu_s2bs_lb_smearing'], d_gpu_local_info['gpu_s2bs_ub_smearing'], drv.In(num_pts_s1pf), d_gpu_local_info['gpu_s1pf_s1s'], d_gpu_local_info['gpu_s1pf_lb_acc'], d_gpu_local_info['gpu_s1pf_mean_acc'], d_gpu_local_info['gpu_s1pf_ub_acc'], drv.In(cut_acceptance_s1_intercept), drv.In(cut_acceptance_s1_slope), drv.In(cut_acceptance_s2_intercept), drv.In(cut_acceptance_s2_slope), drv.In(num_bins_r2), d_gpu_local_info['gpu_bin_edges_r2'], drv.In(num_bins_z), d_gpu_local_info['gpu_bin_edges_z'], d_gpu_local_info['gpu_s1_correction_map'], drv.In(num_bins_x), d_gpu_local_info['gpu_bin_edges_x'], drv.In(num_bins_y), d_gpu_local_info['gpu_bin_edges_y'], d_gpu_local_info['gpu_s2_correction_map'], drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d), drv.In(num_loops))

        d_gpu_local_info['function_to_call'](*tArgs, **self.d_gpu_scale)
        
        
        #print 'MC time: %f' % (time.time() - start_time_mc)
        #start_time_tot_ll = time.time()

        a_s1_s2_mc = np.reshape(a_hist_2d, (len(self.a_log_bin_edges)-1, len(self.a_s1_bin_edges)-1)).T

        sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
        if sum_mc == 0:
            return -np.inf


        #scale_par *= float(self.num_mc_events*self.num_loops) / sum_mc
        #a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information['num_data_pts']/float(self.num_mc_events*self.num_loops))
        
        a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information['num_data_pts']/float(self.num_mc_events*self.num_loops))

        # likelihood for band
        if draw_fit:

            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

            s1_s2_data_plot = np.rot90(self.d_coincidence_data_information['a_log_s2_s1'])
            s1_s2_data_plot = np.flipud(s1_s2_data_plot)
            ax1.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')


            
            s1_s2_mc_plot = np.rot90(a_s1_s2_mc)
            s1_s2_mc_plot = np.flipud(s1_s2_mc_plot)
            
            
            #print 'changed mc to division temporarily'
            #for (x, y), val in np.ndenumerate(s1_s2_mc_plot):
            #    if val != 0:
            #        s1_s2_data_plot[x, y] -= val
            
            
            ax1.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_data_plot)
            ax2.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_mc_plot)
            ax2.set_xlabel(r'$S1 [PE]$')
            ax2.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')
            #plt.colorbar()


            f_flat, (ax_s1, ax_log) = plt.subplots(2)
            
            flat_s1_data = np.sum(s1_s2_data_plot, axis=0)
            flat_s1_mc = np.sum(s1_s2_mc_plot, axis=0)
            ax_s1.errorbar((self.a_s1_bin_edges[1:]+self.a_s1_bin_edges[:-1])/2., flat_s1_data, yerr=flat_s1_data**0.5, fmt='bo', label='Data')
            ax_s1.errorbar((self.a_s1_bin_edges[1:]+self.a_s1_bin_edges[:-1])/2., flat_s1_mc, yerr=flat_s1_mc**0.5, fmt='ro', label='MC')
            ax_s1.set_xlabel('S1 [PE]')
            ax_s1.set_ylabel('Counts')
            ax_s1.legend(loc='best')
            
            flat_log_data = np.sum(s1_s2_data_plot, axis=1)
            flat_log_mc = np.sum(s1_s2_mc_plot, axis=1)
            ax_log.errorbar((self.a_log_bin_edges[1:]+self.a_log_bin_edges[:-1])/2., flat_log_data, yerr=flat_log_data**0.5, fmt='bo')
            ax_log.errorbar((self.a_log_bin_edges[1:]+self.a_log_bin_edges[:-1])/2., flat_log_mc, yerr=flat_log_mc**0.5, fmt='ro')
            ax_log.set_xlabel(r'$Log_{10}(\frac{S2}{S1})$')
            ax_log.set_ylabel('Counts')

            f.savefig('./temp_results/2d_hist_%.3f_kV_%d_deg.png' % (self.cathode_setting, self.degree_setting))
            f_flat.savefig('./temp_results/1d_hists_%.3f_kV_%d_deg.png' % (self.cathode_setting, self.degree_setting))

        flat_s1_log_data = np.asarray(self.d_coincidence_data_information['a_log_s2_s1'].flatten(), dtype=np.float32)
        flat_s1_log_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
        

        logLikelihoodMatching = c_log_likelihood(flat_s1_log_data, flat_s1_log_mc, len(flat_s1_log_data), 1e-6)
        
        matching_ln_likelihood += logLikelihoodMatching

        #print logLikelihoodMatching, prior_ln_likelihood
        #print 'likelihood calculation time: %f' % (time.time() - start_time_tot_ll)
        #print total_ln_likelihood

        #print 'full function time: %f' % (time.time() - start_time_full)
        total_ln_likelihood = matching_ln_likelihood + prior_ln_likelihood

        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor



        if np.isnan(total_ln_likelihood):
            return -np.inf
        else:
            return total_ln_likelihood


        
    def wrapper_ln_likelihood_full_matching_band(self, a_parameters, kwargs={}):
        
        return self.ln_likelihood_full_matching_band(*a_parameters, **kwargs)





    def ln_likelihood_full_matching_band_with_free_pax_eff(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par_0, acceptance_par_1, cut_acceptance_par, prob_bkg, scale_par, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------

        #start_time_full = time.time()

        prior_ln_likelihood = 0
        matching_ln_likelihood = 0


        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(w_value, 'w_value')
        
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(alpha, 'alpha')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(zeta, 'zeta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(beta, 'beta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(delta, 'delta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(eta, 'eta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(lamb, 'lambda')

        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(gamma, 'gamma')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(kappa, 'kappa')
        
        # gamma and kappa free
        #prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(gamma)
        #prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(kappa)

        #prior_ln_likelihood += self.get_prior_log_likelihood_probability(scale_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(scale_par)



        # priors of detector variables
        current_likelihood, g1_value = self.get_g1_default(g1_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        current_likelihood, extraction_efficiency = self.get_extraction_efficiency_default(extraction_efficiency_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_value = self.get_gas_gain_default(gas_gain_mean_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_width = self.get_gas_gain_width_default(gas_gain_width_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        prior_ln_likelihood += self.get_prior_log_likelihood_dpe_prob(dpe_prob)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s1_bias_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s1_smearing_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s2_bias_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s2_smearing_par)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(prob_bkg)
        
        #prior_ln_likelihood += self.get_prior_log_likelihood_probability(acceptance_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(acceptance_par_0)
        prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(acceptance_par_1)
        prior_ln_likelihood += self.get_prior_log_likelihood_gaussian_prior(cut_acceptance_par)
        

        # if prior is -inf then don't bother with MC
        #print 'removed prior infinity catch temporarily'
        if not np.isfinite(prior_ln_likelihood) and not draw_fit:
            return -np.inf



        # -----------------------------------------------
        # -----------------------------------------------
        # run MC
        # -----------------------------------------------
        # -----------------------------------------------
        

        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        
        prob_bkg = np.asarray(prob_bkg, dtype=np.float32)

        w_value = np.asarray(w_value, dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        zeta = np.asarray(zeta, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)
        gamma = np.asarray(gamma, dtype=np.float32)
        delta = np.asarray(delta, dtype=np.float32)
        kappa = np.asarray(kappa, dtype=np.float32)
        eta = np.asarray(eta, dtype=np.float32)
        lamb = np.asarray(lamb, dtype=np.float32)

        g1_value = np.asarray(g1_value, dtype=np.float32)
        extraction_efficiency = np.asarray(extraction_efficiency, dtype=np.float32)
        gas_gain_value = np.asarray(gas_gain_value, dtype=np.float32)
        gas_gain_width = np.asarray(gas_gain_width, dtype=np.float32)

        dpe_prob = np.asarray(dpe_prob, dtype=np.float32)
        s1_bias_par = np.asarray(s1_bias_par, dtype=np.float32)
        s1_smearing_par = np.asarray(s1_smearing_par, dtype=np.float32)
        s2_bias_par = np.asarray(s2_bias_par, dtype=np.float32)
        s2_smearing_par = np.asarray(s2_smearing_par, dtype=np.float32)
        acceptance_par_0 = np.asarray(acceptance_par_0, dtype=np.float32)
        acceptance_par_1 = np.asarray(acceptance_par_1, dtype=np.float32)
        
        cut_acceptance_s1_intercept = np.asarray(self.cut_acceptance_s1_intercept + cut_acceptance_par*self.cut_acceptance_s1_intercept_uncertainty, dtype=np.float32)
        cut_acceptance_s1_slope = np.asarray(self.cut_acceptance_s1_slope + cut_acceptance_par*self.cut_acceptance_s1_slope_uncertainty, dtype=np.float32)
        
        cut_acceptance_s2_intercept = np.asarray(self.cut_acceptance_s2_intercept + cut_acceptance_par*self.cut_acceptance_s2_intercept_uncertainty, dtype=np.float32)
        cut_acceptance_s2_slope = np.asarray(self.cut_acceptance_s2_slope + cut_acceptance_par*self.cut_acceptance_s2_slope_uncertainty, dtype=np.float32)
        
        num_pts_s1bs = np.asarray(len(self.a_s1bs_s1s), dtype=np.int32)
        num_pts_s2bs = np.asarray(len(self.a_s2bs_s2s), dtype=np.int32)
        num_pts_s1pf = np.asarray(len(self.a_s1pf_s1s), dtype=np.int32)
        
        num_bins_r2 = np.asarray(len(self.bin_edges_r2)-1, dtype=np.int32)
        num_bins_z = np.asarray(len(self.bin_edges_z)-1, dtype=np.int32)
        num_bins_x = np.asarray(len(self.bin_edges_x)-1, dtype=np.int32)
        num_bins_y = np.asarray(len(self.bin_edges_y)-1, dtype=np.int32)

        # for histogram binning
        num_bins_s1 = np.asarray(len(self.a_s1_bin_edges)-1, dtype=np.int32)
        num_bins_log = np.asarray(len(self.a_log_bin_edges)-1, dtype=np.int32)
        a_hist_2d = np.zeros(num_bins_log*num_bins_s1, dtype=np.float32)
        
        num_loops = np.asarray(self.num_loops, dtype=np.int32)



        mean_field = np.asarray(self.d_cathode_voltages_to_field[self.cathode_setting], dtype=np.float32)
        
        
        
        
        #start_time_mc = time.time()
        tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['gpu_energies'], d_gpu_local_info['gpu_x_positions'], d_gpu_local_info['gpu_y_positions'], d_gpu_local_info['gpu_z_positions'], d_gpu_local_info['gpu_e_survival_prob'], drv.In(prob_bkg), d_gpu_local_info['gpu_er_band_s1'], d_gpu_local_info['gpu_er_band_log'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(dpe_prob), drv.In(s1_bias_par), drv.In(s1_smearing_par), drv.In(s2_bias_par), drv.In(s2_smearing_par), drv.In(acceptance_par_0), drv.In(acceptance_par_1), drv.In(num_pts_s1bs), d_gpu_local_info['gpu_s1bs_s1s'], d_gpu_local_info['gpu_s1bs_lb_bias'], d_gpu_local_info['gpu_s1bs_ub_bias'], d_gpu_local_info['gpu_s1bs_lb_smearing'], d_gpu_local_info['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_gpu_local_info['gpu_s2bs_s2s'], d_gpu_local_info['gpu_s2bs_lb_bias'], d_gpu_local_info['gpu_s2bs_ub_bias'], d_gpu_local_info['gpu_s2bs_lb_smearing'], d_gpu_local_info['gpu_s2bs_ub_smearing'], drv.In(cut_acceptance_s1_intercept), drv.In(cut_acceptance_s1_slope), drv.In(cut_acceptance_s2_intercept), drv.In(cut_acceptance_s2_slope), drv.In(num_bins_r2), d_gpu_local_info['gpu_bin_edges_r2'], drv.In(num_bins_z), d_gpu_local_info['gpu_bin_edges_z'], d_gpu_local_info['gpu_s1_correction_map'], drv.In(num_bins_x), d_gpu_local_info['gpu_bin_edges_x'], drv.In(num_bins_y), d_gpu_local_info['gpu_bin_edges_y'], d_gpu_local_info['gpu_s2_correction_map'], drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d), drv.In(num_loops))

        d_gpu_local_info['function_to_call'](*tArgs, **self.d_gpu_scale)
        
        
        #print 'MC time: %f' % (time.time() - start_time_mc)
        #start_time_tot_ll = time.time()

        a_s1_s2_mc = np.reshape(a_hist_2d, (len(self.a_log_bin_edges)-1, len(self.a_s1_bin_edges)-1)).T

        sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
        if sum_mc == 0:
            return -np.inf


        #scale_par *= float(self.num_mc_events*self.num_loops) / sum_mc
        #a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information['num_data_pts']/float(self.num_mc_events*self.num_loops))
        
        a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information['num_data_pts']/float(self.num_mc_events*self.num_loops))

        # likelihood for band with free pax eff
        if draw_fit:

            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

            s1_s2_data_plot = np.rot90(self.d_coincidence_data_information['a_log_s2_s1'])
            s1_s2_data_plot = np.flipud(s1_s2_data_plot)
            ax1.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')


            
            s1_s2_mc_plot = np.rot90(a_s1_s2_mc)
            s1_s2_mc_plot = np.flipud(s1_s2_mc_plot)
            
            
            #print 'changed mc to division temporarily'
            #for (x, y), val in np.ndenumerate(s1_s2_mc_plot):
            #    if val != 0:
            #        s1_s2_data_plot[x, y] -= val
            
            
            ax1.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_data_plot)
            ax2.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_mc_plot)
            ax2.set_xlabel(r'$S1 [PE]$')
            ax2.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')
            #plt.colorbar()


            f_flat, (ax_s1, ax_log) = plt.subplots(2)
            
            flat_s1_data = np.sum(s1_s2_data_plot, axis=0)
            flat_s1_mc = np.sum(s1_s2_mc_plot, axis=0)
            ax_s1.errorbar((self.a_s1_bin_edges[1:]+self.a_s1_bin_edges[:-1])/2., flat_s1_data, yerr=flat_s1_data**0.5, fmt='bo')
            ax_s1.errorbar((self.a_s1_bin_edges[1:]+self.a_s1_bin_edges[:-1])/2., flat_s1_mc, yerr=flat_s1_mc**0.5, fmt='ro')
            ax_s1.set_xlabel('S1 [PE]')
            ax_s1.set_ylabel('Counts')
            
            flat_log_data = np.sum(s1_s2_data_plot, axis=1)
            flat_log_mc = np.sum(s1_s2_mc_plot, axis=1)
            ax_log.errorbar((self.a_log_bin_edges[1:]+self.a_log_bin_edges[:-1])/2., flat_log_data, yerr=flat_log_data**0.5, fmt='bo')
            ax_log.errorbar((self.a_log_bin_edges[1:]+self.a_log_bin_edges[:-1])/2., flat_log_mc, yerr=flat_log_mc**0.5, fmt='ro')
            ax_log.set_xlabel(r'$Log_{10}(\frac{S2}{S1})$')
            ax_log.set_ylabel('Counts')

            f.savefig('./temp_results/2d_hist_%.3f_kV_%d_deg.png' % (self.cathode_setting, self.degree_setting))
            f_flat.savefig('./temp_results/1d_hists_%.3f_kV_%d_deg.png' % (self.cathode_setting, self.degree_setting))

        flat_s1_log_data = np.asarray(self.d_coincidence_data_information['a_log_s2_s1'].flatten(), dtype=np.float32)
        flat_s1_log_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
        

        logLikelihoodMatching = c_log_likelihood(flat_s1_log_data, flat_s1_log_mc, len(flat_s1_log_data), 1e-6)
        
        matching_ln_likelihood += logLikelihoodMatching

        #print logLikelihoodMatching, prior_ln_likelihood
        #print 'likelihood calculation time: %f' % (time.time() - start_time_tot_ll)
        #print total_ln_likelihood

        #print 'full function time: %f' % (time.time() - start_time_full)
        total_ln_likelihood = matching_ln_likelihood + prior_ln_likelihood

        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor



        if np.isnan(total_ln_likelihood):
            return -np.inf
        else:
            return total_ln_likelihood


        
    def wrapper_ln_likelihood_full_matching_band_with_free_pax_eff(self, a_parameters, kwargs={}):
        
        return self.ln_likelihood_full_matching_band_with_free_pax_eff(*a_parameters, **kwargs)




    # band matching with ms scale
    def ln_likelihood_full_matching_band_with_ms_scale(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par, cut_acceptance_par, ms_par_0, ms_par_1, prob_bkg, scale_par, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------

        #start_time_full = time.time()

        prior_ln_likelihood = 0
        matching_ln_likelihood = 0


        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(w_value, 'w_value')
        
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(alpha, 'alpha')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(zeta, 'zeta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(beta, 'beta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(delta, 'delta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(eta, 'eta')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(lamb, 'lambda')

        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(gamma, 'gamma')
        prior_ln_likelihood += self.get_prior_log_likelihood_nest_parameter(kappa, 'kappa')
        
        # gamma and kappa free
        #prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(gamma)
        #prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(kappa)

        #prior_ln_likelihood += self.get_prior_log_likelihood_probability(scale_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_greater_than_zero(scale_par)



        # priors of detector variables
        current_likelihood, g1_value = self.get_g1_default(g1_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        current_likelihood, extraction_efficiency = self.get_extraction_efficiency_default(extraction_efficiency_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_value = self.get_gas_gain_default(gas_gain_mean_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_width = self.get_gas_gain_width_default(gas_gain_width_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        prior_ln_likelihood += self.get_prior_log_likelihood_dpe_prob(dpe_prob)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s1_bias_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s1_smearing_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s2_bias_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(s2_smearing_par)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(prob_bkg)
        
        #prior_ln_likelihood += self.get_prior_log_likelihood_probability(acceptance_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_gaussian_prior(acceptance_par)
        prior_ln_likelihood += self.get_prior_log_likelihood_gaussian_prior(cut_acceptance_par)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_ms_pars(ms_par_0, ms_par_1)
        

        # if prior is -inf then don't bother with MC
        #print 'removed prior infinity catch temporarily'
        if not np.isfinite(prior_ln_likelihood) and not draw_fit:
            return -np.inf



        # -----------------------------------------------
        # -----------------------------------------------
        # run MC
        # -----------------------------------------------
        # -----------------------------------------------
        

        num_trials = np.asarray(self.num_mc_events, dtype=np.int32)
        
        prob_bkg = np.asarray(prob_bkg, dtype=np.float32)

        w_value = np.asarray(w_value, dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        zeta = np.asarray(zeta, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)
        gamma = np.asarray(gamma, dtype=np.float32)
        delta = np.asarray(delta, dtype=np.float32)
        kappa = np.asarray(kappa, dtype=np.float32)
        eta = np.asarray(eta, dtype=np.float32)
        lamb = np.asarray(lamb, dtype=np.float32)

        g1_value = np.asarray(g1_value, dtype=np.float32)
        extraction_efficiency = np.asarray(extraction_efficiency, dtype=np.float32)
        gas_gain_value = np.asarray(gas_gain_value, dtype=np.float32)
        gas_gain_width = np.asarray(gas_gain_width, dtype=np.float32)

        dpe_prob = np.asarray(dpe_prob, dtype=np.float32)
        s1_bias_par = np.asarray(s1_bias_par, dtype=np.float32)
        s1_smearing_par = np.asarray(s1_smearing_par, dtype=np.float32)
        s2_bias_par = np.asarray(s2_bias_par, dtype=np.float32)
        s2_smearing_par = np.asarray(s2_smearing_par, dtype=np.float32)
        acceptance_par = np.asarray(acceptance_par, dtype=np.float32)
        
        cut_acceptance_s1_intercept = np.asarray(self.cut_acceptance_s1_intercept + cut_acceptance_par*self.cut_acceptance_s1_intercept_uncertainty, dtype=np.float32)
        cut_acceptance_s1_slope = np.asarray(self.cut_acceptance_s1_slope + cut_acceptance_par*self.cut_acceptance_s1_slope_uncertainty, dtype=np.float32)
        
        cut_acceptance_s2_intercept = np.asarray(self.cut_acceptance_s2_intercept + cut_acceptance_par*self.cut_acceptance_s2_intercept_uncertainty, dtype=np.float32)
        cut_acceptance_s2_slope = np.asarray(self.cut_acceptance_s2_slope + cut_acceptance_par*self.cut_acceptance_s2_slope_uncertainty, dtype=np.float32)
        
        ms_par_0 = np.asarray(ms_par_0, dtype=np.float32)
        ms_par_1 = np.asarray(ms_par_1, dtype=np.float32)
        
        num_pts_s1bs = np.asarray(len(self.a_s1bs_s1s), dtype=np.int32)
        num_pts_s2bs = np.asarray(len(self.a_s2bs_s2s), dtype=np.int32)
        num_pts_s1pf = np.asarray(len(self.a_s1pf_s1s), dtype=np.int32)
        
        num_bins_r2 = np.asarray(len(self.bin_edges_r2)-1, dtype=np.int32)
        num_bins_z = np.asarray(len(self.bin_edges_z)-1, dtype=np.int32)
        num_bins_x = np.asarray(len(self.bin_edges_x)-1, dtype=np.int32)
        num_bins_y = np.asarray(len(self.bin_edges_y)-1, dtype=np.int32)

        # for histogram binning
        num_bins_s1 = np.asarray(len(self.a_s1_bin_edges)-1, dtype=np.int32)
        num_bins_log = np.asarray(len(self.a_log_bin_edges)-1, dtype=np.int32)
        a_hist_2d = np.zeros(num_bins_log*num_bins_s1, dtype=np.float32)
        
        num_loops = np.asarray(self.num_loops, dtype=np.int32)



        mean_field = np.asarray(self.d_cathode_voltages_to_field[self.cathode_setting], dtype=np.float32)
        
        
        
        
        #start_time_mc = time.time()
        tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['gpu_energies'], d_gpu_local_info['gpu_x_positions'], d_gpu_local_info['gpu_y_positions'], d_gpu_local_info['gpu_z_positions'], d_gpu_local_info['gpu_e_survival_prob'], drv.In(prob_bkg), d_gpu_local_info['gpu_er_band_s1'], d_gpu_local_info['gpu_er_band_log'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(dpe_prob), drv.In(s1_bias_par), drv.In(s1_smearing_par), drv.In(s2_bias_par), drv.In(s2_smearing_par), drv.In(acceptance_par), drv.In(num_pts_s1bs), d_gpu_local_info['gpu_s1bs_s1s'], d_gpu_local_info['gpu_s1bs_lb_bias'], d_gpu_local_info['gpu_s1bs_ub_bias'], d_gpu_local_info['gpu_s1bs_lb_smearing'], d_gpu_local_info['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_gpu_local_info['gpu_s2bs_s2s'], d_gpu_local_info['gpu_s2bs_lb_bias'], d_gpu_local_info['gpu_s2bs_ub_bias'], d_gpu_local_info['gpu_s2bs_lb_smearing'], d_gpu_local_info['gpu_s2bs_ub_smearing'], drv.In(num_pts_s1pf), d_gpu_local_info['gpu_s1pf_s1s'], d_gpu_local_info['gpu_s1pf_lb_acc'], d_gpu_local_info['gpu_s1pf_mean_acc'], d_gpu_local_info['gpu_s1pf_ub_acc'], drv.In(cut_acceptance_s1_intercept), drv.In(cut_acceptance_s1_slope), drv.In(cut_acceptance_s2_intercept), drv.In(cut_acceptance_s2_slope), drv.In(ms_par_0), drv.In(ms_par_1), drv.In(num_bins_r2), d_gpu_local_info['gpu_bin_edges_r2'], drv.In(num_bins_z), d_gpu_local_info['gpu_bin_edges_z'], d_gpu_local_info['gpu_s1_correction_map'], drv.In(num_bins_x), d_gpu_local_info['gpu_bin_edges_x'], drv.In(num_bins_y), d_gpu_local_info['gpu_bin_edges_y'], d_gpu_local_info['gpu_s2_correction_map'], drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d), drv.In(num_loops))

        d_gpu_local_info['function_to_call'](*tArgs, **self.d_gpu_scale)
        
        
        #print 'MC time: %f' % (time.time() - start_time_mc)
        #start_time_tot_ll = time.time()

        a_s1_s2_mc = np.reshape(a_hist_2d, (len(self.a_log_bin_edges)-1, len(self.a_s1_bin_edges)-1)).T

        sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
        if sum_mc == 0:
            return -np.inf


        #scale_par *= float(self.num_mc_events*self.num_loops) / sum_mc
        #a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information['num_data_pts']/float(self.num_mc_events*self.num_loops))
        
        a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information['num_data_pts']/float(self.num_mc_events*self.num_loops))

        # likelihood for band with ms scale
        if draw_fit:

            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

            s1_s2_data_plot = np.rot90(self.d_coincidence_data_information['a_log_s2_s1'])
            s1_s2_data_plot = np.flipud(s1_s2_data_plot)
            ax1.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')


            
            s1_s2_mc_plot = np.rot90(a_s1_s2_mc)
            s1_s2_mc_plot = np.flipud(s1_s2_mc_plot)
            
            
            #print 'changed mc to division temporarily'
            #for (x, y), val in np.ndenumerate(s1_s2_mc_plot):
            #    if val != 0:
            #        s1_s2_data_plot[x, y] -= val
            
            
            ax1.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_data_plot)
            ax2.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_mc_plot)
            ax2.set_xlabel(r'$S1 [PE]$')
            ax2.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')
            #plt.colorbar()


            f_flat, (ax_s1, ax_log) = plt.subplots(2)
            
            flat_s1_data = np.sum(s1_s2_data_plot, axis=0)
            flat_s1_mc = np.sum(s1_s2_mc_plot, axis=0)
            ax_s1.errorbar((self.a_s1_bin_edges[1:]+self.a_s1_bin_edges[:-1])/2., flat_s1_data, yerr=flat_s1_data**0.5, fmt='bo', label='Data')
            ax_s1.errorbar((self.a_s1_bin_edges[1:]+self.a_s1_bin_edges[:-1])/2., flat_s1_mc, yerr=flat_s1_mc**0.5, fmt='ro', label='MC')
            ax_s1.set_xlabel('S1 [PE]')
            ax_s1.set_ylabel('Counts')
            ax_s1.legend(loc='best')
            
            flat_log_data = np.sum(s1_s2_data_plot, axis=1)
            flat_log_mc = np.sum(s1_s2_mc_plot, axis=1)
            ax_log.errorbar((self.a_log_bin_edges[1:]+self.a_log_bin_edges[:-1])/2., flat_log_data, yerr=flat_log_data**0.5, fmt='bo')
            ax_log.errorbar((self.a_log_bin_edges[1:]+self.a_log_bin_edges[:-1])/2., flat_log_mc, yerr=flat_log_mc**0.5, fmt='ro')
            ax_log.set_xlabel(r'$Log_{10}(\frac{S2}{S1})$')
            ax_log.set_ylabel('Counts')

            f.savefig('./temp_results/2d_hist_%.3f_kV_%d_deg.png' % (self.cathode_setting, self.degree_setting))
            f_flat.savefig('./temp_results/1d_hists_%.3f_kV_%d_deg.png' % (self.cathode_setting, self.degree_setting))

        flat_s1_log_data = np.asarray(self.d_coincidence_data_information['a_log_s2_s1'].flatten(), dtype=np.float32)
        flat_s1_log_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
        

        logLikelihoodMatching = c_log_likelihood(flat_s1_log_data, flat_s1_log_mc, len(flat_s1_log_data), 1e-6)
        
        matching_ln_likelihood += logLikelihoodMatching

        #print logLikelihoodMatching, prior_ln_likelihood
        #print 'likelihood calculation time: %f' % (time.time() - start_time_tot_ll)
        #print total_ln_likelihood

        #print 'full function time: %f' % (time.time() - start_time_full)
        total_ln_likelihood = matching_ln_likelihood + prior_ln_likelihood

        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor



        if np.isnan(total_ln_likelihood):
            return -np.inf
        else:
            return total_ln_likelihood


        
    def wrapper_ln_likelihood_full_matching_band_with_ms_scale(self, a_parameters, kwargs={}):
        
        return self.ln_likelihood_full_matching_band_with_ms_scale(*a_parameters, **kwargs)




    def initial_positions_for_ensemble(self, a_free_parameters, num_walkers):
    
        if self.fit_type == 'sb':
            #(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par, scale_par, d_gpu_local_info, draw_fit=False)
        
            l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'cut_acceptance_par'] + ['prob_bkg', 'scale_par']
            count_free_pars = 0
        
        elif self.fit_type == 'sbf':
            #(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par, scale_par, d_gpu_local_info, draw_fit=False)
        
            l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par_0', 'acceptance_par_1', 'cut_acceptance_par'] + ['prob_bkg', 'scale_par']
            count_free_pars = 0
            
        
        elif self.fit_type == 'sb_ms':
            #(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par, scale_par, d_gpu_local_info, draw_fit=False)
        
            l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'cut_acceptance_par', 'ms_par_0', 'ms_par_1'] + ['prob_bkg', 'scale_par']
            count_free_pars = 0

        d_variable_arrays = {}
        d_stdevs = {}
        

        # position array should be (num_walkers, num_dim)

        for par_name in l_par_names:
            # handle photon and charge yield initial positions
            if par_name == 'a_py':
                d_variable_arrays[par_name] = np.asarray([np.random.normal(a_free_parameters[i], 0.3*np.asarray(a_free_parameters[i]), size=num_walkers) for i in xrange(num_yields)])

            elif par_name == 'a_qy':
                d_variable_arrays[par_name] = np.asarray([np.random.normal(a_free_parameters[i], 0.3*np.asarray(a_free_parameters[i]), size=num_walkers) for i in xrange(num_yields, 2*num_yields)])
        
            elif par_name == 'w_value':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['w_value'], self.nest_lindhard_model['spreads']['w_value'], size=num_walkers)

            elif par_name == 'alpha':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['alpha'], self.nest_lindhard_model['spreads']['alpha'], size=num_walkers)

            elif par_name == 'zeta':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['zeta'], self.nest_lindhard_model['spreads']['zeta'], size=num_walkers)

            elif par_name == 'beta':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['beta'], self.nest_lindhard_model['spreads']['beta'], size=num_walkers)

            elif par_name == 'gamma':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[count_free_pars], a_free_parameters[count_free_pars]*0.3, size=num_walkers)
                count_free_pars += 1

            elif par_name == 'delta':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['delta'], self.nest_lindhard_model['spreads']['delta'], size=num_walkers)

            elif par_name == 'kappa':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[count_free_pars], a_free_parameters[count_free_pars]*0.3, size=num_walkers)
                count_free_pars += 1

            elif par_name == 'eta':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['eta'], self.nest_lindhard_model['spreads']['eta'], size=num_walkers)

            elif par_name == 'lambda':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['lambda'], self.nest_lindhard_model['spreads']['lambda'], size=num_walkers)

            elif par_name == 'g1_value':
                d_variable_arrays['g1_value'] = np.random.multivariate_normal([self.g1_value], [[self.g1_uncertainty**2 * 0.2]], size=num_walkers).T

            elif par_name == 'extraction_efficiency_value':
                d_variable_arrays['extraction_efficiency_value'] = np.random.multivariate_normal([self.extraction_efficiency_value], [[self.extraction_efficiency_uncertainty**2 * 0.2]], size=num_walkers).T
            
            elif par_name == 'spe_res_value':
                d_variable_arrays['spe_res_value'] = np.random.multivariate_normal([self.spe_res_value], [[self.spe_res_uncertainty**2 * 0.2]], size=num_walkers).T

            elif par_name == 'gas_gain_mean_value':
                d_variable_arrays['gas_gain_mean_value'] = np.random.multivariate_normal([self.gas_gain_value], [[self.gas_gain_uncertainty**2 * 0.2]], size=num_walkers).T
            
            
            elif par_name == 'gas_gain_width_value':
                d_variable_arrays['gas_gain_width_value'] = np.random.multivariate_normal([self.gas_gain_width], [[self.gas_gain_width_uncertainty**2 * 0.2]], size=num_walkers).T

            elif par_name == 'dpe_prob':
                # values taken from DPE paper
                d_variable_arrays['dpe_prob'] = np.random.uniform(0.17, 0.24, size=num_walkers).T
                
            
            elif par_name[:8] == 'prob_bkg':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[count_free_pars], .02, size=num_walkers)
                count_free_pars += 1
            
            elif par_name == 'prob_ac_bkg':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[count_free_pars], .02, size=num_walkers)
                count_free_pars += 1

                

            elif par_name[:5] == 'scale':
                # need to track scale parameters
                # always the last indices so move back by
                # number of degree settings multiplied by
                # number of cathode settings
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[count_free_pars], a_free_parameters[count_free_pars]*0.3, size=num_walkers)
                count_free_pars += 1
                


            elif par_name[:7] == 'exciton':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[count_free_pars], .1, size=num_walkers)
                count_free_pars += 1
                
                count_etoi_pars += 1


            elif (par_name[-8:] == 'bias_par') or (par_name[-8:] == 'ring_par') or (par_name[:6] == 'accept') or (par_name == 'cut_acceptance_par') or (par_name[:6] == 'ms_par'):
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[count_free_pars], abs(a_free_parameters[count_free_pars])*0.3, size=num_walkers)
                count_free_pars += 1
            


            # catch all normally distributed RVs
            else:
                if par_name == 'g2_value' or par_name == 'pf_eff_par1' or par_name == 's1_eff_par1' or par_name == 's2_eff_par1'  or par_name == 'pf_stdev_par1' or par_name == 'pf_stdev_par2':
                    continue
                d_variable_arrays[par_name] = np.random.randn(num_walkers)


        l_parameters = []

        for par_name in l_par_names:
            if d_variable_arrays[par_name].shape[0] != num_walkers:
                for array in d_variable_arrays[par_name]:
                    l_parameters.append(array)
            else:
                l_parameters.append(d_variable_arrays[par_name])

        l_parameters = np.asarray(l_parameters).T

        return l_parameters




    def run_mcmc(self, num_steps, num_walkers, num_threads=1, thin=1):

        # get string for cathode voltages in use
        s_cathode_voltages = '%.3f' % (self.cathode_setting)

        # get string for degree settings in use
        s_degree_settings = '%d' % (self.degree_setting)

        # before emcee, setup save locations
        dir_specifier_name = '%s_kV_%s_deg' % (s_cathode_voltages, s_degree_settings)
        self.results_directory_name = config_xe1t.results_directory_name
        self.path_for_save = '%s/%s/%s/' % (self.results_directory_name, self.directory_name, dir_specifier_name)


        if not os.path.isdir(self.path_for_save):
            os.makedirs(self.path_for_save)


        # chain dictionary will have the following format
        # d_sampler[walkers] = [sampler_000, sampler_001, ...]

        dict_filename = 'sampler_dictionary.p'
        acceptance_filename = 'acceptance_fraction.p'
        autocorrelation_filename = 'autocorrelation.p'


        loaded_prev_sampler = False
        try:
            # two things will fail potentially
            # 1. open if file doesn't exist
            # 2. posWalkers load since initialized to None

            with open(self.path_for_save + dict_filename, 'r') as f_prev_sampler:

                d_sampler = pickle.load(f_prev_sampler)
                prevSampler = d_sampler[num_walkers][-1]


                # need to load in weird way bc can't pickle
                # ensembler object
                starting_pos = prevSampler['_chain'][:,-1,:]
                random_state = prevSampler['_random']
            loaded_prev_sampler = True
            print '\nSuccessfully loaded previous chain!\n'
        except:
            print '\nCould not load previous sampler or none existed - starting new sampler.\n'

        if not loaded_prev_sampler:

            if self.fit_type == 'sb':
                # 1287.7
                # free
                a_free_parameter_guesses = [0.01310503, 0.13468335, 0.09051169, 0.21755514, 0.2818793, 0.6676077, 2.57520554, 1.06633811, 0.06336397, 11.43789846]
                # 282.8
                # prior on gamma and kappa
                #a_free_parameter_guesses = [0.01272,  0.1368, 0.0064, 0.0631, 0.7894 , 0.2446, -3., 0.937, 0.0319, 0.99]
                
            elif self.fit_type == 'sbf':
                a_free_parameter_guesses = [0.01272951, 0.1327721, 0.01858148, 0.07537494, 0.49438128, 0.86396962, 4.54143241, 2.41954711, 0.80112399, 0.05583873, 4.17571822]
                
            elif self.fit_type == 'sb_ms':
                a_free_parameter_guesses = [0.01295242, 0.13219845, 0.02412865, 0.10884726, 0.37039122, 0.38418219, 0.37310507, -0.72129448, 1.44983571, 3.65409564, 0.06681038, 12.07172419]
                
            else:
                print '\nPlease run differential evolution minimizer for this setup and implement results in source code.\n'
                sys.exit()


            starting_pos = self.initial_positions_for_ensemble(a_free_parameter_guesses, num_walkers=num_walkers)
            #print starting_pos.shape

            random_state = None

            # create file if it doesn't exist
            if not os.path.exists(self.path_for_save + dict_filename):
                with open(self.path_for_save + dict_filename, 'w') as f_prev_sampler:
                    d_sampler = {}

                    d_sampler[num_walkers] = []

                    pickle.dump(d_sampler, f_prev_sampler)
            else:
                with open(self.path_for_save + dict_filename, 'r') as f_prev_sampler:
                    d_sampler = pickle.load(f_prev_sampler)
                with open(self.path_for_save + dict_filename, 'w') as f_prev_sampler:

                    d_sampler[num_walkers] = []

                    pickle.dump(d_sampler, f_prev_sampler)



        #sampler = emcee.EnsembleSampler(num_walkers, self.num_dimensions, self.wrapper_ln_likelihood_coincidence_matching, a=proposal_scale, threads=num_threads, pool=self.gpu_pool, kwargs={})
        num_dim = self.num_dimensions
        sampler = emcee.DESampler(num_walkers, num_dim, self.ln_likelihood_function_wrapper, threads=num_threads, autoscale_gamma=True, pool=self.gpu_pool, kwargs={})



        print '\n\nBeginning MCMC sampler\n\n'

        print '\nNumber of walkers * number of steps = %d * %d = %d function calls\n' % (num_walkers, num_steps, num_walkers*num_steps)

        start_time_mcmc = time.time()
        
        with click.progressbar(sampler.sample(p0=starting_pos, iterations=num_steps, rstate0=random_state, thin=thin), length=num_steps) as mcmc_sampler:
            for i, l_iterator_values in enumerate(mcmc_sampler):
                pass
                # should NOT refresh suppresion - screws up acceptance
                # since scale is constantly changing
                """
                if (i != 0 and (i % 25) == 0) or (i == 3):
                    index_max_flattened = np.argmax(sampler.lnprobability[:, :i].flatten())
                    flat_chain = sampler.chain[:, :i, :].reshape(-1, num_dim)
                    self.suppress_likelihood(iterations=200, a_free_par_guesses=flat_chain[index_max_flattened, :])
                """

        total_time_mcmc = (time.time() - start_time_mcmc) / 3600.

        print '\n\n%d function calls took %.2f hours.\n\n' % (num_walkers*num_steps, total_time_mcmc)

        #samples = sampler.chain[:, 10:, :].reshape((-1, num_dim))
        #print samples

        #fig = corner.corner(samples)
        #fig.savefig(self.path_for_save + 'corner_plot.png')


        # ------------------------------------------------
        # Prepare and save data and plots
        # ------------------------------------------------

        #print sampler.__dict__
        dictionary_for_sampler = sampler.__dict__
        if 'lnprobfn' in dictionary_for_sampler:
            del dictionary_for_sampler['lnprobfn']
        if 'pool' in dictionary_for_sampler:
            del dictionary_for_sampler['pool']

        with open(self.path_for_save + dict_filename, 'r') as f_prev_sampler:
            d_sampler = pickle.load(f_prev_sampler)
        #f_prev_sampler.close()

        f_prev_sampler = open(self.path_for_save + dict_filename, 'w')

        d_sampler[num_walkers].append(sampler.__dict__)

        pickle.dump(d_sampler, f_prev_sampler)
        f_prev_sampler.close()



        #sampler.run_mcmc(posWalkers, numSteps) # shortcut of above method
        pickle.dump(sampler.acceptance_fraction, open(self.path_for_save + acceptance_filename, 'w'))
        try:
            pickle.dump(sampler.acor, open(self.path_for_save + autocorrelation_filename, 'w'))
        except:
            print '\n\nWARNING: Not enough sample points to estimate the autocorrelation time - this likely means that the fit is bad since the burn-in time was not reached.\n\n'



    def differential_evolution_minimizer_free_pars(self, a_bounds, maxiter=250, tol=0.05, popsize=15, polish=False):
        def neg_log_likelihood_diff_ev(a_guesses):
            if self.fit_type == 'sb':
                l_parameters = []
                l_parameters += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], a_guesses[0], test.nest_lindhard_model['values']['delta'], a_guesses[1], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
                l_parameters += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2]
                l_parameters += list(a_guesses[-8:])
                
                #print l_parameters
            
                return -self.gpu_pool.map(self.wrapper_ln_likelihood_full_matching_band, [l_parameters])[0]
    
            elif self.fit_type == 'sbf':
                l_parameters = []
                l_parameters += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], a_guesses[0], test.nest_lindhard_model['values']['delta'], a_guesses[1], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
                l_parameters += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2]
                l_parameters += list(a_guesses[-9:])
                
                #print l_parameters
            
                return -self.gpu_pool.map(self.wrapper_ln_likelihood_full_matching_band_with_free_pax_eff, [l_parameters])[0]
    
    
            elif self.fit_type == 'sb_ms':
                l_parameters = []
                l_parameters += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], a_guesses[0], test.nest_lindhard_model['values']['delta'], a_guesses[1], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
                l_parameters += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2]
                l_parameters += list(a_guesses[-10:])
                
                #print l_parameters
            
                return -self.gpu_pool.map(self.wrapper_ln_likelihood_full_matching_band_with_ms_scale, [l_parameters])[0]
        
        
        print '\n\nStarting differential evolution minimizer...\n\n'
        result = optimize.differential_evolution(neg_log_likelihood_diff_ev, a_bounds, disp=True, maxiter=maxiter, tol=tol, popsize=popsize, polish=polish)
        print result



    def suppress_likelihood(self, iterations=200, a_free_par_guesses=None):
    
        # reset variables in case this is not the first time run
        self.b_suppress_likelihood = False
        self.ll_suppression_factor = 1.

        if self.fit_type == 'sb' and a_free_par_guesses == None:
            #a_guesses = [0.01181812,  0.13457987, 0.00640899, 0.06315222, 0.7894911 , 0.24468025, -1.67785755, 0.8232389, 0.03198071, 0.95490475]
            # prior on gamma and kappa
            a_guesses = [0.01310503, 0.13468335, 0.09051169, 0.21755514, 0.2818793, 0.6676077, 2.57520554, 1.06633811, 0.06336397, 11.43789846] # from diff ev
            #a_guesses = [0.01272,  0.1368, 0.0064, 0.0631, 0.7894 , 0.2446, -3., 0.937, 0.0319, 0.99] # from corner plot
            
            
            a_free_par_guesses = []
            a_free_par_guesses += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], a_guesses[0], test.nest_lindhard_model['values']['delta'], a_guesses[1], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
            a_free_par_guesses += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2]
            a_free_par_guesses += list(a_guesses[-8:])
        

        elif self.fit_type == 'sbf' and a_free_par_guesses == None:
        
            a_guesses = [0.01272951, 0.1327721, 0.01858148, 0.07537494, 0.49438128, 0.86396962, 4.54143241, 2.41954711, 0.80112399, 0.05583873, 4.17571822]
        
        
            a_free_par_guesses = []
            a_free_par_guesses += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], a_guesses[0], test.nest_lindhard_model['values']['delta'], a_guesses[1], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
            a_free_par_guesses += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2]
            a_free_par_guesses += list(a_guesses[-9:])


        elif self.fit_type == 'sb_ms' and a_free_par_guesses == None:
        
            a_guesses = [0.01295242, 0.13219845, 0.02412865, 0.10884726, 0.37039122, 0.38418219, 0.37310507, -0.72129448, 1.44983571, 3.65409564, 0.06681038, 12.07172419]
        
        
            a_free_par_guesses = []
            a_free_par_guesses += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], a_guesses[0], test.nest_lindhard_model['values']['delta'], a_guesses[1], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
            a_free_par_guesses += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2]
            a_free_par_guesses += list(a_guesses[-10:])



        
        #print a_free_par_guesses
        l_parameters = [a_free_par_guesses for i in xrange(iterations)]
        l_log_likelihoods = self.gpu_pool.map(self.ln_likelihood_function_wrapper, l_parameters)
        #print l_log_likelihoods

        std_ll = np.std(l_log_likelihoods)

        print 'Mean for %.3e MC iterations is %f' % (self.num_mc_events, np.mean(l_log_likelihoods))
        print 'Standard deviation for %.3e MC iterations is %f' % (self.num_mc_events, std_ll)
        print 'Will scale LL such that stdev is 0.25'
        
        #plt.hist(l_log_likelihoods)
        #plt.show()

        if std_ll < 0.25:
            self.b_suppress_likelihood = True
            self.ll_suppression_factor = 1.
            print 'Standard deviation already small so not supressing\n\n'
        else:
            self.b_suppress_likelihood = True
            self.ll_suppression_factor = std_ll / 0.25
            print 'LL suppression factor: %f\n' % self.ll_suppression_factor







if __name__ == '__main__':
    copy_reg.pickle(types.MethodType, reduce_method)


    d_coincidence_data = {}
    d_coincidence_data['degree_setting'] = -4
    d_coincidence_data['cathode_setting'] = 12
    
    """
    test = fit_nr(d_coincidence_data, 'sb', num_mc_events=2e4, l_gpus=[0], num_loops=40)

    # ln_likelihood_full_matching_band(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par, cut_acceptance_par, prob_bkg, scale_par, d_gpu_local_info, draw_fit=False)
    
    l_parameters = []
    # [0.01284925,  0.13544257, 0.04099268, 0.14286395, 0.5015843, 0.85780362, 2.05229989, 0.50929897, 0.07927303, 8.75781103] # 611
    l_parameters += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], test.nest_lindhard_model['values']['gamma'], test.nest_lindhard_model['values']['delta'], test.nest_lindhard_model['values']['kappa'], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
    l_parameters += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2, 0.04099268, 0.14286395, 0.5015843, 0.85780362, 2.05229989, 0.50929897, 0.07927303, 8.75781103]
    test.gpu_pool.map(test.wrapper_ln_likelihood_full_matching_band, [l_parameters])
    
    
    #a_free_par_bounds = [(0.001, 0.04), (0.1, 0.2), (0, 1.), (0, 1.), (0, 1.), (0, 1.), (0, 6), (-2, 2), (0., 0.5), (1.0, 16.)]
    #test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=150, popsize=15, tol=0.01)
    
    #test.suppress_likelihood()
    #test.run_mcmc(num_steps=160, num_walkers=256)
    
    """
    
    """
    
    test = fit_nr(d_coincidence_data, 'sbf', num_mc_events=2e6, l_gpus=[0, 5], num_loops=4)

    # ln_likelihood_full_matching_band(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par, cut_acceptance_par, prob_bkg, scale_par, d_gpu_local_info, draw_fit=False)
    
    l_parameters = []
    # [0.01272951, 0.1327721, 0.01858148, 0.07537494, 0.49438128, 0.86396962, 4.54143241, 2.41954711, 0.80112399, 0.05583873, 4.17571822]
    l_parameters += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], test.nest_lindhard_model['values']['gamma'], test.nest_lindhard_model['values']['delta'], test.nest_lindhard_model['values']['kappa'], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
    l_parameters += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2, 0.01858148, 0.07537494, 0.49438128, 0.86396962, 6, 2.41954711, 0.80112399, 0.05583873, 4.17571822]
    #test.gpu_pool.map(test.wrapper_ln_likelihood_full_matching_band_with_free_pax_eff, [l_parameters])
    
    
    #a_free_par_bounds = [(0.001, 0.04), (0.1, 0.2), (0, 1.), (0, 1.), (0, 1.), (0, 1.), (0.05, 5), (0.5, 8), (-2, 2), (0., 0.5), (1.0, 16.)]
    #test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=150, popsize=15, tol=0.01)
    
    #test.suppress_likelihood()
    test.run_mcmc(num_steps=80, num_walkers=256)
    
    """
    
    
    test = fit_nr(d_coincidence_data, 'sb_ms', num_mc_events=2e6, l_gpus=[0], num_loops=4)

    # ln_likelihood_full_matching_band(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, dpe_prob, s1_bias_par, s1_smearing_par, s2_bias_par, s2_smearing_par, acceptance_par, cut_acceptance_par, prob_bkg, scale_par, d_gpu_local_info, draw_fit=False)
    
    l_parameters = []
    # [0.01295242, 0.13219845, 0.02412865, 0.10884726, 0.37039122, 0.38418219, 0.37310507, -0.72129448, 1.44983571, 3.65409564, 0.06681038, 12.07172419] # 610.2
    l_parameters += [test.nest_lindhard_model['values']['w_value'], test.nest_lindhard_model['values']['alpha'], test.nest_lindhard_model['values']['zeta'], test.nest_lindhard_model['values']['beta'], test.nest_lindhard_model['values']['gamma'], test.nest_lindhard_model['values']['delta'], test.nest_lindhard_model['values']['kappa'], test.nest_lindhard_model['values']['eta'], test.nest_lindhard_model['values']['lambda']]
    l_parameters += [test.g1_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, 0.2, 0.02412865, 0.10884726, 0.37039122, 0.38418219, 0.37310507, -0.72129448, 1.44983571, 3.65409564, 0.06681038, 12.07172419]
    #test.gpu_pool.map(test.wrapper_ln_likelihood_full_matching_band_with_ms_scale, [l_parameters])
    
    
    #a_free_par_bounds = [(0.001, 0.04), (0.1, 0.2), (0, 1.), (0, 1.), (0, 1.), (0, 1.), (0, 6), (-2, 2), (0.3, 2.1), (1.3, 7), (0., 0.5), (1.0, 16.)]
    #test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=150, popsize=15, tol=0.01)
    
    #test.suppress_likelihood()
    test.run_mcmc(num_steps=160, num_walkers=256)
    
    
    
    
    test.close_workers()



