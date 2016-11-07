
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

import numpy as np
from math import exp, factorial, erf, ceil, log, pow, floor, lgamma

from scipy import optimize, misc, stats
from scipy.stats import norm, multivariate_normal, binom

import copy_reg, types, pickle, click, time
from subprocess import call
from pprint import pprint

from produce_nest_yields import nest_nr_mean_yields

import cuda_example_observables_production
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray
from Queue import Queue
import threading

import example_config
import cPickle as pickle



def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))


d_cathode_voltage_to_field = {0.345:210,
                              1.054:490,
                              2.356:1000}



class gpu_pool:
    def __init__(self, num_gpus, grid_dim, block_dim, num_dim_gpu_call, d_gpu_single_copy_arrays, function_name):
        self.num_gpus = num_gpus
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.num_dim_gpu_call = num_dim_gpu_call
        self.d_gpu_single_copy_arrays = d_gpu_single_copy_arrays
        self.function_name = function_name

        self.alive = True
        self.q_gpu = Queue()
        for i in xrange(self.num_gpus):
            self.q_gpu.put(i)

        self.q_in = Queue()
        self.q_out = Queue()
        self.l_dispatcher_threads = []
        self.dispatcher_dead_time = 0.5

        self.q_dead = Queue()


        for i in xrange(self.num_gpus):
            if self.q_gpu.empty():
                break
            print 'Starting worker!\n'
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
        local_gpu_setup_kernel = pycuda.compiler.SourceModule(cuda_example_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('setup_kernel')

        local_rng_states = drv.mem_alloc(np.int32(self.block_dim*self.grid_dim)*pycuda.characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
        local_gpu_setup_kernel(np.int32(int(self.block_dim*self.grid_dim)), local_rng_states, np.uint64(0), np.uint64(0), grid=(int(self.grid_dim), 1), block=(int(self.block_dim), 1, 1))

        print '\nSetup Kernel run!\n'
        sys.stdout.flush()

        gpu_observables_func = SourceModule(cuda_example_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function(self.function_name)

        print 'Putting energy array on GPU...'
        gpu_energies = self.d_gpu_single_copy_arrays['energy']
        gpu_energies = pycuda.gpuarray.to_gpu(gpu_energies)
        


        print 'Putting bin edges on GPU...'
        gpu_bin_edges_s1 = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_s1'])


        print 'Putting bin edges on GPU...'
        gpu_bin_edges_s2 = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_s2'])


        d_gpu_local_info = {'function_to_call':gpu_observables_func,
                            'rng_states':local_rng_states,
                            'gpu_energies':gpu_energies,
                            'gpu_bin_edges_s1':gpu_bin_edges_s1,
                            'gpu_bin_edges_s2':gpu_bin_edges_s2}

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
                    #print len(args)
                    if True:# or len(args) == self.num_dim_gpu_call or len(args) == 21:
                        args = np.append(args, [d_gpu_local_info])
                    #print len(args)
                    #print args
                    #print task
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

C.register_file('../../python_modules/mc_code/c_log_likelihood.C', ['smart_log_likelihood'])
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


# create dot product in c++ for speed
# and thread safety

C.register_file('../../python_modules/mc_code/c_safe_dot.C', ['safe_dot'])



def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))



class fit_nr(object):
    def __init__(self, dataset_name, num_mc_events=1e5, num_gpus=1):
        
        self.dataset_name = dataset_name

        copy_reg.pickle(types.MethodType, reduce_method)


        # set number of mc events
        self.d_gpu_scale = {}
        block_dim = 1024
        self.d_gpu_scale['block'] = (block_dim,1,1)
        numBlocks = floor(num_mc_events / float(block_dim))
        self.d_gpu_scale['grid'] = (int(numBlocks), 1)
        self.num_mc_events = int(numBlocks*block_dim)


        # ------------------------------------------------
        # ------------------------------------------------
        # Set paths and allowed values
        # ------------------------------------------------
        # ------------------------------------------------



        self.path_to_data = './example_data/%s/' % (self.dataset_name)
        self.d_data_parameters = example_config.d_data_parameters


        self.l_energy_settings = [300, 0, 30]



        # ------------------------------------------------
        # ------------------------------------------------
        # Make 2d histogram for data
        # ------------------------------------------------
        # ------------------------------------------------

        self.s1_settings = [40, 0, 35]
        self.s2_settings = [40, 0, 2500]

        # create bin edges
        # +1 on number of bins needed for edges only
        self.a_s1_bin_edges = np.linspace(self.s1_settings[1], self.s1_settings[2], self.s1_settings[0]+1)
        self.a_s2_bin_edges = np.linspace(self.s2_settings[1], self.s2_settings[2], self.s2_settings[0]+1)
        
        # grab data from pickle files
        self.a_data_s1 = pickle.load(open('%sa_s1_%s.pkl' % (self.path_to_data, self.dataset_name)))
        self.a_data_s2 = pickle.load(open('%sa_s2_%s.pkl' % (self.path_to_data, self.dataset_name)))
        
        
        # fill histogram
        self.a_data_hist_s1_s2, self.a_s1_bin_edges, self.a_s2_bin_edges = np.histogram2d(self.a_data_s1, self.a_data_s2, bins=[self.a_s1_bin_edges, self.a_s2_bin_edges])


        # ------------------------------------------------
        # ------------------------------------------------
        # Load MC files or create reduced file if not present
        # and load into easy_graph
        # ------------------------------------------------
        # ------------------------------------------------

        l_energy_arrays = [0 for i in xrange(len(self.d_data_parameters['l_energies']))]
        for i in xrange(len(l_energy_arrays)):
            l_energy_arrays[i] = np.random.normal(loc=self.d_data_parameters['l_energies'][i], scale=self.d_data_parameters['l_energies'][i]*self.d_data_parameters['energy_width'], size=int(self.num_mc_events/len(l_energy_arrays)))

        # concatenate arrays and make sure they are
        # float32 for use on GPU
        self.a_energies = np.concatenate(l_energy_arrays)
        self.a_energies = np.asarray(self.a_energies, dtype=np.float32)



        # ------------------------------------------------
        # ------------------------------------------------
        # Set default parameter values for use in MC
        # ------------------------------------------------
        # ------------------------------------------------


        # other parameters have uncertainties and covariance
        # matrices in d_data_parameters

        self.w_value = 13.7
        self.w_uncertainty = 0.4

        # below are the NEST values for the Lindhard model
        self.nest_lindhard_model = {}
        self.nest_lindhard_model['values'] = {'alpha':1.240,
                                                'zeta':0.0472,
                                                'beta':239,
                                                'gamma':0.01385,
                                                'delta':0.0620,
                                                'kappa':0.1394,
                                                'eta':3.3,
                                                'lambda':1.14}

        # below are the spreads for walker initialization only
        # these are NOT uncertainties
        self.nest_lindhard_model['spreads'] = {'alpha':0.008,
                                                'zeta':0.008,
                                                'beta':12,
                                                'gamma':0.00065,
                                                'delta':0.006,
                                                'kappa':0.003,
                                                'eta':2,
                                                'lambda':0.25}
        
        # neg side still has positive value
        # since are looking stdev
        # will use for parameters that are NOT free
        self.nest_lindhard_model['sigmas'] = {'alpha':[0.0073, 0.079],
                                                'zeta':[0.0073, 0.0088],
                                                'beta':[8.8, 28],
                                                'gamma':[0.00073, 0.00058],
                                                'delta':[0.0064, 0.0056],
                                                'kappa':[0.0026, 0.0032],
                                                'eta':[0.7, 5.3],
                                                'lambda':[0.09, 0.45]}


        self.ln_likelihood_function = self.ln_likelihood_full_matching
        self.ln_likelihood_function_wrapper = self.wrapper_ln_likelihood_full_matching
        self.num_dimensions = 19
        self.gpu_function_name = 'gpu_example_observables_production_with_hist_lindhard_model'


        d_gpu_single_copy_array_dictionaries = {'energy':self.a_energies,
                                                'bin_edges_s1':self.a_s1_bin_edges,
                                                'bin_edges_s2':self.a_s2_bin_edges
                                                }
        self.gpu_pool = gpu_pool(num_gpus=num_gpus, grid_dim=numBlocks, block_dim=block_dim, num_dim_gpu_call=self.num_dimensions, d_gpu_single_copy_arrays=d_gpu_single_copy_array_dictionaries, function_name=self.gpu_function_name)
        
        
        # before emcee, setup save locations
        self.dir_specifier_name = self.d_data_parameters['name']
        self.results_directory_name = './results'
        self.path_for_save = '%s/%s/' % (self.results_directory_name, self.dir_specifier_name)
        self.s_directory_save_plots_name = './plots/%s/' % (self.dir_specifier_name)
        

        self.b_suppress_likelihood = False


    def close_workers(self):
        self.gpu_pool.close()



    def get_ln_prior_g1(self, g1):
        if g1 < 0 or g1 > 1:
            return -np.inf,
        return norm.logpdf(g1, self.d_data_parameters['g1'][0], self.d_data_parameters['g1'][1])
        
    
    def get_ln_prior_extraction_efficiency(self, extraction_efficiency):
        if extraction_efficiency < 0 or extraction_efficiency > 1:
            return -np.inf
        return norm.logpdf(extraction_efficiency, self.d_data_parameters['extraction_efficiency'][0], self.d_data_parameters['extraction_efficiency'][1])


    def get_ln_prior_gas_gain_value(self, gas_gain_value):
        return norm.logpdf(gas_gain_value, self.d_data_parameters['gas_gain_value'][0], self.d_data_parameters['gas_gain_value'][1])
        
        
    def get_ln_prior_gas_gain_width(self, gas_gain_width):
        return norm.logpdf(gas_gain_width, self.d_data_parameters['gas_gain_width'][0], self.d_data_parameters['gas_gain_width'][1])
        
        
    def get_ln_prior_spe_res(self, spe_res):
        if spe_res < 0:
            return -np.inf
        return norm.logpdf(spe_res, self.d_data_parameters['spe_res'][0], self.d_data_parameters['spe_res'][1])
        
        
    def get_ln_prior_s1_acc_pars(self, s1_acc_par0, s1_acc_par1):
        if s1_acc_par0 < 0 or s1_acc_par1 < 0:
            return -np.inf
        return norm.logpdf(s1_acc_par0, self.d_data_parameters['s1_eff_par0'][0], self.d_data_parameters['s1_eff_par0'][1]) + norm.logpdf(s1_acc_par1, self.d_data_parameters['s1_eff_par1'][0], self.d_data_parameters['s1_eff_par1'][1])


    def get_ln_prior_s2_acc_pars(self, s2_acc_par0, s2_acc_par1):
        if s2_acc_par0 < 0 or s2_acc_par1 < 0:
            return -np.inf
        return multivariate_normal.logpdf([s2_acc_par0, s2_acc_par1], self.d_data_parameters['s2_eff_pars'], self.d_data_parameters['s2_eff_cov_matrix'])



    
    # get likelihood and w-value given random variable (nuissance parameter)
    def get_ln_prior_w_value(self, w_value):
        return norm.logpdf(w_value, self.w_value, self.w_uncertainty)



    def get_ln_prior_scale_par(self, scale_par):
        if scale_par < 0:
            return -np.inf
        else:
            return 0

    
    
    def get_ln_prior_par_greater_than_zero(self, par):
        if par < 0:
            return -np.inf
        else:
            return 0.



    def get_ln_prior_zeta(self, zeta):
        if zeta < 0:
            return -np.inf
        else:
            if zeta > self.nest_lindhard_model['values']['zeta']:
                return norm.logpdf(zeta, self.nest_lindhard_model['values']['zeta'], self.nest_lindhard_model['sigmas']['zeta'][1])
            else:
                return norm.logpdf(zeta, self.nest_lindhard_model['values']['zeta'], self.nest_lindhard_model['sigmas']['zeta'][0])



    def get_ln_prior_delta(self, delta):
        if delta < 0:
            return -np.inf
        else:
            if delta > self.nest_lindhard_model['values']['delta']:
                return norm.logpdf(delta, self.nest_lindhard_model['values']['delta'], self.nest_lindhard_model['sigmas']['delta'][1])
            else:
                return norm.logpdf(delta, self.nest_lindhard_model['values']['delta'], self.nest_lindhard_model['sigmas']['delta'][0])







    def ln_likelihood_full_matching(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency, gas_gain_value, gas_gain_width, spe_res, s1_acc_par0, s1_acc_par1, s2_acc_par0, s2_acc_par1, scale_par, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------


        prior_ln_likelihood = 0
        matching_ln_likelihood = 0

        # get w-value prior lieklihood
        prior_ln_likelihood += self.get_ln_prior_w_value(w_value)

        # priors of detector variables
        prior_ln_likelihood += self.get_ln_prior_g1(g1_value)
        prior_ln_likelihood += self.get_ln_prior_extraction_efficiency(extraction_efficiency)
        prior_ln_likelihood += self.get_ln_prior_gas_gain_value(gas_gain_value)
        prior_ln_likelihood += self.get_ln_prior_gas_gain_width(gas_gain_width)
        prior_ln_likelihood += self.get_ln_prior_spe_res(spe_res)
        
        prior_ln_likelihood += self.get_ln_prior_s1_acc_pars(s1_acc_par0, s1_acc_par1)
        prior_ln_likelihood += self.get_ln_prior_s2_acc_pars(s2_acc_par0, s2_acc_par1)



        # get priors from lindhard parameters
        prior_ln_likelihood += self.get_ln_prior_par_greater_than_zero(alpha)
        prior_ln_likelihood += self.get_ln_prior_par_greater_than_zero(beta)
        prior_ln_likelihood += self.get_ln_prior_par_greater_than_zero(gamma)
        prior_ln_likelihood += self.get_ln_prior_par_greater_than_zero(kappa)
        prior_ln_likelihood += self.get_ln_prior_par_greater_than_zero(eta)
        prior_ln_likelihood += self.get_ln_prior_par_greater_than_zero(lamb)

        prior_ln_likelihood += self.get_ln_prior_zeta(zeta)
        prior_ln_likelihood += self.get_ln_prior_delta(delta)
        

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
        mean_field = np.asarray(self.d_data_parameters['mean_field'], dtype=np.float32)

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
        spe_res = np.asarray(spe_res, dtype=np.float32)

        s1_acc_par0 = np.asarray(s1_acc_par0, dtype=np.float32)
        s1_acc_par1 = np.asarray(s1_acc_par1, dtype=np.float32)

        s2_acc_par0 = np.asarray(s2_acc_par0, dtype=np.float32)
        s2_acc_par1 = np.asarray(s2_acc_par1, dtype=np.float32)

        # for histogram binning
        num_bins_s1 = np.asarray(self.s1_settings[0], dtype=np.int32)
        num_bins_s2 = np.asarray(self.s2_settings[0], dtype=np.int32)

        a_hist_2d = np.zeros(self.s1_settings[0]*self.s2_settings[0], dtype=np.int32)
        
        #print d_gpu_local_info['d_gpu_energy'][degree_setting]
        
        l_gpu_args = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['gpu_energies'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(spe_res), drv.In(s1_acc_par0), drv.In(s1_acc_par1), drv.In(s2_acc_par0), drv.In(s2_acc_par1), drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_s2), d_gpu_local_info['gpu_bin_edges_s2'], drv.InOut(a_hist_2d))

        d_gpu_local_info['function_to_call'](*l_gpu_args, **self.d_gpu_scale)
        

        a_s1_s2_mc = np.reshape(a_hist_2d, (self.s2_settings[0], self.s1_settings[0])).T
        
        #print list(a_s1_s2_mc)
        
        sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
        if sum_mc == 0:
            #print 'sum mc == 0'
            return -np.inf

        # this forces our scale to be close to 1 (difference comes from acceptance)
        a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*len(self.a_data_s1)/float(self.num_mc_events))

        #'ml'
        if draw_fit:

            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
            
            ax1.set_xlabel('S1 [PE]')
            ax1.set_ylabel('log(S2/S1)')
            ax2.set_xlabel('S1 [PE]')
            ax2.set_ylabel('log(S2/S1)')

            s1_s2_data_plot = np.rot90(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][degree_setting]['a_log_s2_s1'])
            s1_s2_data_plot = np.flipud(s1_s2_data_plot)
            ax1.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_data_plot)

            s1_s2_mc_plot = np.rot90(a_s1_s2_mc)
            s1_s2_mc_plot = np.flipud(s1_s2_mc_plot)
            #print self.l_s1_settings
            #print self.l_log_settings
            #print self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].shape
            #print a_s1_s2_mc.shape
            #print s1_s2_data_plot.shape
            #print s1_s2_mc_plot.shape
            ax2.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_mc_plot)
            #plt.colorbar()


            c1 = Canvas(1400, 400)
            c1.Divide(2)

            h_s1_data = Hist(*self.l_s1_settings, name='hS1_draw_data')
            root_numpy.array2hist(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][degree_setting]['a_log_s2_s1'].sum(axis=1), h_s1_data)

            hS1MC = Hist(*self.l_s1_settings, name='hS1_draw_mc')
            root_numpy.array2hist(a_s1_s2_mc.sum(axis=1), hS1MC)

            s1_scale_factor = h_s1_data.Integral() / float(hS1MC.Integral())

            g_s1_data = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s1_data)
            g_s1_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(hS1MC, scale=s1_scale_factor)

            g_s1_mc.SetFillColor(root.kBlue)
            g_s1_mc.SetMarkerColor(root.kBlue)
            g_s1_mc.SetLineColor(root.kBlue)
            g_s1_mc.SetFillStyle(3005)

            g_s1_data.SetTitle('S1 Comparison')
            g_s1_data.GetXaxis().SetTitle('S1 [PE]')
            g_s1_data.GetYaxis().SetTitle('Counts')

            g_s1_data.SetLineColor(root.kRed)
            g_s1_data.SetMarkerSize(0)
            g_s1_data.GetXaxis().SetRangeUser(self.l_s1_settings[1], self.l_s1_settings[2])
            g_s1_data.GetYaxis().SetRangeUser(0, 1.2*max(h_s1_data.GetMaximum(), hS1MC.GetMaximum()))

            c1.cd(1)
            g_s1_data.Draw('ap')
            g_s1_mc.Draw('same')
            g_s1_mc_band = g_s1_mc.Clone()
            g_s1_mc_band.Draw('3 same')

            h_s2_data = Hist(*self.l_log_settings, name='h_s2_draw_data')
            root_numpy.array2hist(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][degree_setting]['a_log_s2_s1'].sum(axis=0), h_s2_data)

            h_s2_mc = Hist(*self.l_log_settings, name='h_s2_draw_mc')
            root_numpy.array2hist(a_s1_s2_mc.sum(axis=0), h_s2_mc)

            s2_scale_factor = h_s2_data.Integral() / float(h_s2_mc.Integral())

            g_s2_data = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s2_data)
            g_s2_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s2_mc, scale=s2_scale_factor)

            g_s2_mc.SetFillColor(root.kBlue)
            g_s2_mc.SetMarkerColor(root.kBlue)
            g_s2_mc.SetLineColor(root.kBlue)
            g_s2_mc.SetFillStyle(3005)

            g_s2_data.SetTitle('Log(S2/S1) Comparison')
            g_s2_data.GetXaxis().SetTitle('Log(S2/S1)')
            g_s2_data.GetYaxis().SetTitle('Counts')

            g_s2_data.SetLineColor(root.kRed)
            g_s2_data.SetMarkerSize(0)
            g_s2_data.GetXaxis().SetRangeUser(self.l_log_settings[1], self.l_log_settings[2])
            g_s2_data.GetYaxis().SetRangeUser(0, 1.2*max(h_s2_data.GetMaximum(), h_s2_mc.GetMaximum()))

            c1.cd(2)
            g_s2_data.Draw('ap')
            g_s2_mc.Draw('same')
            g_s2_mc_band = g_s2_mc.Clone()
            g_s2_mc_band.Draw('3 same')

            c1.Update()

            neriX_analysis.save_plot(['temp_results'], c1, '%d_deg_1d_hists' % (degree_setting), batch_mode=True)
            f.savefig('./temp_results/%d_deg_2d_hist.png' % (degree_setting))

        flat_s1_s2_data = np.asarray(self.a_data_hist_s1_s2.flatten(), dtype=np.float32)
        flat_s1_s2_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
        logLikelihoodMatching = c_log_likelihood(flat_s1_s2_data, flat_s1_s2_mc, len(flat_s1_s2_data), int(self.num_mc_events), 0.95)

        #print prior_ln_likelihood
        #print logLikelihoodMatching
        #print max(flat_s1_s2_data)
        #print max(flat_s1_s2_mc)
        #print '\n\n'


        if np.isnan(logLikelihoodMatching):
            return -np.inf
        else:
            matching_ln_likelihood += logLikelihoodMatching
    
        total_ln_likelihood = prior_ln_likelihood + matching_ln_likelihood
        #print total_ln_likelihood
        
        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor
                
        if np.isnan(total_ln_likelihood):
            return -np.inf
        else:
            return total_ln_likelihood
            
            
            
    def wrapper_ln_likelihood_full_matching(self, a_parameters, kwargs={}):
        return self.ln_likelihood_full_matching(*a_parameters, **kwargs)





    def initial_positions_for_ensemble(self, num_walkers):
    
        l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency', 'gas_gain_value', 'gas_gain_width', 'spe_res', 's1_acc_par0', 's1_acc_par1', 's2_acc_par0', 's2_acc_par1', 'scale_par']

        d_variable_arrays = {}
        d_stdevs = {}


        # position array should be (num_walkers, num_dim)

        for par_name in l_par_names:

            if par_name == 'w_value':
                # special case since w_value is an RV
                d_variable_arrays[par_name] = np.random.normal(self.w_value, 1, size=num_walkers)

            elif par_name == 'alpha':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['alpha'], self.nest_lindhard_model['spreads']['alpha'], size=num_walkers)

            elif par_name == 'zeta':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['zeta'], self.nest_lindhard_model['spreads']['zeta'], size=num_walkers)

            elif par_name == 'beta':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['beta'], self.nest_lindhard_model['spreads']['beta'], size=num_walkers)

            elif par_name == 'gamma':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['gamma'], self.nest_lindhard_model['spreads']['gamma'], size=num_walkers)

            elif par_name == 'delta':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['delta'], self.nest_lindhard_model['spreads']['delta'], size=num_walkers)

            elif par_name == 'kappa':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['kappa'], self.nest_lindhard_model['spreads']['kappa'], size=num_walkers)

            elif par_name == 'eta':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['eta'], self.nest_lindhard_model['spreads']['eta'], size=num_walkers)

            elif par_name == 'lambda':
                d_variable_arrays[par_name] = np.random.normal(self.nest_lindhard_model['values']['lambda'], self.nest_lindhard_model['spreads']['lambda'], size=num_walkers)

            elif par_name == 'extraction_efficiency':
                d_variable_arrays['extraction_efficiency'] = np.random.normal(*self.d_data_parameters['extraction_efficiency'], size=num_walkers)
                
            elif par_name == 'g1_value':
                d_variable_arrays['g1_value'] = np.random.normal(*self.d_data_parameters['g1'], size=num_walkers)

            elif par_name == 'gas_gain_value':
                d_variable_arrays['gas_gain_value'] = np.random.normal(*self.d_data_parameters['gas_gain_value'], size=num_walkers)

            elif par_name == 'gas_gain_width':
                d_variable_arrays['gas_gain_width'] = np.random.normal(*self.d_data_parameters['gas_gain_width'], size=num_walkers)

            elif par_name == 'spe_res':
                d_variable_arrays['spe_res'] = np.random.normal(*self.d_data_parameters['spe_res'], size=num_walkers)

            elif par_name == 's1_acc_par0':
                d_variable_arrays['s1_acc_par0'] = np.random.normal(*self.d_data_parameters['s1_eff_par0'], size=num_walkers)

            elif par_name == 's1_acc_par1':
                d_variable_arrays['s1_acc_par1'] = np.random.normal(*self.d_data_parameters['s1_eff_par1'], size=num_walkers)

            elif par_name == 's2_acc_par0':
                d_variable_arrays['s2_acc_par0'], d_variable_arrays['s2_acc_par1'] = np.random.multivariate_normal(self.d_data_parameters['s2_eff_pars'], self.d_data_parameters['s2_eff_cov_matrix'], size=num_walkers).T
                
            elif par_name == 's2_eff_par1':
                continue
                
            elif par_name == 'scale_par':
                d_variable_arrays['scale_par'] = np.random.normal(1.3, 0.3, size=num_walkers)
                


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





        if not os.path.isdir(self.path_for_save):
            os.makedirs(self.path_for_save)


        # chain dictionary will have the following format
        # d_sampler[walkers] = [sampler_000, sampler_001, ...]

        dict_filename = 'sampler_dictionary.p'


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

            starting_pos = self.initial_positions_for_ensemble(num_walkers=num_walkers)

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
                if (i != 0 and (i % 25) == 0) or (i == 3):
                    index_max_flattened = np.argmax(sampler.lnprobability[:, :i].flatten())
                    flat_chain = sampler.chain[:, :i, :].reshape(-1, num_dim)
                    self.suppress_likelihood(iterations=200, a_passed_free_par_guesses=flat_chain[index_max_flattened, :])

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
        


    def create_corner_plot(self, num_walkers, num_steps_to_include):
        
        l_labels_for_corner_plot = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency', 'gas_gain_value', 'gas_gain_width', 'spe_res', 's1_acc_par0', 's1_acc_par1', 's2_acc_par0', 's2_acc_par1', 'scale_par']
        num_dim = len(l_labels_for_corner_plot)
        
        sPathToFile = '%s/%s' % (self.path_for_save, 'sampler_dictionary.p')
        
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

        fig.savefig(self.s_directory_save_plots_name + self.d_data_parameters['name'] + '_corner.png')
        
        try:
            print emcee.autocorr.integrated_time(np.mean(a_sampler, axis=0), axis=0,
                                        low=10, high=None, step=1, c=2,
                                        fast=False)
        except:
            print 'Chain too short to find autocorrelation time!'





    def differential_evolution_minimizer_free_pars(self, a_bounds, maxiter=250, tol=0.05, popsize=15):
        def neg_log_likelihood_diff_ev(a_guesses):
            return -self.gpu_pool.map(self.wrapper_ln_likelihood_full_matching, [a_guesses])[0]
        
        print '\n\nStarting differential evolution minimizer...\n\n'
        result = optimize.differential_evolution(neg_log_likelihood_diff_ev, a_bounds, disp=True, maxiter=maxiter, tol=tol, popsize=popsize)
        print result



    def suppress_likelihood(self, a_passed_free_par_guesses=None, iterations=200):
    
        # reset variables in case this is not the first time run
        self.b_suppress_likelihood = False
        self.ll_suppression_factor = 1.

        if a_passed_free_par_guesses == None:
            a_free_par_guesses = [13.7, 1.24, 0.0472, 239, 0.01385, 0.0620, 0.1394, 3.3, 1.14, 0.129, 0.984, 21.29, 8.01, 0.6, 1.96, 0.46, 91.2, 432.1, 1.3]
        else:
            a_free_par_guesses = a_passed_free_par_guesses
        

        l_parameters = [a_free_par_guesses for i in xrange(iterations)]
        l_log_likelihoods = self.gpu_pool.map(self.wrapper_ln_likelihood_full_matching, l_parameters)
        #print l_log_likelihoods

        std_ll = np.std(l_log_likelihoods)

        print 'Standard deviation for %.3e MC iterations is %f' % (self.num_mc_events, std_ll)
        print 'Will scale LL such that standard deviation is 0.1'

        if std_ll < 0.1:
            self.b_suppress_likelihood = True
            self.ll_suppression_factor = 1.
            print 'Standard deviation already small so not supressing\n\n'
        else:
            self.b_suppress_likelihood = True
            self.ll_suppression_factor = std_ll / 0.1
            print 'LL suppression factor: %f\n' % self.ll_suppression_factor






if __name__ == '__main__':
    copy_reg.pickle(types.MethodType, reduce_method)


    test = fit_nr('nerix-like_nr', num_mc_events=1e4, num_gpus=1)
    test.suppress_likelihood()
    
    #test.gpu_pool.map(test.wrapper_ln_likelihood_full_matching_multiple_energies_lindhard_model, [l_test_parameters_multiple_energies_lindhard_model])
    #a_free_par_bounds = [(13, 14.2), (0.5, 1.7), (0.01, 0.75), (200, 300), (0.007, 0.019), (0.02, 0.1), (0.05, 0.25), (0.5, 10), (0.1, 3)] + [(0.12, 0.14), (0.95, 1), (20, 23), (7, 9), (0.58, 0.62), (1, 3), (0.1, 2), (50, 150), (350, 550), (0.3, 3)]
    #test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=50, popsize=15, tol=0.05)
    
    #ln_likelihood_full_matching(self, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb, g1_value, extraction_efficiency, gas_gain_value, gas_gain_width, spe_res, s1_acc_par0, s1_acc_par1, s2_acc_par0, s2_acc_par1, d_gpu_local_info, draw_fit=False)
    
    #test.gpu_pool.map(test.wrapper_ln_likelihood_full_matching, [[13.7, 1.24, 0.0472, 239, 0.01385, 0.0620, 0.1394, 3.3, 1.14, 0.129, 0.984, 21.29, 8.01, 0.6, 1.96, 0.46, 91.2, 432.1, 1.3]])
    
    
    test.run_mcmc(num_steps=10, num_walkers=256)
    
    test.close_workers()



