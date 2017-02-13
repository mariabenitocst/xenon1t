
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
import xe1t_config

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

        print 'Putting energy array on GPU...'
        d_energy_arrays = self.d_gpu_single_copy_arrays['energy']
        d_gpu_energy_arrays = {}
        for degree_setting in d_energy_arrays:
            d_gpu_energy_arrays[degree_setting] = pycuda.gpuarray.to_gpu(d_energy_arrays[degree_setting]['a_energy'])


        print 'Putting bin edges on GPU...'
        gpu_bin_edges_s1 = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_s1'])


        print 'Putting bin edges on GPU...'
        gpu_bin_edges_log = pycuda.gpuarray.to_gpu(self.d_gpu_single_copy_arrays['bin_edges_log'])


        d_gpu_local_info = {'function_to_call':gpu_observables_func,
                            'rng_states':local_rng_states,
                            'd_gpu_energy':d_gpu_energy_arrays,
                            'gpu_energy_bkg':gpu_energy_bkg,
                            'gpu_bin_edges_s1':gpu_bin_edges_s1,
                            'gpu_bin_edges_log':gpu_bin_edges_log}

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

path_to_this_module = path_to_this_module.path_to_this_module
C.register_file('%s/c_log_likelihood.C' % (xe1t_config.path_to_this_module), ['smart_log_likelihood'])
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

C.register_file('../../../python_modules/mc_code/c_safe_dot.C', ['safe_dot'])



def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))



class fit_nr(object):
    def __init__(self, d_fit_data, fit_type, num_mc_events=, name_notes=None, num_gpus=1, num_loops=1):
        
        
        # total MC events = num_mc_events*num_loops
        
        self.num_loops = num_loops
        self.fit_type = fit_type
        
        if fit_type == 'sb':
            assert len(d_fit_data['cathode_settings']) == 1
            assert d_coincidence_data['degree_settings'] == -4
        else:
            print '\nDo not recognize fit type: %s.\n' % (fit_type)
            sys.exit()

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

        self.name_notes = name_notes
        
        self.path_to_coincidence_data = xe1t_config.path_to_coincidence_data
        self.path_to_energy_spectra = xe1t_config.path_to_energy_spectra
        self.path_to_reduced_energy_spectra = xe1t_config.path_to_reduced_energy_spectra




        # ------------------------------------------------
        # ------------------------------------------------
        # Get information for data
        # ------------------------------------------------
        # ------------------------------------------------
        
        self.l_degree_settings_in_use = sorted(d_coincidence_data['degree_settings'])
        self.l_cathode_settings_in_use = sorted(d_coincidence_data['cathode_settings'])

        self.l_energy_settings = xe1t_config.l_energy_settings
        self.l_s1_settings = xe1t_config.l_s1_settings
        self.l_s2_settings = xe1t_config.l_s2_settings
        self.l_log_settings = xe1t_config.l_log_settings

        self.l_quantiles = xe1t_config.l_quantiles
        
        # define bin edges for use in MC
        self.a_s1_bin_edges = np.linspace(self.l_s1_settings[1], self.l_s1_settings[2], num=self.l_s1_settings[0]+1, dtype=np.float32)
        self.a_s2_bin_edges = np.linspace(self.l_s2_settings[1], self.l_s2_settings[2], num=self.l_s2_settings[0]+1, dtype=np.float32)
        self.a_log_bin_edges = np.linspace(self.l_log_settings[1], self.l_log_settings[2], num=self.l_log_settings[0]+1, dtype=np.float32)
        
        

        for cathode_setting in self.l_cathode_settings_in_use:
            self.d_coincidence_data_information[cathode_setting] = {}
            for degree_setting in self.l_degree_settings_in_use:
                self.d_coincidence_data_information[cathode_setting][degree_setting] = {}

                # add basic information to dictionary
                #self.d_coincidence_data_information[cathode_setting][degree_setting]['file'] = File('%s%.3fkV_%ddeg/s1_s2.pkl' % (self.path_to_coincidence_data, cathode_setting, degree_setting), 'read')
                self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2'] = pickle.load(open('%s%.3fkV_%ddeg/s1_s2.pkl' % (self.path_to_coincidence_data, cathode_setting, degree_setting), 'r'))
                
                
                self.d_coincidence_data_information[cathode_setting][degree_setting]['s1_hist'] = Hist(*self.l_s1_settings, name='hS1')
                self.d_coincidence_data_information[cathode_setting][degree_setting]['s1_hist'].fill_array(self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s1'])
                
                self.d_coincidence_data_information[cathode_setting][degree_setting]['s2_hist'] = Hist(*self.l_s2_settings, name='hS2')
                self.d_coincidence_data_information[cathode_setting][degree_setting]['s2_hist'].fill_array(self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s2'])
                
                self.d_coincidence_data_information[cathode_setting][degree_setting]['s1_s2_hist'] = Hist2D(*(self.l_s1_settings+self.l_s2_settings), name='hS1S2')
                self.d_coincidence_data_information[cathode_setting][degree_setting]['s1_s2_hist'].fill_array(np.asarray([self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s1'], self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s2']]).T)
                
                self.d_coincidence_data_information[cathode_setting][degree_setting]['log_s2_s1_hist'] = Hist2D(*(self.l_s1_settings+self.l_log_settings), name='hLogS2S1')
                self.d_coincidence_data_information[cathode_setting][degree_setting]['log_s2_s1_hist'].fill_array(np.asarray([self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s1'], np.log10(self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s2']/self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s1'])]).T)
                
                self.d_coincidence_data_information[cathode_setting][degree_setting]['num_data_pts'] = len(self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s1'])
                
                self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_s1'] = neriX_analysis.convert_2D_hist_to_matrix(self.d_coincidence_data_information[cathode_setting][degree_setting]['log_s2_s1_hist'], dtype=np.float32)
                
                
                self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_over_s1'] = np.log10(self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s2']/self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s1'])


                # ------------------------------------------------
                # Get quantiles
                # ------------------------------------------------
                
                
                # get values for bounds
                lb_s1, ub_s1 = np.percentile(self.d_coincidence_data_information[cathode_setting][degree_setting]['d_s1_s2']['s1'], self.l_quantiles)
                lb_logs2s1, ub_logs2s1 = np.percentile(self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_over_s1'], self.l_quantiles)
                
                lower_bin_s1 = -1
                upper_bin_s1 = -1
                for bin_index, bin_edge in enumerate(self.a_s1_bin_edges):
                    # get lower index
                    if self.a_s1_bin_edges[bin_index] < lb_s1 and self.a_s1_bin_edges[bin_index+1] > lb_s1:
                        lower_bin_s1 = bin_index
                        
                    if self.a_s1_bin_edges[bin_index] < ub_s1 and self.a_s1_bin_edges[bin_index+1] > ub_s1:
                        upper_bin_s1 = bin_index + 1
                        
                    if lower_bin_s1 != -1 and upper_bin_s1 != -1:
                        break

                # repeat for logs2s1
                lower_bin_logs2s1 = -1
                upper_bin_logs2s1 = -1
                for bin_index, bin_edge in enumerate(self.a_log_bin_edges):
                    # get lower index
                    #print lb_logs2s1, ub_logs2s1
                    if self.a_log_bin_edges[bin_index] < lb_logs2s1 and self.a_log_bin_edges[bin_index+1] > lb_logs2s1:
                        lower_bin_logs2s1 = bin_index
                        
                    if self.a_log_bin_edges[bin_index] < ub_logs2s1 and self.a_log_bin_edges[bin_index+1] > ub_logs2s1:
                        upper_bin_logs2s1 = bin_index + 1
                        
                    if lower_bin_logs2s1 != -1 and upper_bin_logs2s1 != -1:
                        break
                    
            
                self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_s1'] = lower_bin_s1
                self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_s1'] = upper_bin_s1
                self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_log'] = lower_bin_logs2s1
                self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_log'] = upper_bin_logs2s1
                
                



        # ------------------------------------------------
        # ------------------------------------------------
        # Check that given settings are appropriate
        # ------------------------------------------------
        # ------------------------------------------------


        for cathode_setting in self.l_cathode_settings_in_use:
            assert cathode_setting in l_allowed_cathode_settings, '%.3f kV is not an allowed cathode setting: %s' % (cathode_setting, str(self.l_allowed_cathode_settings))

        for degree_setting in self.l_degree_settings_in_use:
            assert degree_setting in l_allowed_degree_settings, '%d deg is not an allowed degree setting: %s' % (degree_setting, str(self.l_allowed_degree_settings))





        # ------------------------------------------------
        # ------------------------------------------------
        # Load MC files or create reduced file if not present
        # and load into easy_graph
        # ------------------------------------------------
        # ------------------------------------------------

        self.load_mc_data('MC')

        # fill energy array
        self.fill_energy_array(view_energy_spectrum=False)



        # ------------------------------------------------
        # ------------------------------------------------
        # Set default parameter values for use in MC
        # ------------------------------------------------
        # ------------------------------------------------


        neriX_analysis.warning_message('Detector parameters hard-coded')

        
        self.w_value = xe1t_config.w_value
        self.w_value_uncertainty = xe1t_config.w_value_uncertainty

        self.g1_value = xe1t_config.g1_value
        self.g1_uncertainty = xe1t_config.g1_uncertainty
        
        self.extraction_efficiency_value = xe1t_config.extraction_efficiency
        self.extraction_efficiency_uncertainty = xe1t_config.extraction_efficiency_uncertainty

        self.gas_gain_value = xe1t_config.gas_gain_value
        self.gas_gain_uncertainty = xe1t_config.gas_gain_uncertainty

        self.gas_gain_width = xe1t_config.gas_gain_width
        self.gas_gain_width_uncertainty = xe1t_config.gas_gain_width_uncertainty

        self.spe_res_value = xe1t_config.spe_res_value
        self.spe_res_uncertainty = xe1t_config.spe_res_uncertainty

        self.dpe_lb = xe1t_config.dpe_lb
        self.dpe_ub = xe1t_config.dpe_ub

        self.l_means_pf_eff_pars = xe1t_config.l_means_pf_eff_pars
        self.l_cov_matrix_pf_eff_pars = xe1t_config.l_cov_matrix_pf_eff_pars

        self.l_means_pax_bias = xe1t_config.l_means_pax_bias
        self.l_cov_matrix_pax_bias = xe1t_config.l_cov_matrix_pax_bias

        self.l_means_pax_smearing = xe1t_config.l_means_pax_smearing
        self.l_cov_matrix_pax_smearing = xe1t_config.l_cov_matrix_pax_smearing

        self.prob_er_value = xe1t_config.prob_er_value
        self.prob_er_uncertainty = xe1t_config.prob_er_uncertainty



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
        self.nest_lindhard_model['uncertainty'] = {'alpha':0.008,
                                                'zeta':0.008,
                                                'beta':12,
                                                'gamma':0.00065,
                                                'delta':0.006,
                                                'kappa':0.003,
                                                'eta':2,
                                                'lambda':0.25}



        if fit_type == 'sb':
            self.ln_likelihood_function = self.ln_likelihood_full_matching_single_energy
            self.ln_likelihood_function_wrapper = self.wrapper_ln_likelihood_full_matching_single_energy
            self.num_dimensions = 21
            self.gpu_function_name = 'gpu_full_observables_production_with_log_hist'
            self.directory_name = 'run_0_band'

        else:
            print 'Do not know how to handle fit type %s.  Please check input and try again.\n\n' % self.fit_type
            sys.exit()




        d_gpu_single_copy_array_dictionaries = {'energy':self.d_energy_arrays,
                                                'bin_edges_s1':self.a_s1_bin_edges,
                                                'bin_edges_log':self.a_log_bin_edges
                                                }
        self.gpu_pool = gpu_pool(num_gpus=num_gpus, grid_dim=numBlocks, block_dim=block_dim, num_dim_gpu_call=self.num_dimensions, d_gpu_single_copy_arrays=d_gpu_single_copy_array_dictionaries, function_name=self.gpu_function_name)

        self.b_suppress_likelihood = False




    def fill_energy_array(self, view_energy_spectrum=False):

        random.seed()
    
        for degree_setting in self.l_degree_settings_in_use:
    
            self.d_energy_arrays[degree_setting]['a_energy'] = np.zeros(self.num_mc_events, dtype=np.float32)
            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                self.d_energy_arrays[degree_setting]['a_energy'][i] = self.d_energy_arrays[degree_setting]['h_energy'].GetRandom()
                    
            if view_energy_spectrum:
                cEnergySpec = Canvas()
                hEnergySpec = Hist(100, 0, 40)
                hEnergySpec.fill_array(self.d_energy_arrays[degree_setting]['a_energy'])
                hEnergySpec.Draw()
                cEnergySpec.Update()
                raw_input('Press enter to continue...')
                
                del hEnergySpec
                del cEnergySpec

        # fill bkg energy array
        # same for all degrees
        self.a_energy_bkg = np.zeros(self.num_mc_events, dtype=np.float32)
        for i in tqdm.tqdm(xrange(self.num_mc_events)):
            self.a_energy_bkg[i] = self.h_energy_bkg.GetRandom()





    def load_mc_data(self, sFileType):

        self.d_energy_arrays = {}

        for degree_setting in self.l_degree_settings_in_use:
            self.d_energy_arrays[degree_setting] = {}

            # check if reduced file exists
            coincidence_degree_setting = True
            if degree_setting > 0:
                current_path_to_reduced_energy_spectra = '%snerixsim-%dkeV.root' % (self.path_to_reduced_energy_spectra, self.d_degree_setting_to_energy_name[degree_setting])
            else:
                current_path_to_reduced_energy_spectra = '%snerixsim-pure_nr_spec.root' % (self.path_to_reduced_energy_spectra)
                coincidence_degree_setting = False
            
            
            
            if coincidence_degree_setting and not os.path.exists(current_path_to_reduced_energy_spectra):
                # if the file doesn't exist we need to create it!
            
                # load the raw data file
                current_path_to_energy_spectra = '%snerixsim-%dkeV.root' % (self.path_to_energy_spectra, self.d_degree_setting_to_energy_name[degree_setting])
                f_simulation = File(current_path_to_energy_spectra, 'read')

                if sFileType == 'MC':

                    # set cuts
                    xRadius = '(sqrt(xpos[0]**2+ypos[0]**2) < 18)'
                    xZ = '(zpos[0]>-20 && zpos[0]<-4)'
                    xSingleScatter = '(nsteps_target==1)'
                    xLiqSciHeight = '(etotliqsci>700)'
                    xLXeEnergy = '(etot_target>0)'
                    #xLXeEnergy = '(etot_target>0)'
                    xLXeEnergy = '(Alt$(ed_nr_target[1],-1)==-1 && ed_nr_target[0] > 0)'

                    #print 'No TOF cut!\n'
                    neriX_analysis.warning_message('Need to finalize how tof is chosen in MC data!')
                    
                    low_time_tof = -1
                    high_time_tof = -1
                    if degree_setting == 2300:
                        low_time_tof = 24
                        high_time_tof = 34
                    elif degree_setting == 3000:
                        low_time_tof = 31
                        high_time_tof = 41
                    elif degree_setting == 3500:
                        low_time_tof = 15
                        high_time_tof = 25
                    elif degree_setting == 4500:
                        low_time_tof = 22
                        high_time_tof = 34
                    elif degree_setting == 5300:
                        low_time_tof = 14
                        high_time_tof = 24
                    elif degree_setting == 6200:
                        low_time_tof = 8
                        high_time_tof = 25
                    
                    
                    xTOF = '(timeliqsci-tpos[0]> %d && timeliqsci-tpos[0] < %d)' % (low_time_tof, high_time_tof)

                    xAll = '%s && %s && %s && %s && %s && %s' % (xRadius, xZ, xSingleScatter, xLiqSciHeight, xLXeEnergy, xTOF)
                
                elif sFileType == 'MC_accidentals':
                    # set cuts
                    xRadius = '(sqrt(xpos[0]**2+ypos[0]**2) < 18)'
                    xZ = '(zpos[0]>-20 && zpos[0]<-4)'
                    xSingleScatter = '(nsteps_target==1)'
                    #xLXeEnergy = '(etot_target>0)' # this includes ER
                    xSingleScatter_2 = '(Alt$(ed_nr_target[1],-1)==-1)'
                    xLXeEnergy = '(ed_nr_target[0] > 0.)'

                    xAll = '%s && %s && %s && %s && %s' % (xRadius, xZ, xSingleScatter, xLXeEnergy, xSingleScatter_2)


                # create and fill histogram
                t_mc = f_simulation.t2
                h_mc = Hist(*self.l_energy_settings, name='h_mc', title='Simulated Spectrum', drawstyle='hist')
                #t_mc.Draw('etot_target', hist=hMC, selection=xAll)
                t_mc.Draw('ed_nr_target[0]', hist=h_mc, selection=xAll)
                h_mc.Sumw2()

                # draw histogram to new file for quick retrieval
                f_reduced = File(current_path_to_reduced_energy_spectra, 'recreate')
                h_mc.Write()
                f_reduced.Close()
            
                neriX_analysis.success_message('Created new reduced MC file!')
            
            

            self.f_reduced = File(current_path_to_reduced_energy_spectra, 'read')
            h_mc = self.f_reduced.h_mc
            #g_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_mc)
            
            # now get accidental bkg
            current_path_to_bkg_spectra = '%snerixsim-pure_nr_spec.root' % (self.path_to_reduced_energy_spectra)
            self.f_reduced_bkg = File(current_path_to_bkg_spectra, 'read')
            h_mc_bkg = self.f_reduced_bkg.h_mc

            self.d_energy_arrays[degree_setting]['h_energy'] = h_mc
            self.h_energy_bkg = h_mc_bkg





    def get_prior_log_likelihood_nuissance(self, likelihoodNuissance):
        if likelihoodNuissance > 1e-550:
            return np.log(likelihoodNuissance)
        else:
            return -np.inf



    def get_prior_log_likelihood_dpe_prob(self, dpe_prob):
        if 0.17 < dpe_prob < 0.24:
            return 0
        else:
            return -np.inf



    def get_log_likelihood_s1_eff(self, l_s1_eff_pars):
        # first par is center, second is shape
        if -10 < l_s1_eff_pars[0] < 10 and 0.01 < l_s1_eff_pars[1] < 10:
            return 0.
        else:
            return -np.inf



    def get_log_likelihood_s2_eff(self, l_s2_eff_pars):
        # first par is center, second is shape
        if -100 < l_s2_eff_pars[0] < 1000 and 1 < l_s2_eff_pars[1] < 1000:
            return 0.
        else:
            return -np.inf



    def get_prior_log_likelihood_probabilities(self, scale):
        if isinstance(scale, dict):
            for id in scale:
                # check if 2d dict
                if isinstance(scale[id], dict):
                    for id_2 in scale[id]:
                        if scale[id][id_2] < 0. or scale[id][id_2] > 1.:
                            return -np.inf
                        else:
                            return 0.
            
                else:
                    if scale[id] < 0. or scale[id] > 1.:
                        return -np.inf
                    else:
                        return 0.
        
        else:
            if scale < 0. or scale > 1.:
                return -np.inf
            else:
                return 0



    def get_prior_log_likelihood_resolution(self, intrinsicResolution):
        if intrinsicResolution < 0 or intrinsicResolution > 1.5:
            return -np.inf
        else:
            return 0



    def get_prior_log_likelihood_scale_par(self, scale_par):
        if scale_par < 0:
            return -np.inf
        else:
            return 0



    def get_prior_log_likelihood_light_yields(self, l_yields):
        for value in l_yields:
            if value < 3. or value > 15.:
                return -np.inf
        return 0.



    def get_prior_log_likelihood_charge_yields(self, l_yields):
        for value in l_yields:
            if value < 3.5 or value > 15.:
                return -np.inf
        return 0.
    
    
    
    def get_prior_log_likelihood_alpha(self, alpha):
        if alpha < 0:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_zeta(self, zeta):
        if zeta < 0:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_beta(self, beta):
        # beta represents an exponential suppression
        # so above certain limit it has no effect (~5000)
        if beta < 0 or beta > 5000.:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_gamma(self, gamma):
        if gamma < 0:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_delta(self, delta):
        if delta < 0:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_kappa(self, kappa):
        if kappa < 0:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_eta(self, eta):
        if eta < 0:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_lamb(self, lamb):
        if lamb < 0:
            return -np.inf
        else:
            return 0.



    def get_prior_log_likelihood_probability(self, prob):
        if prob < 0 or prob > 1:
            return -np.inf
        else:
            return 0.



    def get_indices_for_given_quantile(self, aBinCenters, aCountsInBins, leftPercentile, rightPercentile):
        aX = aBinCenters.flatten()
        aY = aCountsInBins.flatten()

        totalCounts = int(np.rint(np.sum(aCountsInBins)))
        aForQuantiles = np.zeros(totalCounts)

        # build array for quantile analysis
        # essentially add another index for each count in array
        currentIndex = 0

        for i in xrange(len(aX)):
            for j in xrange(int(np.rint(aY[i]))):
                aForQuantiles[currentIndex] = aX[i]
                currentIndex += 1

        valueLeft = np.percentile(aForQuantiles, leftPercentile*100)
        valueRight = np.percentile(aForQuantiles, rightPercentile*100)

        indexLeft = np.argmin(np.abs(aBinCenters - valueLeft))
        indexRight = np.argmin(np.abs(aBinCenters - valueRight))

        # add 1 to right index to keep it inclusive
        return (indexLeft, indexRight + 1)





    def ln_likelihood_full_matching_single_energy(self, py, qy, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, exciton_to_ion_par0_rv, exciton_to_ion_par1_rv, exciton_to_ion_par2_rv, prob_bkg_event, scale_par, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------


        prior_ln_likelihood = 0


        # priors of yields
        # can do them all together since uniform within
        # the same range
        prior_ln_likelihood += self.get_prior_log_likelihood_light_yields([py])
        prior_ln_likelihood += self.get_prior_log_likelihood_charge_yields([qy])


        # priors of resolutions
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s1)
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s2)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_probabilities(scale_par)


        # get exciton to ion prior
        current_likelihood = self.get_likelihood_exciton_to_ion(exciton_to_ion_par0_rv, exciton_to_ion_par1_rv, exciton_to_ion_par2_rv)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        # priors of detector variables
        current_likelihood, g1_value = self.get_g1_default(g1_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        current_likelihood, extraction_efficiency = self.get_extraction_efficiency_default(extraction_efficiency_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, spe_res = self.get_spe_res_default(spe_res_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_value = self.get_gas_gain_default(gas_gain_mean_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_width = self.get_gas_gain_width_default(gas_gain_width_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        # priors of efficiencies
        current_likelihood, pf_eff_par0, pf_eff_par1 = self.get_pf_eff_default(pf_eff_par0, pf_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, s2_eff_par0, s2_eff_par1 = self.get_s2_eff_default(s2_eff_par0, s2_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_ln_likelihood, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2 = self.get_pf_stdev_default(pf_stdev_par0, pf_stdev_par1, pf_stdev_par2)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_ln_likelihood)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_probability(prob_bkg_event)


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
        mean_field = np.asarray(self.d_cathode_voltages_to_field[self.l_cathode_settings_in_use[0]], dtype=np.float32)

        py = np.asarray(py, dtype=np.float32)
        qy = np.asarray(qy, dtype=np.float32)

        g1_value = np.asarray(g1_value, dtype=np.float32)
        extraction_efficiency = np.asarray(extraction_efficiency, dtype=np.float32)
        spe_res = np.asarray(spe_res, dtype=np.float32)
        gas_gain_value = np.asarray(gas_gain_value, dtype=np.float32)
        gas_gain_width = np.asarray(gas_gain_width, dtype=np.float32)

        intrinsic_res_s1 = np.asarray(intrinsic_res_s1, dtype=np.float32)
        intrinsic_res_s2 = np.asarray(intrinsic_res_s2, dtype=np.float32)

        exciton_to_ion_par0_rv = np.asarray(exciton_to_ion_par0_rv, dtype=np.float32)
        exciton_to_ion_par1_rv = np.asarray(exciton_to_ion_par1_rv, dtype=np.float32)
        exciton_to_ion_par2_rv = np.asarray(exciton_to_ion_par2_rv, dtype=np.float32)

        pf_eff_par0 = np.asarray(pf_eff_par0, dtype=np.float32)
        pf_eff_par1 = np.asarray(pf_eff_par1, dtype=np.float32)

        s2_eff_par0 = np.asarray(s2_eff_par0, dtype=np.float32)
        s2_eff_par1 = np.asarray(s2_eff_par1, dtype=np.float32)

        a_pf_stdev = np.asarray([pf_stdev_par0, pf_stdev_par1, pf_stdev_par2], dtype=np.float32)
        
        prob_bkg_event = np.asarray(prob_bkg_event, dtype=np.float32)

        # for histogram binning
        num_bins_s1 = np.asarray(self.l_s1_settings[0], dtype=np.int32)
        num_bins_log = np.asarray(self.l_log_settings[0], dtype=np.int32)
        a_hist_2d = np.zeros(self.l_s1_settings[0]*self.l_log_settings[0], dtype=np.float32)
        
        num_loops = np.asarray(self.num_loops, dtype=np.int32)


        tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['d_gpu_energy'][self.l_degree_settings_in_use[0]], d_gpu_local_info['gpu_energy_bkg'], drv.In(py), drv.In(qy), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(spe_res), drv.In(intrinsic_res_s1), drv.In(intrinsic_res_s2), drv.In(a_pf_stdev), drv.In(exciton_to_ion_par0_rv), drv.In(exciton_to_ion_par1_rv), drv.In(exciton_to_ion_par2_rv), drv.In(pf_eff_par0), drv.In(pf_eff_par1), drv.In(s2_eff_par0), drv.In(s2_eff_par1), drv.In(prob_bkg_event), drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d), drv.In(num_loops))

        d_gpu_local_info['function_to_call'](*tArgs, **self.d_gpu_scale)

        a_s1_s2_mc = np.reshape(a_hist_2d, (self.l_log_settings[0], self.l_s1_settings[0])).T

        sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
        if sum_mc == 0:
            return -np.inf

        #a_s1_s2_mc = np.multiply(a_s1_s2_mc, np.sum(self.a_s1_s2, dtype=np.float32) / sum_mc)

        # if making PDF rather than scaling for rate
        scale_par *= float(self.num_mc_events*self.num_loops) / sum_mc

        a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['num_data_pts']/float(self.num_mc_events*self.num_loops))

        # likelihood for single energy
        if draw_fit:

            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

            s1_s2_data_plot = np.rot90(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['a_log_s2_s1'])
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
            root_numpy.array2hist(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].sum(axis=1), h_s1_data)

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
            root_numpy.array2hist(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].sum(axis=0), h_s2_data)

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

            neriX_analysis.save_plot(['temp_results'], c1, '1d_hists', batch_mode=True)
            f.savefig('./temp_results/2d_hist.png')

            #plt.show()

        #flat_s1_log_data = np.asarray(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].flatten(), dtype=np.float32)
        #flat_s1_log_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
        
        flat_s1_log_data = np.asarray(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['a_log_s2_s1'][self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
        flat_s1_log_mc = np.asarray(a_s1_s2_mc[self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)


        flat_s1_log_data = np.asarray(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['a_log_s2_s1'][self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
        flat_s1_log_mc = np.asarray(a_s1_s2_mc[self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
        logLikelihoodMatching = c_log_likelihood(flat_s1_log_data, flat_s1_log_mc, len(flat_s1_log_data), int(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][self.l_degree_settings_in_use[0]]['num_data_pts']), 0.95)
        
        total_ln_likelihood = logLikelihoodMatching + prior_ln_likelihood

        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor

        #print logLikelihoodMatching, prior_ln_likelihood
        #print 'Total time: %f' % (time.time() - start_time_tot_ll)
        #print total_ln_likelihood

        if np.isnan(total_ln_likelihood):
            return -np.inf
        else:
            return total_ln_likelihood


        
    def wrapper_ln_likelihood_full_matching_single_energy(self, a_parameters, kwargs={}):
        return self.ln_likelihood_full_matching_single_energy(*a_parameters, **kwargs)
    
    
    # band matching
    def ln_likelihood_full_matching_band(self, w_value, alpha, zeta, beta, gamma, delta, kappa, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, d_scale_pars, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------

        #start_time_full = time.time()

        prior_ln_likelihood = 0
        matching_ln_likelihood = 0


        current_likelihood, w_value = self.get_w_value_default(w_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_alpha(alpha)
        prior_ln_likelihood += self.get_prior_log_likelihood_zeta(zeta)
        prior_ln_likelihood += self.get_prior_log_likelihood_beta(beta)
        prior_ln_likelihood += self.get_prior_log_likelihood_gamma(gamma)
        prior_ln_likelihood += self.get_prior_log_likelihood_delta(delta)
        prior_ln_likelihood += self.get_prior_log_likelihood_kappa(kappa)


        # priors of resolutions
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s1)
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s2)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_probabilities(d_scale_pars)



        # priors of detector variables
        current_likelihood, g1_value = self.get_g1_default(g1_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        current_likelihood, extraction_efficiency = self.get_extraction_efficiency_default(extraction_efficiency_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, spe_res = self.get_spe_res_default(spe_res_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_value = self.get_gas_gain_default(gas_gain_mean_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_width = self.get_gas_gain_width_default(gas_gain_width_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        # priors of efficiencies
        current_likelihood, pf_eff_par0, pf_eff_par1 = self.get_pf_eff_default(pf_eff_par0, pf_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, s2_eff_par0, s2_eff_par1 = self.get_s2_eff_default(s2_eff_par0, s2_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_ln_likelihood, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2 = self.get_pf_stdev_default(pf_stdev_par0, pf_stdev_par1, pf_stdev_par2)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_ln_likelihood)
        


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

        w_value = np.asarray(w_value, dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        zeta = np.asarray(zeta, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)
        gamma = np.asarray(gamma, dtype=np.float32)
        delta = np.asarray(delta, dtype=np.float32)
        kappa = np.asarray(kappa, dtype=np.float32)

        g1_value = np.asarray(g1_value, dtype=np.float32)
        extraction_efficiency = np.asarray(extraction_efficiency, dtype=np.float32)
        spe_res = np.asarray(spe_res, dtype=np.float32)
        gas_gain_value = np.asarray(gas_gain_value, dtype=np.float32)
        gas_gain_width = np.asarray(gas_gain_width, dtype=np.float32)

        intrinsic_res_s1 = np.asarray(intrinsic_res_s1, dtype=np.float32)
        intrinsic_res_s2 = np.asarray(intrinsic_res_s2, dtype=np.float32)

        #print 'hard coding pf eff'
        #pf_eff_par0 = np.asarray(0.01, dtype=np.float32)
        #pf_eff_par1 = np.asarray(0.01, dtype=np.float32)
        pf_eff_par0 = np.asarray(pf_eff_par0, dtype=np.float32)
        pf_eff_par1 = np.asarray(pf_eff_par1, dtype=np.float32)

        s2_eff_par0 = np.asarray(s2_eff_par0, dtype=np.float32)
        s2_eff_par1 = np.asarray(s2_eff_par1, dtype=np.float32)

        a_pf_stdev = np.asarray([pf_stdev_par0, pf_stdev_par1, pf_stdev_par2], dtype=np.float32)
        
        # for histogram binning
        num_bins_s1 = np.asarray(self.l_s1_settings[0], dtype=np.int32)
        num_bins_log = np.asarray(self.l_log_settings[0], dtype=np.int32)
        a_hist_2d = np.zeros(self.l_s1_settings[0]*self.l_log_settings[0], dtype=np.float32)
        
        num_loops = np.asarray(self.num_loops, dtype=np.int32)



        for cathode_setting in self.l_cathode_settings_in_use:
            mean_field = np.asarray(self.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)
            
            #start_time_mc = time.time()
        
            tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['d_gpu_energy'][self.l_degree_settings_in_use[0]], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(spe_res), drv.In(intrinsic_res_s1), drv.In(intrinsic_res_s2), drv.In(a_pf_stdev), drv.In(pf_eff_par0), drv.In(pf_eff_par1), drv.In(s2_eff_par0), drv.In(s2_eff_par1), drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d), drv.In(num_loops))

            d_gpu_local_info['function_to_call'](*tArgs, **self.d_gpu_scale)
            
            #print 'MC time: %f' % (time.time() - start_time_mc)
            #start_time_tot_ll = time.time()

            a_s1_s2_mc = np.reshape(a_hist_2d, (self.l_log_settings[0], self.l_s1_settings[0])).T

            sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
            if sum_mc == 0:
                return -np.inf

            #a_s1_s2_mc = np.multiply(a_s1_s2_mc, np.sum(self.a_s1_s2, dtype=np.float32) / sum_mc)

            # if making PDF rather than scaling for rate
            scale_par = d_scale_pars[cathode_setting]
            scale_par *= float(self.num_mc_events*self.num_loops) / sum_mc

            a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['num_data_pts']/float(self.num_mc_events*self.num_loops))

            # likelihood for band
            if draw_fit:

                f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

                s1_s2_data_plot = np.rot90(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'])
                s1_s2_data_plot = np.flipud(s1_s2_data_plot)
                ax1.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_data_plot)

                s1_s2_mc_plot = np.rot90(a_s1_s2_mc)
                s1_s2_mc_plot = np.flipud(s1_s2_mc_plot)
                #print self.l_s1_settings
                #print self.l_log_settings
                #print self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].shape
                #print a_s1_s2_mc.shape
                #print s1_s2_data_plot.shape
                #print s1_s2_mc_plot.shape
                ax2.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_mc_plot)
                #plt.colorbar()


                c1 = Canvas(1400, 400)
                c1.Divide(2)

                h_s1_data = Hist(*self.l_s1_settings, name='hS1_draw_data')
                root_numpy.array2hist(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].sum(axis=1), h_s1_data)

                hS1MC = Hist(*self.l_s1_settings, name='hS1_draw_mc')
                root_numpy.array2hist(a_s1_s2_mc.sum(axis=1), hS1MC)

                s1_scale_factor = h_s1_data.Integral() / float(hS1MC.Integral())

                g_s1_data = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s1_data)
                g_s1_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(hS1MC, scale=1.)

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
                root_numpy.array2hist(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].sum(axis=0), h_s2_data)

                h_s2_mc = Hist(*self.l_log_settings, name='h_s2_draw_mc')
                root_numpy.array2hist(a_s1_s2_mc.sum(axis=0), h_s2_mc)

                s2_scale_factor = h_s2_data.Integral() / float(h_s2_mc.Integral())

                g_s2_data = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s2_data)
                g_s2_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s2_mc, scale=1.)

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

                neriX_analysis.save_plot(['temp_results'], c1, '1d_hists_%.3f_kV' % cathode_setting, batch_mode=True)
                f.savefig('./temp_results/2d_hist_%.3f_kV.png' % cathode_setting)

                #plt.show()

            #flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].flatten(), dtype=np.float32)
            #flat_s1_log_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
            
            flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'][self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
            flat_s1_log_mc = np.asarray(a_s1_s2_mc[self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)


            flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'][self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
            flat_s1_log_mc = np.asarray(a_s1_s2_mc[self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
            logLikelihoodMatching = c_log_likelihood(flat_s1_log_data, flat_s1_log_mc, len(flat_s1_log_data), int(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['num_data_pts']), 0.95)
            
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
        #print a_parameters
        #print '\n\n\n'
        d_scale_pars = {}
        for i, cathode_setting in enumerate(self.l_cathode_settings_in_use):
            d_scale_pars[cathode_setting] = a_parameters[-4+i]
        
        return self.ln_likelihood_full_matching_band(*(list(a_parameters[:-4]) + [d_scale_pars, a_parameters[-1]]), **kwargs)





    def ln_likelihood_full_matching_multiple_energies(self, a_py, a_qy, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_rv, g2_value, gas_gain_rv, gas_gain_width_rv, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, exciton_to_ion_par0_rv, exciton_to_ion_par1_rv, exciton_to_ion_par2_rv, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------


        prior_ln_likelihood = 0
        matching_ln_likelihood = 0


        # priors of yields
        # can do them all together since uniform within
        # the same range
        prior_ln_likelihood += self.get_prior_log_likelihood_light_yields(a_py)
        prior_ln_likelihood += self.get_prior_log_likelihood_charge_yields(a_qy)


        # priors of resolutions
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s1)
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s2)


        # get exciton to ion prior
        current_likelihood = self.get_likelihood_exciton_to_ion(exciton_to_ion_par0_rv, exciton_to_ion_par1_rv, exciton_to_ion_par2_rv)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        # priors of detector variables
        current_likelihood, g1_value, g2_value = self.get_g1_g2_default(g1_value, g2_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, spe_res = self.get_spe_res_default(spe_res_rv)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_value = self.get_gas_gain_default(gas_gain_rv)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_width = self.get_gas_gain_width_default(gas_gain_width_rv)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        # priors of efficiencies
        current_likelihood, pf_eff_par0, pf_eff_par1 = self.get_pf_eff_default(pf_eff_par0, pf_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, s2_eff_par0, s2_eff_par1 = self.get_s2_eff_default(s2_eff_par0, s2_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_ln_likelihood, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2 = self.get_pf_stdev_default(pf_stdev_par0, pf_stdev_par1, pf_stdev_par2)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_ln_likelihood)

        extraction_efficiency = g2_value / float(gas_gain_value)


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
        mean_field = np.asarray(self.d_cathode_voltages_to_field[self.l_cathode_settings_in_use[0]], dtype=np.float32)
        a_spline_energies = np.asarray(self.a_spline_energies, dtype=np.float32)
        num_spline_pts = np.asarray(len(self.a_spline_energies), dtype=np.int32)

        a_py = np.asarray(a_py, dtype=np.float32)
        a_qy = np.asarray(a_qy, dtype=np.float32)

        g1_value = np.asarray(g1_value, dtype=np.float32)
        extraction_efficiency = np.asarray(extraction_efficiency, dtype=np.float32)
        spe_res = np.asarray(spe_res, dtype=np.float32)
        gas_gain_value = np.asarray(gas_gain_value, dtype=np.float32)
        gas_gain_width = np.asarray(gas_gain_width, dtype=np.float32)

        intrinsic_res_s1 = np.asarray(intrinsic_res_s1, dtype=np.float32)
        intrinsic_res_s2 = np.asarray(intrinsic_res_s2, dtype=np.float32)

        exciton_to_ion_par0_rv = np.asarray(exciton_to_ion_par0_rv, dtype=np.float32)
        exciton_to_ion_par1_rv = np.asarray(exciton_to_ion_par1_rv, dtype=np.float32)
        exciton_to_ion_par2_rv = np.asarray(exciton_to_ion_par2_rv, dtype=np.float32)

        pf_eff_par0 = np.asarray(pf_eff_par0, dtype=np.float32)
        pf_eff_par1 = np.asarray(pf_eff_par1, dtype=np.float32)

        s2_eff_par0 = np.asarray(s2_eff_par0, dtype=np.float32)
        s2_eff_par1 = np.asarray(s2_eff_par1, dtype=np.float32)

        a_pf_stdev = np.asarray([pf_stdev_par0, pf_stdev_par1, pf_stdev_par2], dtype=np.float32)
        
        a_nr_band_cut = np.asarray([-1000000000, 0, 0, 1, 1000000000000], dtype=np.float32)

        # for histogram binning
        num_bins_s1 = np.asarray(self.l_s1_settings[0], dtype=np.int32)
        num_bins_log = np.asarray(self.l_log_settings[0], dtype=np.int32)

        for degree_setting in self.l_degree_settings_in_use:
        
            a_hist_2d = np.zeros(self.l_s1_settings[0]*self.l_log_settings[0], dtype=np.int32)
            
            #print d_gpu_local_info['d_gpu_energy'][degree_setting]
            
            tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['d_gpu_energy'][degree_setting], drv.In(num_spline_pts), drv.In(a_spline_energies), drv.In(a_py), drv.In(a_qy), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(spe_res), drv.In(intrinsic_res_s1), drv.In(intrinsic_res_s2), drv.In(a_pf_stdev), drv.In(exciton_to_ion_par0_rv), drv.In(exciton_to_ion_par1_rv), drv.In(exciton_to_ion_par2_rv), drv.In(pf_eff_par0), drv.In(pf_eff_par1), drv.In(s2_eff_par0), drv.In(s2_eff_par1), drv.In(a_nr_band_cut), drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d))

            d_gpu_local_info['function_to_call'](*tArgs, **self.d_gpu_scale)
            

            a_s1_s2_mc = np.reshape(a_hist_2d, (self.l_log_settings[0], self.l_s1_settings[0])).T
            
            #print a_s1_s2_mc

            sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
            if sum_mc == 0:
                #print 'sum mc == 0'
                return -np.inf

            scale_par = float(self.num_mc_events*self.num_loops) / sum_mc

            a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][degree_setting]['num_data_pts']/float(self.num_mc_events*self.num_loops))

            if draw_fit:

                f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

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

                g_s2_data.SetTitle('S2 Comparison')
                g_s2_data.GetXaxis().SetTitle('S2 [PE]')
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

            flat_s1_log_data = np.asarray(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][degree_setting]['a_log_s2_s1'].flatten(), dtype=np.float32)
            flat_s1_log_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
            logLikelihoodMatching = c_log_likelihood(flat_s1_log_data, flat_s1_log_mc, len(flat_s1_log_data), int(self.num_mc_events*self.num_loops), scale_par, int(self.d_coincidence_data_information[self.l_cathode_settings_in_use[0]][degree_setting]['num_data_pts']), 0.95)
            
            if np.isnan(logLikelihoodMatching):
                #print '\nbad matching:'
                #print logLikelihoodMatching
                #print flat_s1_log_data
                #print flat_s1_log_mc
                return -np.inf
            else:
                matching_ln_likelihood += logLikelihoodMatching
            
            #print degree_setting
            #print matching_ln_likelihood
                
        total_ln_likelihood = prior_ln_likelihood + matching_ln_likelihood
        #print total_ln_likelihood
                
        if np.isnan(total_ln_likelihood):
            return -np.inf
        else:
            return total_ln_likelihood
            
            
            
    def wrapper_ln_likelihood_full_matching_multiple_energies(self, a_parameters, kwargs={}):
        num_non_yield_pars = 17 + 1
        l_non_yield_parameters = a_parameters[-num_non_yield_pars:]
        l_pars = [list(a_parameters[0:(len(a_parameters)-num_non_yield_pars)/2]), list(a_parameters[(len(a_parameters)-num_non_yield_pars)/2:len(a_parameters)-num_non_yield_pars])]
        for par in l_non_yield_parameters:
            l_pars.append(par)
        #print l_pars
        #print 'call'
        return self.ln_likelihood_full_matching_multiple_energies(*l_pars, **kwargs)
        


    def ln_likelihood_full_matching_multiple_energies_lindhard_model(self, w_value, alpha, zeta, beta, gamma, delta, kappa, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, dpe_prob, d_bkg_probs, d_scale_pars, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------

        #start_time_full = time.time()

        prior_ln_likelihood = 0
        matching_ln_likelihood = 0


        current_likelihood, w_value = self.get_w_value_default(w_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_alpha(alpha)
        prior_ln_likelihood += self.get_prior_log_likelihood_zeta(zeta)
        prior_ln_likelihood += self.get_prior_log_likelihood_beta(beta)
        prior_ln_likelihood += self.get_prior_log_likelihood_gamma(gamma)
        prior_ln_likelihood += self.get_prior_log_likelihood_delta(delta)
        prior_ln_likelihood += self.get_prior_log_likelihood_kappa(kappa)


        # priors of resolutions
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s1)
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s2)
        
        # scale prior can be used for both (limits between 0 and 1)
        prior_ln_likelihood += self.get_prior_log_likelihood_probabilities(d_bkg_probs)
        prior_ln_likelihood += self.get_prior_log_likelihood_probabilities(d_scale_pars)



        # priors of detector variables
        current_likelihood, g1_value = self.get_g1_default(g1_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        current_likelihood, extraction_efficiency = self.get_extraction_efficiency_default(extraction_efficiency_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, spe_res = self.get_spe_res_default(spe_res_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_value = self.get_gas_gain_default(gas_gain_mean_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_width = self.get_gas_gain_width_default(gas_gain_width_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        # priors of efficiencies
        current_likelihood, pf_eff_par0, pf_eff_par1 = self.get_pf_eff_default(pf_eff_par0, pf_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, s2_eff_par0, s2_eff_par1 = self.get_s2_eff_default(s2_eff_par0, s2_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_ln_likelihood, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2 = self.get_pf_stdev_default(pf_stdev_par0, pf_stdev_par1, pf_stdev_par2)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_ln_likelihood)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_dpe_prob(dpe_prob)
        

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

        w_value = np.asarray(w_value, dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        zeta = np.asarray(zeta, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)
        gamma = np.asarray(gamma, dtype=np.float32)
        delta = np.asarray(delta, dtype=np.float32)
        kappa = np.asarray(kappa, dtype=np.float32)

        g1_value = np.asarray(g1_value, dtype=np.float32)
        extraction_efficiency = np.asarray(extraction_efficiency, dtype=np.float32)
        spe_res = np.asarray(spe_res, dtype=np.float32)
        gas_gain_value = np.asarray(gas_gain_value, dtype=np.float32)
        gas_gain_width = np.asarray(gas_gain_width, dtype=np.float32)

        intrinsic_res_s1 = np.asarray(intrinsic_res_s1, dtype=np.float32)
        intrinsic_res_s2 = np.asarray(intrinsic_res_s2, dtype=np.float32)

        #print 'hard coding pf eff'
        #pf_eff_par0 = np.asarray(0.01, dtype=np.float32)
        #pf_eff_par1 = np.asarray(0.01, dtype=np.float32)
        pf_eff_par0 = np.asarray(pf_eff_par0, dtype=np.float32)
        pf_eff_par1 = np.asarray(pf_eff_par1, dtype=np.float32)

        s2_eff_par0 = np.asarray(s2_eff_par0, dtype=np.float32)
        s2_eff_par1 = np.asarray(s2_eff_par1, dtype=np.float32)

        a_pf_stdev = np.asarray([pf_stdev_par0, pf_stdev_par1, pf_stdev_par2], dtype=np.float32)
        
        dpe_prob = np.asarray(dpe_prob, dtype=np.float32)
        
        # for histogram binning
        num_bins_s1 = np.asarray(self.l_s1_settings[0], dtype=np.int32)
        num_bins_log = np.asarray(self.l_log_settings[0], dtype=np.int32)
        a_hist_2d = np.zeros(self.l_s1_settings[0]*self.l_log_settings[0], dtype=np.float32)
        
        num_loops = np.asarray(self.num_loops, dtype=np.int32)


        for cathode_setting in self.l_cathode_settings_in_use:
            for degree_setting in self.l_degree_settings_in_use:
                
                mean_field = np.asarray(self.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)
                
                #start_time_mc = time.time()
            
                tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['d_gpu_energy'][degree_setting], d_gpu_local_info['gpu_energy_bkg'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(spe_res), drv.In(intrinsic_res_s1), drv.In(intrinsic_res_s2), drv.In(a_pf_stdev), drv.In(pf_eff_par0), drv.In(pf_eff_par1), drv.In(s2_eff_par0), drv.In(s2_eff_par1), drv.In(dpe_prob), np.asarray(d_bkg_probs[cathode_setting][degree_setting], dtype=np.float32), drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d), drv.In(num_loops))

                d_gpu_local_info['function_to_call'](*tArgs, **self.d_gpu_scale)
                
                #print 'MC time: %f' % (time.time() - start_time_mc)
                #start_time_tot_ll = time.time()

                a_s1_s2_mc = np.reshape(a_hist_2d, (self.l_log_settings[0], self.l_s1_settings[0])).T

                sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
                if sum_mc == 0:
                    return -np.inf
                

                #a_s1_s2_mc = np.multiply(a_s1_s2_mc, np.sum(self.a_s1_s2, dtype=np.float32) / sum_mc)

                # correct scale factor for number of events
                scale_par = d_scale_pars[cathode_setting][degree_setting]
                scale_par *= float(self.num_mc_events*self.num_loops) / sum_mc

                a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information[cathode_setting][degree_setting]['num_data_pts']/float(self.num_mc_events*self.num_loops))

                # likelihood for band
                if draw_fit:

                    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

                    s1_s2_data_plot = np.rot90(self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_s1'])
                    s1_s2_data_plot = np.flipud(s1_s2_data_plot)
                    ax1.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_data_plot)

                    s1_s2_mc_plot = np.rot90(a_s1_s2_mc)
                    s1_s2_mc_plot = np.flipud(s1_s2_mc_plot)
                    #print self.l_s1_settings
                    #print self.l_log_settings
                    #print self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_s1'].shape
                    #print a_s1_s2_mc.shape
                    #print s1_s2_data_plot.shape
                    #print s1_s2_mc_plot.shape
                    ax2.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_mc_plot)
                    #plt.colorbar()


                    c1 = Canvas(1400, 400)
                    c1.Divide(2)

                    h_s1_data = Hist(*self.l_s1_settings, name='hS1_draw_data')
                    root_numpy.array2hist(self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_s1'].sum(axis=1), h_s1_data)

                    hS1MC = Hist(*self.l_s1_settings, name='hS1_draw_mc')
                    root_numpy.array2hist(a_s1_s2_mc.sum(axis=1), hS1MC)

                    s1_scale_factor = h_s1_data.Integral() / float(hS1MC.Integral())

                    g_s1_data = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s1_data)
                    g_s1_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(hS1MC, scale=1.)

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
                    root_numpy.array2hist(self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_s1'].sum(axis=0), h_s2_data)

                    h_s2_mc = Hist(*self.l_log_settings, name='h_s2_draw_mc')
                    root_numpy.array2hist(a_s1_s2_mc.sum(axis=0), h_s2_mc)

                    s2_scale_factor = h_s2_data.Integral() / float(h_s2_mc.Integral())

                    g_s2_data = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s2_data)
                    g_s2_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s2_mc, scale=1.)

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

                    neriX_analysis.save_plot(['temp_results'], c1, '1d_hists_%.3f_kV_%d_deg' % (cathode_setting, degree_setting), batch_mode=True)
                    f.savefig('./temp_results/2d_hist_%.3f_kV_%d_deg.png' % (cathode_setting, degree_setting))

                    #plt.show()

                #flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].flatten(), dtype=np.float32)
                #flat_s1_log_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
                
                flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_s1'][self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_log']].flatten(), dtype=np.float32)
                flat_s1_log_mc = np.asarray(a_s1_s2_mc[self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_log']].flatten(), dtype=np.float32)


                flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][degree_setting]['a_log_s2_s1'][self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_log']].flatten(), dtype=np.float32)
                flat_s1_log_mc = np.asarray(a_s1_s2_mc[self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][degree_setting]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][degree_setting]['high_bin_log']].flatten(), dtype=np.float32)
                logLikelihoodMatching = c_log_likelihood(flat_s1_log_data, flat_s1_log_mc, len(flat_s1_log_data), int(self.d_coincidence_data_information[cathode_setting][degree_setting]['num_data_pts']), 0.95)
                
                matching_ln_likelihood += logLikelihoodMatching

                #print logLikelihoodMatching, prior_ln_likelihood
                #print 'likelihood calculation time: %f' % (time.time() - start_time_tot_ll)
                #print total_ln_likelihood

        #print 'full function time: %f' % (time.time() - start_time_full)
        total_ln_likelihood = matching_ln_likelihood + prior_ln_likelihood

        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor


        #print total_ln_likelihood

        if np.isnan(total_ln_likelihood):
            return -np.inf
        else:
            return total_ln_likelihood


        
    def wrapper_ln_likelihood_full_matching_multiple_energies_lindhard_model(self, a_parameters, kwargs={}):
        #print a_parameters
        #print '\n\n\n'
        num_scale_parameters = len(self.l_cathode_settings_in_use)*len(self.l_degree_settings_in_use)
        
        d_bkg_probs = {}
        d_scale_pars = {}
        counter = 0
        for cathode_setting in self.l_cathode_settings_in_use:
            d_bkg_probs[cathode_setting] = {}
            d_scale_pars[cathode_setting] = {}
            for degree_setting in self.l_degree_settings_in_use:
                d_bkg_probs[cathode_setting][degree_setting] = a_parameters[-(1+num_scale_parameters*2)+counter]
                d_scale_pars[cathode_setting][degree_setting] = a_parameters[-(1+num_scale_parameters)+counter]
                counter += 1
        
        return self.ln_likelihood_full_matching_multiple_energies_lindhard_model(*(list(a_parameters[:-(1+num_scale_parameters*2)]) + [d_bkg_probs, d_scale_pars, a_parameters[-1]]), **kwargs)


    # -----------------------------------------------
    # -----------------------------------------------
    # Likelihood function for testing models
    # -----------------------------------------------
    # -----------------------------------------------

    # model parameters
    # Thomas-Imel
    # (self, w_value, alpha, gamma, kappa, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, dpe_prob, d_scale_pars, d_gpu_local_info, draw_fit=False)
    
    # pol1
    # (self, w_value, alpha, pol_par0, pol_par1, kappa, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, dpe_prob, d_scale_pars, d_gpu_local_info, draw_fit=False)


    def ln_likelihood_test(self, w_value, alpha, pol_par0, pol_par1, kappa, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, dpe_prob, d_scale_pars, d_gpu_local_info, draw_fit=False):



        # -----------------------------------------------
        # -----------------------------------------------
        # determine prior likelihood and variables
        # -----------------------------------------------
        # -----------------------------------------------

        #start_time_full = time.time()

        prior_ln_likelihood = 0
        matching_ln_likelihood = 0


        current_likelihood, w_value = self.get_w_value_default(w_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_alpha(alpha)
        #prior_ln_likelihood += self.get_prior_log_likelihood_zeta(zeta)
        #prior_ln_likelihood += self.get_prior_log_likelihood_beta(beta)
        prior_ln_likelihood += self.get_prior_log_likelihood_kappa(kappa)

        # Thomas-Imel Model
        #prior_ln_likelihood += self.get_prior_log_likelihood_gamma(gamma)
        #prior_ln_likelihood += self.get_prior_log_likelihood_delta(delta)
        
        # Pol1 Recombination
        # no prior

        # priors of resolutions
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s1)
        prior_ln_likelihood += self.get_prior_log_likelihood_resolution(intrinsic_res_s2)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_probabilities(d_scale_pars)



        # priors of detector variables
        current_likelihood, g1_value = self.get_g1_default(g1_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)
        
        current_likelihood, extraction_efficiency = self.get_extraction_efficiency_default(extraction_efficiency_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, spe_res = self.get_spe_res_default(spe_res_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_value = self.get_gas_gain_default(gas_gain_mean_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, gas_gain_width = self.get_gas_gain_width_default(gas_gain_width_value)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)


        # priors of efficiencies
        current_likelihood, pf_eff_par0, pf_eff_par1 = self.get_pf_eff_default(pf_eff_par0, pf_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_likelihood, s2_eff_par0, s2_eff_par1 = self.get_s2_eff_default(s2_eff_par0, s2_eff_par1)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_likelihood)

        current_ln_likelihood, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2 = self.get_pf_stdev_default(pf_stdev_par0, pf_stdev_par1, pf_stdev_par2)
        prior_ln_likelihood += self.get_prior_log_likelihood_nuissance(current_ln_likelihood)
        
        prior_ln_likelihood += self.get_prior_log_likelihood_dpe_prob(dpe_prob)
        


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

        w_value = np.asarray(w_value, dtype=np.float32)
        alpha = np.asarray(alpha, dtype=np.float32)
        #zeta = np.asarray(zeta, dtype=np.float32)
        #beta = np.asarray(beta, dtype=np.float32)
        kappa = np.asarray(kappa, dtype=np.float32)

        # Thomas-Imel
        #gamma = np.asarray(gamma, dtype=np.float32)
        #delta = np.asarray(delta, dtype=np.float32)
        
        # Pol1
        pol_par0 = np.asarray(pol_par0, dtype=np.float32)
        pol_par1 = np.asarray(pol_par1, dtype=np.float32)

        g1_value = np.asarray(g1_value, dtype=np.float32)
        extraction_efficiency = np.asarray(extraction_efficiency, dtype=np.float32)
        spe_res = np.asarray(spe_res, dtype=np.float32)
        gas_gain_value = np.asarray(gas_gain_value, dtype=np.float32)
        gas_gain_width = np.asarray(gas_gain_width, dtype=np.float32)

        intrinsic_res_s1 = np.asarray(intrinsic_res_s1, dtype=np.float32)
        intrinsic_res_s2 = np.asarray(intrinsic_res_s2, dtype=np.float32)

        #print 'hard coding pf eff'
        #pf_eff_par0 = np.asarray(0.01, dtype=np.float32)
        #pf_eff_par1 = np.asarray(0.01, dtype=np.float32)
        pf_eff_par0 = np.asarray(pf_eff_par0, dtype=np.float32)
        pf_eff_par1 = np.asarray(pf_eff_par1, dtype=np.float32)

        s2_eff_par0 = np.asarray(s2_eff_par0, dtype=np.float32)
        s2_eff_par1 = np.asarray(s2_eff_par1, dtype=np.float32)

        a_pf_stdev = np.asarray([pf_stdev_par0, pf_stdev_par1, pf_stdev_par2], dtype=np.float32)

        dpe_prob = np.asarray(dpe_prob, dtype=np.float32)
        
        # for histogram binning
        num_bins_s1 = np.asarray(self.l_s1_settings[0], dtype=np.int32)
        num_bins_log = np.asarray(self.l_log_settings[0], dtype=np.int32)
        a_hist_2d = np.zeros(self.l_s1_settings[0]*self.l_log_settings[0], dtype=np.float32)
        
        num_loops = np.asarray(self.num_loops, dtype=np.int32)



        for cathode_setting in self.l_cathode_settings_in_use:
            mean_field = np.asarray(self.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)
            
            #start_time_mc = time.time()
        
            # Thomas-Imel
            #tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['d_gpu_energy'][self.l_degree_settings_in_use[0]], drv.In(w_value), drv.In(alpha), drv.In(gamma), drv.In(kappa), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(spe_res), drv.In(intrinsic_res_s1), drv.In(intrinsic_res_s2), drv.In(a_pf_stdev), drv.In(pf_eff_par0), drv.In(pf_eff_par1), drv.In(s2_eff_par0), drv.In(s2_eff_par1), drv.In(dpe_prob), drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d), drv.In(num_loops))

            # Pol1
            tArgs = (d_gpu_local_info['rng_states'], drv.In(num_trials), drv.In(mean_field), d_gpu_local_info['d_gpu_energy'][self.l_degree_settings_in_use[0]], drv.In(w_value), drv.In(alpha), drv.In(pol_par0), drv.In(pol_par1), drv.In(kappa), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(spe_res), drv.In(intrinsic_res_s1), drv.In(intrinsic_res_s2), drv.In(a_pf_stdev), drv.In(pf_eff_par0), drv.In(pf_eff_par1), drv.In(s2_eff_par0), drv.In(s2_eff_par1), drv.In(dpe_prob), drv.In(num_bins_s1), d_gpu_local_info['gpu_bin_edges_s1'], drv.In(num_bins_log), d_gpu_local_info['gpu_bin_edges_log'], drv.InOut(a_hist_2d), drv.In(num_loops))

            d_gpu_local_info['function_to_call'](*tArgs, **self.d_gpu_scale)
            
            #print 'MC time: %f' % (time.time() - start_time_mc)
            #start_time_tot_ll = time.time()

            a_s1_s2_mc = np.reshape(a_hist_2d, (self.l_log_settings[0], self.l_s1_settings[0])).T

            sum_mc = np.sum(a_s1_s2_mc, dtype=np.float32)
            if sum_mc == 0:
                return -np.inf

            #a_s1_s2_mc = np.multiply(a_s1_s2_mc, np.sum(self.a_s1_s2, dtype=np.float32) / sum_mc)

            # if making PDF rather than scaling for rate
            scale_par = d_scale_pars[cathode_setting]
            scale_par *= float(self.num_mc_events*self.num_loops) / sum_mc

            a_s1_s2_mc = np.multiply(a_s1_s2_mc, float(scale_par)*self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['num_data_pts']/float(self.num_mc_events*self.num_loops))

            # likelihood for band
            if draw_fit:

                f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

                s1_s2_data_plot = np.rot90(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'])
                s1_s2_data_plot = np.flipud(s1_s2_data_plot)
                ax1.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_data_plot)

                s1_s2_mc_plot = np.rot90(a_s1_s2_mc)
                s1_s2_mc_plot = np.flipud(s1_s2_mc_plot)
                #print self.l_s1_settings
                #print self.l_log_settings
                #print self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].shape
                #print a_s1_s2_mc.shape
                #print s1_s2_data_plot.shape
                #print s1_s2_mc_plot.shape
                ax2.pcolormesh(self.a_s1_bin_edges, self.a_log_bin_edges, s1_s2_mc_plot)
                #plt.colorbar()


                c1 = Canvas(1400, 400)
                c1.Divide(2)

                h_s1_data = Hist(*self.l_s1_settings, name='hS1_draw_data')
                root_numpy.array2hist(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].sum(axis=1), h_s1_data)

                hS1MC = Hist(*self.l_s1_settings, name='hS1_draw_mc')
                root_numpy.array2hist(a_s1_s2_mc.sum(axis=1), hS1MC)

                s1_scale_factor = h_s1_data.Integral() / float(hS1MC.Integral())

                g_s1_data = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s1_data)
                g_s1_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(hS1MC, scale=1.)

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
                root_numpy.array2hist(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].sum(axis=0), h_s2_data)

                h_s2_mc = Hist(*self.l_log_settings, name='h_s2_draw_mc')
                root_numpy.array2hist(a_s1_s2_mc.sum(axis=0), h_s2_mc)

                s2_scale_factor = h_s2_data.Integral() / float(h_s2_mc.Integral())

                g_s2_data = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s2_data)
                g_s2_mc = neriX_analysis.convert_hist_to_graph_with_poisson_errors(h_s2_mc, scale=1.)

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

                neriX_analysis.save_plot(['temp_results'], c1, '1d_hists_%.3f_kV' % cathode_setting, batch_mode=True)
                f.savefig('./temp_results/2d_hist_%.3f_kV.png' % cathode_setting)

                #plt.show()

            #flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'].flatten(), dtype=np.float32)
            #flat_s1_log_mc = np.asarray(a_s1_s2_mc.flatten(), dtype=np.float32)
            
            flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'][self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
            flat_s1_log_mc = np.asarray(a_s1_s2_mc[self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)


            flat_s1_log_data = np.asarray(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['a_log_s2_s1'][self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
            flat_s1_log_mc = np.asarray(a_s1_s2_mc[self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_s1']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_s1'],self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['low_bin_log']:self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['high_bin_log']].flatten(), dtype=np.float32)
            logLikelihoodMatching = c_log_likelihood(flat_s1_log_data, flat_s1_log_mc, len(flat_s1_log_data), int(self.d_coincidence_data_information[cathode_setting][self.l_degree_settings_in_use[0]]['num_data_pts']), 0.95)
            
            matching_ln_likelihood += logLikelihoodMatching

            #print logLikelihoodMatching, prior_ln_likelihood
            #print 'likelihood calculation time: %f' % (time.time() - start_time_tot_ll)
            #print total_ln_likelihood


        #print 'full function time: %f' % (time.time() - start_time_full)
        total_ln_likelihood = matching_ln_likelihood + prior_ln_likelihood
        #print total_ln_likelihood

        if self.b_suppress_likelihood:
            total_ln_likelihood /= self.ll_suppression_factor



        if np.isnan(total_ln_likelihood):
            return -np.inf
        else:
            return total_ln_likelihood


        
    def wrapper_ln_likelihood_test(self, a_parameters, kwargs={}):
        #print a_parameters
        #print '\n\n\n'
        d_scale_pars = {}
        for i, cathode_setting in enumerate(self.l_cathode_settings_in_use):
            d_scale_pars[cathode_setting] = a_parameters[-(1+len(self.l_cathode_settings_in_use))+i]
        
        return self.ln_likelihood_test(*(list(a_parameters[:-2]) + [d_scale_pars, a_parameters[-1]]), **kwargs)





    def initial_positions_for_ensemble(self, a_free_parameters, num_walkers):
    
        if self.fit_type == 's':
            l_par_names = ['py', 'qy', 'intrinsic_res_s1', 'intrinsic_res_s2', 'g1_value', 'spe_res_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'pf_eff_par0', 'pf_eff_par1', 's2_eff_par0', 's2_eff_par1', 'pf_stdev_par0', 'pf_stdev_par1', 'pf_stdev_par2', 'exciton_to_ion_par0_rv', 'exciton_to_ion_par1_rv', 'exciton_to_ion_par2_rv', 'prob_bkg', 'scale']
        elif self.fit_type == 'm':
            l_par_names = ['a_py', 'a_qy', 'intrinsic_res_s1', 'intrinsic_res_s2', 'g1_value', 'spe_res_rv', 'g2_value', 'gas_gain_rv', 'gas_gain_width_rv', 'pf_eff_par0', 'pf_eff_par1', 's2_eff_par0', 's2_eff_par1', 'pf_stdev_par0', 'pf_stdev_par1', 'pf_stdev_par2', 'exciton_to_ion_par0_rv', 'exciton_to_ion_par1_rv', 'exciton_to_ion_par2_rv']
        elif self.fit_type == 'ml':
            l_prob_bkg_names = []
            l_scale_par_names = []
            for cathode_setting in self.l_cathode_settings_in_use:
                for degree_setting in self.l_degree_settings_in_use:
                    l_prob_bkg_names.append('prob_bkg_%.3f_kV_%d_deg' % (cathode_setting, degree_setting))
                    l_scale_par_names.append('scale_par_%.3f_kV_%d_deg' % (cathode_setting, degree_setting))
        
            l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'intrinsic_res_s1', 'intrinsic_res_s2', 'g1_value', 'spe_res_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'pf_eff_par0', 'pf_eff_par1', 's2_eff_par0', 's2_eff_par1', 'pf_stdev_par0', 'pf_stdev_par1', 'pf_stdev_par2'] + l_prob_bkg_names + l_scale_par_names
        elif self.fit_type == 'mb':
            l_par_names = ['a_py', 'a_qy', 'intrinsic_res_s1', 'intrinsic_res_s2', 'g1_value', 'spe_res_rv', 'g2_value', 'gas_gain_rv', 'gas_gain_width_rv', 'pf_eff_par0', 'pf_eff_par1', 's1_eff_par0', 's1_eff_par1', 's2_eff_par0', 's2_eff_par1', 'pf_stdev_par0', 'pf_stdev_par1', 'pf_stdev_par2', 'exciton_to_ion_par0_rv', 'exciton_to_ion_par1_rv', 'exciton_to_ion_par2_rv']
        elif self.fit_type == 'bl':
            l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'intrinsic_res_s1', 'intrinsic_res_s2', 'g1_value', 'spe_res_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'pf_eff_par0', 'pf_eff_par1', 's2_eff_par0', 's2_eff_par1', 'pf_stdev_par0', 'pf_stdev_par1', 'pf_stdev_par2'] + ['scale_par_%d' for cathode_setting in self.l_cathode_settings_in_use]

        d_variable_arrays = {}
        d_stdevs = {}
        if self.fit_type == 'm' or self.fit_type == 'mb':
            num_yields = 2*len(self.a_spline_energies)
        elif self.fit_type == 's':
            num_yields = 1
        elif self.fit_type == 'ml':
            num_yields = 9


        # position array should be (num_walkers, num_dim)

        """
        g1_value, g2_value = self.l_means_g1_g2
        gas_gain_rv = 0
        gas_gain_width_rv = 0
        spe_res_rv = 0
        pf_eff_par0, pf_eff_par1 = self.l_means_pf_eff_pars
        pf_stdev_par0, pf_stdev_par1, pf_stdev_par2 = self.l_means_pf_stdev_pars
        exciton_to_ion_par0_rv, exciton_to_ion_par1_rv, exciton_to_ion_par2_rv = 0, 0, 0
        """

        count_bkg_pars = 0
        count_scale_pars = 0

        for par_name in l_par_names:
            # handle photon and charge yield initial positions
            if par_name == 'a_py':
                d_variable_arrays[par_name] = np.asarray([np.random.normal(a_free_parameters[i], 0.3*np.asarray(a_free_parameters[i]), size=num_walkers) for i in xrange(num_yields)])

            elif par_name == 'a_qy':
                d_variable_arrays[par_name] = np.asarray([np.random.normal(a_free_parameters[i], 0.3*np.asarray(a_free_parameters[i]), size=num_walkers) for i in xrange(num_yields, 2*num_yields)])
        
            elif par_name == 'py':
                d_variable_arrays[par_name] = np.asarray([np.random.normal(a_free_parameters[i], 0.05*np.asarray(a_free_parameters[i]), size=num_walkers) for i in xrange(num_yields)])

            elif par_name == 'qy':
                d_variable_arrays[par_name] = np.asarray([np.random.normal(a_free_parameters[i], 0.05*np.asarray(a_free_parameters[i]), size=num_walkers) for i in xrange(num_yields, 2*num_yields)])

            elif par_name == 'w_value_rv':
                # special case since w_value is an RV
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[0], 1, size=num_walkers)
            
            elif par_name == 'w_value':
                # special case since w_value is an RV
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[0], 0.05*a_free_parameters[0], size=num_walkers)

            elif par_name == 'alpha':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[1], a_free_parameters[1]*0.3, size=num_walkers)

            elif par_name == 'zeta':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[2], a_free_parameters[2]*0.3, size=num_walkers)

            elif par_name == 'beta':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[3], a_free_parameters[3]*0.3, size=num_walkers)

            elif par_name == 'gamma':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[4], a_free_parameters[4]*0.3, size=num_walkers)

            elif par_name == 'delta':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[5], a_free_parameters[5]*0.3, size=num_walkers)

            elif par_name == 'kappa':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[6], a_free_parameters[6]*0.3, size=num_walkers)

            elif par_name == 'eta':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[7], a_free_parameters[7]*0.3, size=num_walkers)

            elif par_name == 'lambda':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[8], a_free_parameters[8]*0.3, size=num_walkers)

            elif par_name == 'intrinsic_res_s1':
                if self.fit_type == 's':
                    d_variable_arrays[par_name] = np.random.normal(a_free_parameters[2*num_yields+0], .04, size=num_walkers)
                elif self.fit_type == 'bl':
                    d_variable_arrays[par_name] = np.random.normal(a_free_parameters[7], .04, size=num_walkers)

            elif par_name == 'intrinsic_res_s2':
                if self.fit_type == 's':
                    d_variable_arrays[par_name] = np.random.normal(a_free_parameters[2*num_yields+1], .04, size=num_walkers)
                elif self.fit_type == 'bl':
                    d_variable_arrays[par_name] = np.random.normal(a_free_parameters[8], .04, size=num_walkers)


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


            # catch both parameters of s1 efficiency prior
            elif par_name == 'pf_eff_par0':
                d_variable_arrays['pf_eff_par0'], d_variable_arrays['pf_eff_par1'] = np.random.multivariate_normal(self.l_means_pf_eff_pars, self.l_cov_matrix_pf_eff_pars, size=num_walkers).T


            elif par_name == 's1_eff_par0':
                d_variable_arrays['s1_eff_par0'], d_variable_arrays['s1_eff_par1'] = np.asarray([np.random.normal(a_free_parameters[i], 0.3*np.asarray(a_free_parameters[i]), size=num_walkers) for i in xrange(2*num_yields+2, 2*num_yields+4)])
                #d_variable_arrays[par_name] = np.random.normal(a_free_parameters[2*num_yields+2:2*num_yields+4], 0.3*np.asarray(a_free_parameters[2*num_yields+2:2*num_yields+4]), size=num_walkers)


            elif par_name == 's2_eff_par0':
                d_variable_arrays['s2_eff_par0'], d_variable_arrays['s2_eff_par1'] = np.random.multivariate_normal(self.l_means_s2_eff_pars, self.l_cov_matrix_s2_eff_pars, size=num_walkers).T


            elif par_name == 'pf_stdev_par0':
                d_variable_arrays['pf_stdev_par0'], d_variable_arrays['pf_stdev_par1'], d_variable_arrays['pf_stdev_par2'] = np.random.multivariate_normal(self.l_means_pf_stdev_pars, self.l_cov_matrix_pf_stdev_pars, size=num_walkers).T
                
            
            elif par_name[:8] == 'prob_bkg':
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[-2*(len(self.l_cathode_settings_in_use)*len(self.l_degree_settings_in_use)) + count_bkg_pars], .02, size=num_walkers)
                count_bkg_pars += 1
                

            elif par_name[:5] == 'scale':
                # need to track scale parameters
                # always the last indices so move back by
                # number of degree settings multiplied by
                # number of cathode settings
                d_variable_arrays[par_name] = np.random.normal(a_free_parameters[-(len(self.l_cathode_settings_in_use)*len(self.l_degree_settings_in_use)) + count_scale_pars], .02, size=num_walkers)
                count_scale_pars += 1



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
        s_cathode_voltages = ''
        for cathode_setting in self.l_cathode_settings_in_use:
            s_cathode_voltages += '%s,' % (cathode_setting)
        s_cathode_voltages = s_cathode_voltages[:-1]

        # get string for degree settings in use
        s_degree_settings = ''
        for degree_setting in self.l_degree_settings_in_use:
            s_degree_settings += '%s,' % (degree_setting)
        s_degree_settings = s_degree_settings[:-1]

        # before emcee, setup save locations
        dir_specifier_name = '%s_kV_%s_deg' % (s_cathode_voltages, s_degree_settings)
        self.results_directory_name = xe1t_config.results_directory_name
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

            if self.fit_type == 's' and self.num_dimensions == 21:
                degree_setting = self.l_degree_settings_in_use[0]
                cathode_setting = self.l_cathode_settings_in_use[0]
                
                if degree_setting == 2300:
                    if cathode_setting == 1.054:
                        a_free_parameter_guesses = [7.24, 6.10, 0.07, 0.31, 0.25, 0.97]
                        # [ 7.24279474,  6.10423574,  0.06494089,  0.30873822, 0.25022568,  0.99936306] # 2300
                    elif cathode_setting == 0.345:
                        a_free_parameter_guesses = [7.04, 5.68, 0.23, 0.12, 0.29, 0.97]
                        # [ 7.04498957,  5.6815345,  0.23072485,  0.11705023, 0.28815432,  0.97119736] # 2300
                        
                    elif cathode_setting == 2.356:
                        a_free_parameter_guesses = [7.79, 6.27, 0.19, 0.18, 0.16, 0.97]
                        # [ 8.21997358,  6.06919105,  0.06841404 ,  0.46676469,  0.14950445, 0.98737901] # 2300
                        
                elif degree_setting == 3000:
                    if cathode_setting == 1.054:
                        a_free_parameter_guesses = [7.56, 5.76, 0.07, 0.26, 0.38, 0.97]
                        # [ 7.56098549,  5.76996929,  0.07485624 ,  0.25600894, 0.38586383,  0.98883613] # 3000
                    
                    elif cathode_setting == 0.345:
                        a_free_parameter_guesses = [8.58, 6.40, 0.06, 0.32, 0.28, 0.97]
                        # [ 8.58012424,  6.4082583,  0.059447,  0.32507151, 0.28044492,  0.97627979] # 3500
                        
                    elif cathode_setting == 2.356:
                        a_free_parameter_guesses = [8.20, 7.40, 0.07, 0.22, 0.28, 0.92]
                        # [ 8.20653994,  7.45641684,  0.06750863 ,  0.21867223,  0.28004732, 0.92374844] # 3000
                        
                elif degree_setting == 3500:
                    if cathode_setting == 1.054:
                        a_free_parameter_guesses = [7.71, 6.03, 0.04, 0.06, 0.32, 0.92]
                        # [ 7.71422274,  6.03296573,  0.03810478,  0.05889366, 0.31801688,  0.91945319] # 3500
                        
                    elif cathode_setting == 0.345:
                        a_free_parameter_guesses = [7.76, 6.17, 0.21, 0.26, 0.05, 0.93]
                        # [ 7.76790283,  6.17368043,  0.21353381,  0.26405303, 0.05305485,  0.92298929] # 3500
                        
                    elif cathode_setting == 2.356:
                        a_free_parameter_guesses = [7.79, 6.27, 0.19, 0.18, 0.16, 0.97]
                        # [ 7.78806321,  6.27176934,  0.19965031 ,  0.17596064,  0.164257, 0.99944881] # 3500
                        
                elif degree_setting == 4500:
                    if cathode_setting == 1.054:
                        a_free_parameter_guesses = [7.80, 5.83, 0.18, 0.23, 0.05, 0.97]
                        # [ 7.78910186,  5.832333,  0.17891269,  0.22816425, 0.02462351,  0.99851682] # 4500
                    
                    elif cathode_setting == 0.345:
                        a_free_parameter_guesses = [8.82, 5.96, 0.19, 0.04, 0.48, 0.97]
                        # [ 8.82078054,  5.96911495,  0.1908735 ,  0.02383425,  0.48874606, 0.99311512] # 4500
                        
                    elif cathode_setting == 2.356:
                        a_free_parameter_guesses = [7.65, 5.96, 0.22, 0.05, 0.32, 0.97]
                        # [ 7.65321071,  5.96063235,  0.2200952 ,  0.0500525,  0.31784079, 0.98759496] # 4500
                    
                elif degree_setting == 5300:
                    if cathode_setting == 1.054:
                        a_free_parameter_guesses = [9.78, 6.12, 0.27, 0.04, 0.17, 0.97]
                        # [ 9.7874405,  6.12108864,  0.27462468,  0.01749649, 0.16826173,  0.99297773] # 5300
                        
                    elif cathode_setting == 0.345:
                        a_free_parameter_guesses = [9.82, 6.18, 0.21, 0.02, 0.13, 0.96]
                        # [ 9.82589491,  6.18360509,  0.20806021,  0.01750014,  0.13195983,  0.96117175] # 5300
                        
                    elif cathode_setting == 2.356:
                        a_free_parameter_guesses = [9.82, 6.18, 0.21, 0.02, 0.13, 0.96]
                        # [ 9.760886,  5.96287371,  0.2028719,  0.03330193,  0.22689327,  0.99626046] # 5300


                else:
                    a_free_parameter_guesses = [8., 6., 0.25, 0.25, 0.95]

            elif self.fit_type == 'bl':
                a_free_parameter_guesses = [13.7, 1.07, 0.042, 744, 0.012, 0.08, 0.11, 0.086, 0.18, 0.84, 0.84, 0.87]
                
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
        if self.fit_type == 's' or self.fit_type == 'bl':
            num_dim = self.num_dimensions
        elif self.fit_type == 'm':
            num_dim = self.num_dimensions - 2 + 2*len(self.a_spline_energies)
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
            if self.fit_type == 's':
                a_guesses = list(a_guesses[:-2]) + [test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2], 0, 0, 0] + list(a_guesses[-2:])
                return -self.gpu_pool.map(self.wrapper_ln_likelihood_full_matching_single_energy, [a_guesses])[0]
            elif self.fit_type == 'ml':
                a_guesses = [13.7] + list(a_guesses[:8]) + [test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2]] + list(a_guesses[-2*(len(self.l_cathode_settings_in_use)*len(self.l_degree_settings_in_use))-1:])
                return -self.gpu_pool.map(self.wrapper_ln_likelihood_full_matching_multiple_energies_lindhard_model, [a_guesses])[0]
            elif self.fit_type == 'bl':
                a_guesses = [13.7] + list(a_guesses[:8]) + [test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2]] + list(a_guesses[-3:])
                return -self.gpu_pool.map(self.wrapper_ln_likelihood_full_matching_band, [a_guesses])[0]
                
                
            elif self.fit_type == 'test':
                # Thomas-Imel
                #num_pars_to_keep = 5
                
                # Pol1
                num_pars_to_keep = 6
            
                a_guesses = [13.7] + list(a_guesses[:num_pars_to_keep]) + [test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2]] + list(a_guesses[-2:])
                return -self.gpu_pool.map(self.wrapper_ln_likelihood_test, [a_guesses])[0]
            
        print '\n\nStarting differential evolution minimizer...\n\n'
        result = optimize.differential_evolution(neg_log_likelihood_diff_ev, a_bounds, disp=True, maxiter=maxiter, tol=tol, popsize=popsize, polish=polish)
        print result



    def suppress_likelihood(self, iterations=200, a_free_par_guesses=None):
    
        # reset variables in case this is not the first time run
        self.b_suppress_likelihood = False
        self.ll_suppression_factor = 1.

        if self.fit_type == 's' and a_free_par_guesses == None:
        
            degree_setting = self.l_degree_settings_in_use[0]
            cathode_setting = self.l_cathode_settings_in_use[0]
            
            
            if degree_setting == 2300 and cathode_setting == 1.054:
                l_free_pars = [7.24, 6.10, 0.07, 0.31, 0.25, 0.99]
            elif degree_setting == 2300 and cathode_setting == 0.345:
                l_free_pars = [7.04, 5.68, 0.23, 0.12, 0.29, 0.97]
            elif degree_setting == 2300 and cathode_setting == 2.356:
                l_free_pars = [7.79, 6.27, 0.19, 0.18, 0.16, 0.97]
            
            elif degree_setting == 3000 and cathode_setting == 1.054:
                l_free_pars = [7.56, 5.76, 0.07, 0.26, 0.38, 0.99]
            elif degree_setting == 3000 and cathode_setting == 0.345:
                l_free_pars = [8.58, 6.40, 0.06, 0.32, 0.28, 0.97]
            elif degree_setting == 3000 and cathode_setting == 2.356:
                l_free_pars = [8.20, 7.40, 0.07, 0.22, 0.28, 0.92]
            
            elif degree_setting == 3500 and cathode_setting == 1.054:
                l_free_pars = [7.71, 6.03, 0.04, 0.06, 0.32, 0.92]
            elif degree_setting == 3500 and cathode_setting == 0.345:
                l_free_pars = [7.76, 6.17, 0.21, 0.26, 0.05, 0.93]
            elif degree_setting == 3500 and cathode_setting == 2.356:
                l_free_pars = [7.79, 6.27, 0.19, 0.18, 0.16, 0.97]
            
            elif degree_setting == 4500 and cathode_setting == 1.054:
                l_free_pars = [7.80, 5.83, 0.18, 0.23, 0.024, 0.99]
            elif degree_setting == 4500 and cathode_setting == 0.345:
                l_free_pars = [8.82, 5.96, 0.19, 0.04, 0.48, 0.97]
            elif degree_setting == 4500 and cathode_setting == 2.356:
                l_free_pars = [7.65, 5.96, 0.22, 0.05, 0.32, 0.97]
            
            elif degree_setting == 5300 and cathode_setting == 1.054:
                l_free_pars = [9.78, 6.12, 0.27, 0.04, 0.17, 0.97]
            elif degree_setting == 5300 and cathode_setting == 0.345:
                l_free_pars = [9.82, 6.18, 0.21, 0.02, 0.13, 0.96]
            elif degree_setting == 5300 and cathode_setting == 2.356:
                l_free_pars = [9.76, 5.96, 0.20, 0.03, 0.22, 0.99]
            
            else:
                print 'Not set up for %d at %.3f yet!' % (degree_setting, cathode_setting)
                sys.exit()
            
        
        
            a_free_par_guesses = l_free_pars[:4] + [self.g1_value, self.spe_res_value, self.extraction_efficiency_value, self.gas_gain_value, self.gas_gain_width, self.l_means_pf_eff_pars[0], self.l_means_pf_eff_pars[1], self.l_means_s2_eff_pars[0], self.l_means_s2_eff_pars[1], self.l_means_pf_stdev_pars[0], self.l_means_pf_stdev_pars[1], self.l_means_pf_stdev_pars[2], 0, 0, 0] + l_free_pars[-2:]
        
        
        elif self.fit_type == 'bl' and a_free_par_guesses == None:
            #array([  1.07397396e+00,   4.21667498e-02,   7.44142179e+02,
         #1.19552604e-02,   7.95398343e-02,   1.09040459e-01,
         #8.55429713e-02,   1.80184491e-01,   8.57737930e-01,
         #8.17652356e-01,   8.71049657e-01])
            a_free_par_guesses = [13.7] + [1.07, 0.042, 744, 0.012, 0.08, 0.11] + [0.086, 0.18] + [self.g1_value, self.spe_res_value, self.extraction_efficiency_value, self.gas_gain_value, self.gas_gain_width, self.l_means_pf_eff_pars[0], self.l_means_pf_eff_pars[1], self.l_means_s2_eff_pars[0], self.l_means_s2_eff_pars[1], self.l_means_pf_stdev_pars[0], self.l_means_pf_stdev_pars[1], self.l_means_pf_stdev_pars[2]] + [0.86, 0.82, 0.87]

        elif self.fit_type == 'ml' and a_free_par_guesses == None:
            #[  1.42475072e+00,   8.07800345e-02,   7.94421935e+02,
            #     7.43919929e-03,   3.39877754e-02,   2.01769226e-01,
            #     1.23666318e-01,   5.62583665e-02,
            
            #     5.67468989e-01,
            #     5.92734844e-01,   5.82802675e-01,   4.78844646e-01,
            #     5.99885545e-01,   6.26031425e-01,   6.56838708e-01,
            #     5.16040459e-01,   8.84988711e-01,   9.13862653e-01,
            #     5.22622912e-01,   5.12298521e-01,   4.04311312e-01,
            #     6.48159435e-01,   4.26403364e-01,   6.47203218e-01,
            #     9.81905058e-01,   8.74409918e-01,   9.59801804e-01,
            #     9.12234782e-01,   9.80097693e-01,   7.06467727e-01,
            #     8.99561951e-01,   9.97011005e-01,   9.19141442e-01,
            #     9.69757534e-01,   6.81001034e-01,   9.26606863e-01,
            #     9.40575747e-01,   9.44464535e-01]
            a_free_par_guesses = [13.7] + [1.79, 0.105, 637, 0.012, 0.14, 0.22] + [0.05, 0.05] + [self.g1_value, self.spe_res_value, self.extraction_efficiency_value, self.gas_gain_value, self.gas_gain_width, self.l_means_pf_eff_pars[0], self.l_means_pf_eff_pars[1], self.l_means_s2_eff_pars[0], self.l_means_s2_eff_pars[1], self.l_means_pf_stdev_pars[0], self.l_means_pf_stdev_pars[1], self.l_means_pf_stdev_pars[2]] + [0.86, 0.82, 0.87]
            
            
            
        elif self.fit_type == 'test' and a_free_par_guesses == None:
            # Thomas-Imel only!
            # 487.6
            # [ 0.89733072,  0.00678207,  0.09370585,  0.01598999,  0.16250782, 0.17604912,  0.8097433 ]
            #a_free_par_guesses = [13.7, 0.89733072, 0.00678207, 0.09370585, 0.01598999, 0.16250782, test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2], 0.17604912, 0.8097433]
        
            # Pol1 only!
            #
            #
            a_free_par_guesses = [13.7, 0.89733072, 0.00678207, 0.09370585, 0.01598999, 0.16250782, test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2], 0.17604912, 0.8097433]




        
        #print a_free_par_guesses
        l_parameters = [a_free_par_guesses for i in xrange(iterations)]
        l_log_likelihoods = self.gpu_pool.map(self.ln_likelihood_function_wrapper, l_parameters)
        #print l_log_likelihoods

        std_ll = np.std(l_log_likelihoods)

        print 'Mean for %.3e MC iterations is %f' % (self.num_mc_events, np.mean(l_log_likelihoods))
        print 'Standard deviation for %.3e MC iterations is %f' % (self.num_mc_events, std_ll)
        print 'Will scale LL such that stdev is 0.1'

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

    # d_coincidence_data should be in the following form:
    # d_coincidence_data['degree_settings'] = [2300, 3000, ...]
    # d_coincidence_data['cathode_settings'] = [0.345, 1.054, ...]
    # d_coincidence_data['cathode_settings'] = [4.5]

    """
    d_coincidence_data = {}
    d_coincidence_data['degree_settings'] = [4500]
    d_coincidence_data['cathode_settings'] = [0.345]

    test = fit_nr('s', d_coincidence_data, num_mc_events=2.5e6, num_gpus=3, num_loops=4)
    test.suppress_likelihood()
    
    #ln_likelihood_full_matching_single_energy(self, py, qy, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, exciton_to_ion_par0_rv, exciton_to_ion_par1_rv, exciton_to_ion_par2_rv, d_gpu_local_info, draw_fit=False):
    
    # 2300, 1054: [7.24, 6.10, 0.07, 0.31, 0.25, 0.99]
    # 3000, 1054: [7.95, 5.84, 0.05, 0.34, 0.32, 0.99]
    # 3500, 1054: [7.71, 6.03, 0.04, 0.06, 0.32, 0.92]
    # 4500, 1054: [7.80, 5.83, 0.18, 0.23, 0.024, 0.99]
    # 5300, 1054: [9.69, 6.04, 0.23, 0.05, 0.17, 0.98]
    #l_test_parameters_single_energy = [9.69, 6.04, 0.23, 0.05, test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2], 0, 0, 0, 0.17, 0.98]
    #test.gpu_pool.map(test.wrapper_ln_likelihood_full_matching_single_energy, [l_test_parameters_single_energy])

    #a_free_par_bounds = [(3., 15.), (3.5, 15.), (0.01, 0.5), (0.01, 0.5), (0., 1.), (0, 1.2)]
    #test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=50, popsize=15, tol=0.05)
    
    test.run_mcmc(num_steps=500, num_walkers=256)
    """
    
    
    # band fitting
    """
    d_coincidence_data = {}
    d_coincidence_data['degree_settings'] = [-4]
    d_coincidence_data['cathode_settings'] = [0.345, 1.054, 2.356]
    
    test = fit_nr('bl', d_coincidence_data, num_mc_events=1e4, num_gpus=1, num_loops=4)
    #test.suppress_likelihood()
    
    # ln_likelihood_full_matching_band(self, w_value, alpha, zeta, beta, gamma, delta, kappa, intrinsic_res_s1, intrinsic_res_s2, g1_value, spe_res_value, extraction_efficiency_value, gas_gain_mean_value, gas_gain_width_value, pf_eff_par0, pf_eff_par1, s2_eff_par0, s2_eff_par1, pf_stdev_par0, pf_stdev_par1, pf_stdev_par2, d_scale_par, d_gpu_local_info, draw_fit=False)
    
    #array([  1.07397396e+00,   4.21667498e-02,   7.44142179e+02,
         #1.19552604e-02,   7.95398343e-02,   1.09040459e-01,
         #8.55429713e-02,   1.80184491e-01,   8.57737930e-01,
         #8.17652356e-01,   8.71049657e-01])
    #l_test_parameters_single_energy = [13.7, 0.96, 0.0151, 3000, 0.008, 0.015, 0.145, 0.132, 0.152, test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2], 0.884, 0.871, 0.903]
    l_test_parameters_single_energy = [13.7, 0.96, 0.0151, 3000, 0.008, 0.015, 0.145, 0.132, 0.152, 0.1165, 0.59, 0.89, 21.2, 8.0, 3.0, 0.74, 6.51, 284., 0.015, 0.53, 4.3, 0.884, 0.871, 0.903]
    test.gpu_pool.map(test.wrapper_ln_likelihood_full_matching_band, [l_test_parameters_single_energy])
    

    #a_free_par_bounds = [(0., 5.), (0., 0.15), (0., 800), (0., 0.03), (0., 0.15), (0, 0.4), (0, 0.5), (0, 0.5), (0, 1), (0, 1), (0, 1)]
    #test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=50, popsize=15, tol=0.05)
    
    #test.run_mcmc(num_steps=50, num_walkers=256)
    """
    
    """
    d_coincidence_data = {}
    d_coincidence_data['degree_settings'] = [2300, 3000, 3500, 4500, 5300]
    d_coincidence_data['cathode_settings'] = [0.345, 1.054, 2.356]
    
    test = fit_nr('ml', d_coincidence_data, num_mc_events=5e5, num_gpus=3, num_loops=2)
    #test.suppress_likelihood()
    
    
    #l_test_parameters_single_energy = [13.7, 0.96, 0.0146, 300, 0.008, 0.016, 0.143, 0.132, 0.152, test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2], 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.848, 0.871, 0.903, 0.848, 0.871, 0.903]
    #test.gpu_pool.map(test.wrapper_ln_likelihood_full_matching_multiple_energies_lindhard_model, [l_test_parameters_single_energy])
    

    #[  1.79459409e+00,   1.15749171e-01,   7.78241578e+02,
    #     8.25944316e-03,   8.36785794e-02,   2.08119157e-01,
    #     6.62843567e-03,   6.10040587e-02,   2.11731168e-01,
    #     2.96062936e-01,   7.00887763e-01,   8.37754642e-01,
    #     7.97519318e-01,   5.31585548e-01,   7.79560892e-01,
    #     8.46711475e-01,   4.94155039e-01,   6.63799457e-01,
    #     7.14485676e-01,   6.98872503e-01,   5.98270822e-01,
    #     4.51235938e-01,   5.48717372e-01,   3.29102108e-01,
    #     6.29271392e-01,   8.14208496e-01,   8.29873667e-01,
    #     8.95159524e-01,   9.72899255e-01,   8.06508236e-01,
    #     9.12591984e-01,   7.95359826e-01,   8.79512208e-01,
    #     9.64076691e-01,   9.61312253e-01,   7.21248650e-01,
    #     8.90707315e-01,   9.24789908e-01,   9.33714639e-01]

    a_free_par_bounds = [(0., 5.), (0., 0.15), (0., 800), (0., 0.03), (0., 0.15), (0, 0.4), (0, 0.5), (0, 0.5), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=150, popsize=5, tol=0.05)
    
    #test.run_mcmc(num_steps=50, num_walkers=256)
    """
   
   
    d_coincidence_data = {}
    d_coincidence_data['degree_settings'] = [-4]
    d_coincidence_data['cathode_settings'] = [1.054]
    
    test = fit_nr('test', d_coincidence_data, num_mc_events=5e4, num_gpus=3, num_loops=4)
    
    #test.suppress_likelihood()
    
    # Thomas-Imel
    # [ 0.90315826,  0.00681885,  0.09396569,  0.06339429,  0.16932282, 0.17990872,  0.80983547]
    #l_test_parameters = [13.7, 0.90315826, 0.00681885, 0.09396569, 0.06339429, 0.16932282, test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2], 0.17990872, 0.80983547]
    #test.gpu_pool.map(test.wrapper_ln_likelihood_test, [l_test_parameters])
    
    # Pol1 Recombination
    # [ 0.90315826,  0.00681885,  0.09396569,  0.06339429,  0.16932282, 0.17990872,  0.80983547]
    #l_test_parameters = [13.7, 0.90315826, 0.00681885, 4534534, 0.09396569, 0.06339429, 0.16932282, test.g1_value, test.spe_res_value, test.extraction_efficiency_value, test.gas_gain_value, test.gas_gain_width, test.l_means_pf_eff_pars[0], test.l_means_pf_eff_pars[1], test.l_means_s2_eff_pars[0], test.l_means_s2_eff_pars[1], test.l_means_pf_stdev_pars[0], test.l_means_pf_stdev_pars[1], test.l_means_pf_stdev_pars[2], 0.17990872, 0.80983547]
    #test.gpu_pool.map(test.wrapper_ln_likelihood_test, [l_test_parameters])
    
    
    
    # Thomas-Imel
    #a_free_par_bounds = [(0., 5.), (0., 0.03), (0, 0.4), (0, 0.5), (0, 0.5), (0.17, 0.24), (0.7, 1)]
    #test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=150, popsize=15, tol=0.01, polish=True)
    
    # Pol1
    a_free_par_bounds = [(0., 5.), (0., 1.), (0, 3.), (0, 0.5), (0, 0.5), (0.17, 0.24), (0.7, 1)]
    test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=150, popsize=15, tol=0.01, polish=True)
    
    # Pol2
    #a_free_par_bounds = [(0., 5.), (0., 1.), (0, 3.), (0, 1.), (0, 0.5), (0, 0.5), (0.17, 0.24), (0.7, 1)]
    #test.differential_evolution_minimizer_free_pars(a_free_par_bounds, maxiter=150, popsize=15, tol=0.01, polish=True)
    
    
    test.close_workers()



