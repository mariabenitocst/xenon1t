#!/usr/bin/python
import ROOT as root

import sys
import os, re
import random

sys.path.insert(0, '..')
import config_xe1t

import pandas as pd

import numpy as np

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

import tqdm

#from rootpy import stl
#from rootpy.io import File
#from rootpy.tree import Tree, TreeModel, TreeChain
#from rootpy.plotting import Hist, Hist2D, Canvas, Graph, func


from math import floor

from sklearn import neighbors
from sklearn import grid_search
from sklearn import preprocessing

import scipy.optimize as op

import cPickle as pickle


# -----------------------------------------
#  Setup and initialization
# -----------------------------------------

l_grid_parameters = ['gamma', 'alpha', 'lambda', 'extraction_efficiency']
num_mc_events = int(2e6)

fiducial_volume_mass = 1000. # kg
print '\n\nGiving rate assuming %0.f kg FV\n\n\n' % (fiducial_volume_mass)

lax_version = 'lax_0.9.2'
print '\n\nCurrently using %s for fitting\n\n\n' % (lax_version)


l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1', 'extraction_efficiency', 'gas_gain_mean', 'gas_gain_width', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par'] + ['prob_bkg', 'scale_par']
num_dim = len(l_par_names)

l_sigma_levels = [-1, 0, 1]
assert 0 in l_sigma_levels

d_sigma_to_percentile = {-10:0.00001, -2:2.5, -1:16, 0:50, 1:84, 2:97.5, 10:99.999999}

d_parameter_to_index = {'kappa':6,
                        'alpha':1,
                        'gamma':4,
                        'w_value':0,
                        'zeta':2,
                        'delta':5,
                        'eta':7,
                        'lambda':8,
                        'g1':9,
                        'extraction_efficiency':10}

# num_steps_to_include is how large of sampler you keep
num_steps_to_include = 1000
num_steps_to_pull_from = 1000


if(len(sys.argv) != 3):
	print 'Usage is python make_pl_histograms_wimps.py <num walkers> <wimp mass>'
	sys.exit(1)

num_walkers = int(sys.argv[1])
wimp_mass = int(sys.argv[2]) # GeV
s_wimp_mass = '%04d' % (wimp_mass)


dir_specifier_name = 'run_0_band'


l_degree_settings_in_use = [-4]
s_degree_settings = ''
for degree_setting in l_degree_settings_in_use:
    s_degree_settings += '%s,' % (degree_setting)
s_degree_settings = s_degree_settings[:-1]


l_cathode_settings_in_use = [12.]
s_cathode_settings = ''
for cathode_setting in l_cathode_settings_in_use:
    s_cathode_settings += '%.3f,' % (cathode_setting)
s_cathode_settings = s_cathode_settings[:-1]

name_of_results_directory = config_xe1t.results_directory_name
l_plots = ['plots', dir_specifier_name, '%s_kV_%s_deg' % (s_cathode_settings, s_degree_settings), 'parameter_variations']

#print '\n\nDEBUG: Using old sampler temporarily\n\n\n'
s_path_to_file = './%s/%s/%s_kV_%s_deg/sampler_dictionary.p' % (name_of_results_directory, dir_specifier_name, s_cathode_settings, s_degree_settings)


if os.path.exists(s_path_to_file):
    dSampler = pickle.load(open(s_path_to_file, 'r'))
    l_chains = []
    for sampler in dSampler[num_walkers]:
        l_chains.append(sampler['_chain'])

    a_full_sampler = np.concatenate(l_chains, axis=1)

    print 'Successfully loaded sampler!'
else:
    print s_path_to_file
    print 'Could not find file!'
    sys.exit()

total_length_sampler = len(a_full_sampler)

# get block and grid size
d_gpu_scale = {}
block_dim = 1024
d_gpu_scale['block'] = (block_dim,1,1)
num_blocks = floor(num_mc_events / float(block_dim))
d_gpu_scale['grid'] = (int(num_blocks), 1)
num_mc_events = int(num_blocks*block_dim)


# get significant value sets to loop over later
a_samples = a_full_sampler[:, -num_steps_to_include:, :].reshape((-1, a_full_sampler.shape[2]))




# -----------------------------------------
#  Find maxima for each parameter
#  individually
# -----------------------------------------

d_all_parameter_info = {}



a_partial_sampler = np.zeros((num_steps_to_include*num_walkers, num_dim))
for i in tqdm.tqdm(xrange(num_steps_to_include*num_walkers)):
    a_partial_sampler[i, :] = a_samples[(-(np.random.randint(1, num_steps_to_pull_from*num_walkers) % total_length_sampler)), :]

for i, par_name in enumerate(l_par_names):
    d_all_parameter_info[par_name] = {}
    d_all_parameter_info[par_name]['best_fit'] = np.percentile(a_partial_sampler[:, i], [50])



# -----------------------------------------
#  Find maxima for each parameter
#  individually
# -----------------------------------------

d_grid_parameter_percentiles = {}

for par_name in l_grid_parameters:
    d_grid_parameter_percentiles[par_name] = {}
    
    for sigma_level in l_sigma_levels:
        d_grid_parameter_percentiles[par_name][sigma_level] = np.percentile(a_partial_sampler[:, d_parameter_to_index[par_name]], [d_sigma_to_percentile[sigma_level]])






# -----------------------------------------
#  Load GPU arrays onto GPU
# -----------------------------------------


l_s1_settings = config_xe1t.l_s1_settings
l_log_settings = config_xe1t.l_log_settings
l_s2_settings = config_xe1t.l_s2_settings

figure_sizes = (11, 13)





d_plotting_information = {}

# need to prepare GPU for MC simulations
import cuda_full_observables_production
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray

drv.init()
dev = drv.Device(0)
ctx = dev.make_context()
print 'Device Name: %s\n' % (dev.name())


# only for electron lifetime
d_mc_energy = pickle.load(open('%swimp_mc.p' % config_xe1t.path_to_fit_inputs, 'r'))
d_er_band = pickle.load(open('%ser_band.p' % config_xe1t.path_to_fit_inputs, 'r'))

random.seed()

df_wimp_hist = pd.read_csv('./wimp_spectra/wimp_%dgev_1e-45cm2.csv' % (wimp_mass))
d_wimp_hist = {}
d_wimp_hist['energy_kev'] = np.asarray(df_wimp_hist['kev'], dtype=np.float32)
d_wimp_hist['rate_per_kev_kg_day'] = np.asarray(df_wimp_hist['events_perkg_perkev_perday'], dtype=np.float32)

# ['energy_kev', 'rate_per_kev_kg_day']
    
# -----------------------------------------
#  Energy
# -----------------------------------------

bin_width = d_wimp_hist['energy_kev'][1] - d_wimp_hist['energy_kev'][0]

cdf = np.cumsum(d_wimp_hist['rate_per_kev_kg_day'])
integrated_rate = cdf[-1]*bin_width # at 1E-45 cm^2
cdf = cdf / cdf[-1]



values = np.random.rand(num_mc_events)
value_bins = np.searchsorted(cdf, values)
random_from_cdf = d_wimp_hist['energy_kev'][value_bins]

a_mc_energy = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_random_num = np.random.random()*bin_width + random_from_cdf[i]
    
    # need to cover edge case of zero bin
    if current_random_num < 0:
        current_random_num = -current_random_num
    
    a_mc_energy[i] = current_random_num

#plt.hist(a_mc_energy, bins=100)
#plt.show()


# -----------------------------------------
#  Z
# -----------------------------------------


a_mc_z = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    
    a_mc_z[i] = np.random.uniform(config_xe1t.min_z_cylinder, config_xe1t.max_z_cylinder)





# -----------------------------------------
#  X, Y
# -----------------------------------------


a_mc_x = np.zeros(num_mc_events, dtype=np.float32)
a_mc_y = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_r = np.random.uniform(0, config_xe1t.max_r_cylinder**2)**0.5
    current_phi = np.random.uniform(0, 2*np.pi)

    current_random_num_x = current_r * np.cos(current_phi)
    current_random_num_y = current_r * np.sin(current_phi)
    
    
    a_mc_x[i] = current_random_num_x
    a_mc_y[i] = current_random_num_y



# -----------------------------------------
#  Electron Lifetime
# -----------------------------------------

bin_width = d_mc_energy['a_el_bins'][1] - d_mc_energy['a_el_bins'][0]

cdf = np.cumsum(d_mc_energy['a_el_hist'])
cdf = cdf / cdf[-1]
values = np.random.rand(num_mc_events)
value_bins = np.searchsorted(cdf, values)
random_from_cdf = d_mc_energy['a_el_bins'][value_bins]

a_e_survival_prob = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_random_num = np.random.random()*bin_width + random_from_cdf[i]
    
    # current random number is lifetime whch we need to convert
    # draw from z array to make sure they are connected!
    a_e_survival_prob[i] = np.exp(-(config_xe1t.z_gate - a_mc_z[i]) / config_xe1t.e_drift_velocity / current_random_num)
    #print a_mc_z[i], a_e_survival_prob[i], current_random_num







# ------------------------------------------------
# ------------------------------------------------
# Pull required arrays for correction maps
# ------------------------------------------------
# ------------------------------------------------


d_corrections = pickle.load(open('%ssignal_correction_maps.p' % config_xe1t.path_to_fit_inputs, 'r'))
    
bin_edges_r2 = np.asarray(d_corrections['s1']['r2_bin_edges'], dtype=np.float32)
bin_edges_z = np.asarray(d_corrections['s1']['z_bin_edges'], dtype=np.float32)
s1_correction_map = np.asarray(d_corrections['s1']['map'], dtype=np.float32).T
s1_correction_map = s1_correction_map.flatten()

bin_edges_x = np.asarray(d_corrections['s2']['x_bin_edges'], dtype=np.float32)
bin_edges_y = np.asarray(d_corrections['s2']['y_bin_edges'], dtype=np.float32)
s2_correction_map = np.asarray(d_corrections['s2']['map'], dtype=np.float32).T
s2_correction_map = s2_correction_map.flatten()


# -----------------------------------------
#  Get array of ER bands S1 and S2
# -----------------------------------------


bin_width_s1 = d_er_band['er_band_s1_edges'][1] - d_er_band['er_band_s1_edges'][0]
bin_width_log = d_er_band['er_band_log_edges'][1] - d_er_band['er_band_log_edges'][0]

cdf = np.cumsum(d_er_band['er_band_hist'].ravel())
cdf = cdf / cdf[-1]
values = np.random.rand(num_mc_events)
value_bins = np.searchsorted(cdf, values)
s1_idx, log_idx = np.unravel_index(value_bins, (len(d_er_band['er_band_s1_edges'])-1, len(d_er_band['er_band_log_edges'])-1))

a_er_s1 = np.zeros(num_mc_events, dtype=np.float32)
a_er_log = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_random_num_s1 = np.random.random()*bin_width_s1 + d_er_band['er_band_s1_edges'][s1_idx[i]]
    current_random_num_log = np.random.random()*bin_width_log + d_er_band['er_band_log_edges'][log_idx[i]]
    
    
    a_er_s1[i] = current_random_num_s1
    a_er_log[i] = current_random_num_log





# ------------------------------------------------
# ------------------------------------------------
# Pull required arrays for splines
# ------------------------------------------------
# ------------------------------------------------

d_bias_smearing = pickle.load(open('%ss1_s2_bias_and_smearing_wimps.p' % (config_xe1t.path_to_fit_inputs), 'r'))
a_s1bs_s1s = np.asarray(d_bias_smearing['s1']['points'], dtype=np.float32)
a_s1bs_lb_bias = np.asarray(d_bias_smearing['s1']['lb_bias'], dtype=np.float32)
a_s1bs_ub_bias = np.asarray(d_bias_smearing['s1']['ub_bias'], dtype=np.float32)
a_s1bs_lb_smearing = np.asarray(d_bias_smearing['s1']['lb_smearing'], dtype=np.float32)
a_s1bs_ub_smearing = np.asarray(d_bias_smearing['s1']['ub_smearing'], dtype=np.float32)
a_s2bs_s2s = np.asarray(d_bias_smearing['s2']['points'], dtype=np.float32)
a_s2bs_lb_bias = np.asarray(d_bias_smearing['s2']['lb_bias'], dtype=np.float32)
a_s2bs_ub_bias = np.asarray(d_bias_smearing['s2']['ub_bias'], dtype=np.float32)
a_s2bs_lb_smearing = np.asarray(d_bias_smearing['s2']['lb_smearing'], dtype=np.float32)
a_s2bs_ub_smearing = np.asarray(d_bias_smearing['s2']['ub_smearing'], dtype=np.float32)


d_acceptances = pickle.load(open('%sacceptances_wimps.p' % (config_xe1t.path_to_fit_inputs), 'r'))
# PF acceptance
a_s1pf_s1s = np.asarray(d_acceptances['pf_s1']['x_values'], dtype=np.float32)
a_s1pf_lb_acc = np.asarray(d_acceptances['pf_s1']['y_values_lower'], dtype=np.float32)
a_s1pf_mean_acc = np.asarray(d_acceptances['pf_s1']['y_values_mean'], dtype=np.float32)
a_s1pf_ub_acc = np.asarray(d_acceptances['pf_s1']['y_values_upper'], dtype=np.float32)


d_cut_acceptances = pickle.load(open('%scut_acceptances.p' % (config_xe1t.path_to_fit_inputs), 'r'))
# PF acceptance
#print '\n\nTesting with 100% acceptance\n\n'
#d_cut_acceptances['s1']['acceptance'] = [1 for i in xrange(len(d_cut_acceptances['s1']['spline_pts']))]
#d_cut_acceptances['s2']['acceptance'] = [1 for i in xrange(len(d_cut_acceptances['s2']['spline_pts']))]

a_s1cuts_s1s = np.asarray(d_cut_acceptances['s1']['spline_pts'], dtype=np.float32)
a_s1cuts_mean_acc = np.asarray(d_cut_acceptances['s1']['acceptance'], dtype=np.float32)

a_s2cuts_s2s = np.asarray(d_cut_acceptances['s2']['spline_pts'], dtype=np.float32)
a_s2cuts_mean_acc = np.asarray(d_cut_acceptances['s2']['acceptance'], dtype=np.float32)


# ------------------------------------------------
# ------------------------------------------------
#  Bin edges for histograms
# ------------------------------------------------
# ------------------------------------------------

a_bin_edges_s1 = np.linspace(config_xe1t.l_s1_settings[1], config_xe1t.l_s1_settings[2], config_xe1t.l_s1_settings[0]+1, dtype=np.float32)
a_bin_edges_log = np.linspace(config_xe1t.l_log_settings[1], config_xe1t.l_log_settings[2], config_xe1t.l_log_settings[0]+1, dtype=np.float32)


print 'Putting arrays on GPU...'
d_plotting_information['gpu_energies'] = pycuda.gpuarray.to_gpu(a_mc_energy)
d_plotting_information['mc_energy'] = a_mc_energy

d_plotting_information['gpu_x_positions'] = pycuda.gpuarray.to_gpu(a_mc_x)
d_plotting_information['mc_x'] = a_mc_x

d_plotting_information['gpu_y_positions'] = pycuda.gpuarray.to_gpu(a_mc_y)
d_plotting_information['mc_y'] = a_mc_y

d_plotting_information['gpu_z_positions'] = pycuda.gpuarray.to_gpu(a_mc_z)
d_plotting_information['mc_z'] = a_mc_z

d_plotting_information['gpu_e_survival_prob'] = pycuda.gpuarray.to_gpu(a_e_survival_prob)
        
d_plotting_information['gpu_er_band_s1'] = pycuda.gpuarray.to_gpu(a_er_s1)
d_plotting_information['gpu_er_band_log'] = pycuda.gpuarray.to_gpu(a_er_log)

d_plotting_information['gpu_bin_edges_r2'] = pycuda.gpuarray.to_gpu(bin_edges_r2)

d_plotting_information['gpu_bin_edges_z'] = pycuda.gpuarray.to_gpu(bin_edges_z)

d_plotting_information['gpu_s1_correction_map'] = pycuda.gpuarray.to_gpu(s1_correction_map)

d_plotting_information['gpu_bin_edges_x'] = pycuda.gpuarray.to_gpu(bin_edges_x)

d_plotting_information['gpu_bin_edges_y'] = pycuda.gpuarray.to_gpu(bin_edges_y)

d_plotting_information['gpu_s2_correction_map'] = pycuda.gpuarray.to_gpu(s2_correction_map)

d_plotting_information['gpu_bin_edges_s1'] = pycuda.gpuarray.to_gpu(a_bin_edges_s1)
d_plotting_information['gpu_bin_edges_log'] = pycuda.gpuarray.to_gpu(a_bin_edges_log)

d_plotting_information['gpu_s1bs_s1s'] = pycuda.gpuarray.to_gpu(a_s1bs_s1s)

d_plotting_information['gpu_s1bs_lb_bias'] = pycuda.gpuarray.to_gpu(a_s1bs_lb_bias)

d_plotting_information['gpu_s1bs_ub_bias'] = pycuda.gpuarray.to_gpu(a_s1bs_ub_bias)

d_plotting_information['gpu_s1bs_lb_smearing'] = pycuda.gpuarray.to_gpu(a_s1bs_lb_smearing)

d_plotting_information['gpu_s1bs_ub_smearing'] = pycuda.gpuarray.to_gpu(a_s1bs_ub_smearing)

d_plotting_information['gpu_s2bs_s2s'] = pycuda.gpuarray.to_gpu(a_s2bs_s2s)

d_plotting_information['gpu_s2bs_lb_bias'] = pycuda.gpuarray.to_gpu(a_s2bs_lb_bias)

d_plotting_information['gpu_s2bs_ub_bias'] = pycuda.gpuarray.to_gpu(a_s2bs_ub_bias)

d_plotting_information['gpu_s2bs_lb_smearing'] = pycuda.gpuarray.to_gpu(a_s2bs_lb_smearing)

d_plotting_information['gpu_s2bs_ub_smearing'] = pycuda.gpuarray.to_gpu(a_s2bs_ub_smearing)

d_plotting_information['gpu_s1pf_s1s'] = pycuda.gpuarray.to_gpu(a_s1pf_s1s)

d_plotting_information['gpu_s1pf_lb_acc'] = pycuda.gpuarray.to_gpu(a_s1pf_lb_acc)

d_plotting_information['gpu_s1pf_mean_acc'] = pycuda.gpuarray.to_gpu(a_s1pf_mean_acc)

d_plotting_information['gpu_s1pf_ub_acc'] = pycuda.gpuarray.to_gpu(a_s1pf_ub_acc)


d_plotting_information['gpu_s1cuts_s1s'] = pycuda.gpuarray.to_gpu(np.asarray(a_s1cuts_s1s, dtype=np.float32))
d_plotting_information['gpu_s1cuts_mean_acc'] = pycuda.gpuarray.to_gpu(np.asarray(a_s1cuts_mean_acc, dtype=np.float32))
d_plotting_information['gpu_s2cuts_s2s'] = pycuda.gpuarray.to_gpu(np.asarray(a_s2cuts_s2s, dtype=np.float32))
d_plotting_information['gpu_s2cuts_mean_acc'] = pycuda.gpuarray.to_gpu(np.asarray(a_s2cuts_mean_acc, dtype=np.float32))




# get random seeds setup
local_gpu_setup_kernel = pycuda.compiler.SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('setup_kernel')
local_rng_states = drv.mem_alloc(np.int32(num_blocks*block_dim)*pycuda.characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
local_gpu_setup_kernel(np.int32(int(num_blocks*block_dim)), local_rng_states, np.uint64(0), np.uint64(0), grid=(int(num_blocks), 1), block=(int(block_dim), 1, 1))


# get observables function
gpu_observables_func = SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production_with_arrays_no_fv')



a_selected_sampler = []

for par_name in l_grid_parameters:
    a_selected_sampler.append(a_samples[:, d_parameter_to_index[par_name]])

a_selected_sampler = np.asarray(a_selected_sampler)

a_means = np.mean(a_selected_sampler, axis=1)
a_cov_matrix = np.cov(a_selected_sampler)

bin_edges_s1_th2 = np.linspace(config_xe1t.l_s1_settings_pl[1], config_xe1t.l_s1_settings_pl[2], config_xe1t.l_s1_settings_pl[0]+1)
bin_edges_s1_th2 = np.asarray(bin_edges_s1_th2, dtype=np.float32)
#bin_edges_s2_th2 = np.logspace(np.log10(config_xe1t.l_s2_settings_pl[1]), np.log10(config_xe1t.l_s2_settings_pl[2]), config_xe1t.l_s2_settings_pl[0]+1)
bin_edges_s2_th2 = np.linspace(np.log10(config_xe1t.l_s2_settings_pl[1]), np.log10(config_xe1t.l_s2_settings_pl[2]), config_xe1t.l_s2_settings_pl[0]+1)
bin_edges_s2_th2 = np.asarray(bin_edges_s2_th2, dtype=np.float32)

#print bin_edges_s1_th2
#print bin_edges_s2_th2



# -----------------------------------------------
# -----------------------------------------------
# run MC
# -----------------------------------------------
# -----------------------------------------------

d_histograms = {}
d_parameters_for_gpu = {}

s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)

f_fit_info = open('%swimp_%sgev_mean_and_cov_matrix.txt' % (s_path_for_save, s_wimp_mass), 'w')
f_fit_info.write('Bins in Events/Day\n')
f_fit_info.write('Using %.2f kg Fiducial Volume Mass\n\n' % (fiducial_volume_mass))
f_fit_info.write('Parameters: gamma, alpha, lambda, extraction_efficiency\n\n')
f_fit_info.write('Means:\n')
f_fit_info.write('%s\n\n' % (str(a_means)))
f_fit_info.write('Covariance Matrix:\n')
f_fit_info.write('%s\n' % str(a_cov_matrix))
f_fit_info.close()

f_constant = root.TF2('constant', '0.001', bin_edges_s1_th2[0], bin_edges_s1_th2[-1], bin_edges_s2_th2[0], bin_edges_s2_th2[-1])
    
f_hist = root.TFile('%swimp_%sgev_variations_%s.root' % (s_path_for_save, s_wimp_mass, lax_version), 'RECREATE')

for gamma_sigma_level in tqdm.tqdm(l_sigma_levels):
    d_parameters_for_gpu['gamma'] = np.asarray(d_grid_parameter_percentiles['gamma'][gamma_sigma_level], dtype=np.float32)

    for alpha_sigma_level in l_sigma_levels:
        d_parameters_for_gpu['alpha'] = np.asarray(d_grid_parameter_percentiles['alpha'][alpha_sigma_level], dtype=np.float32)

        for lambda_sigma_level in l_sigma_levels:
            d_parameters_for_gpu['lambda'] = np.asarray(d_grid_parameter_percentiles['lambda'][lambda_sigma_level], dtype=np.float32)

            for extraction_efficiency_sigma_level in l_sigma_levels:
                d_parameters_for_gpu['extraction_efficiency'] = np.asarray(d_grid_parameter_percentiles['extraction_efficiency'][extraction_efficiency_sigma_level], dtype=np.float32)
                
                #s_key = 'wimp_%sgev_gamma_%+.3e_%.2fsigma_alpha_%+.3e_%.2fsigma_lambda_%+.3e_%.2fsigma_extraction_efficiency_%+.3e_%.2fsigma' % (s_wimp_mass, d_grid_parameter_percentiles['gamma'][gamma_sigma_level], gamma_sigma_level, d_grid_parameter_percentiles['alpha'][alpha_sigma_level], alpha_sigma_level, d_grid_parameter_percentiles['lambda'][lambda_sigma_level], lambda_sigma_level, d_grid_parameter_percentiles['extraction_efficiency'][extraction_efficiency_sigma_level], extraction_efficiency_sigma_level)
                s_key = 'wimp_%sgev_gamma_%.2fsigma_alpha_%.2fsigma_lambda_%.2fsigma_extraction_efficiency_%.2fsigma' % (s_wimp_mass, gamma_sigma_level, alpha_sigma_level, lambda_sigma_level, extraction_efficiency_sigma_level)

                for i, par_name in enumerate(l_par_names):
                    if par_name == 'prob_bkg':
                        d_parameters_for_gpu[par_name] = np.asarray(0, dtype=np.float32)
                
                    elif not (par_name in l_grid_parameters):
                        d_parameters_for_gpu[par_name] = np.asarray(d_all_parameter_info[par_name]['best_fit'], dtype=np.float32)
                    else:
                        # already filled so pass
                        pass

                num_trials = np.asarray(num_mc_events, dtype=np.int32)
                mean_field = np.asarray(config_xe1t.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)

                num_pts_s1bs = np.asarray(len(a_s1bs_s1s), dtype=np.int32)
                num_pts_s2bs = np.asarray(len(a_s2bs_s2s), dtype=np.int32)
                num_pts_s1pf = np.asarray(len(a_s1pf_s1s), dtype=np.int32)
                num_pts_s1cuts = np.asarray(len(a_s1cuts_s1s), dtype=np.int32)
                num_pts_s2cuts = np.asarray(len(a_s2cuts_s2s), dtype=np.int32)
                
                num_bins_r2 = np.asarray(len(bin_edges_r2)-1, dtype=np.int32)
                num_bins_z = np.asarray(len(bin_edges_z)-1, dtype=np.int32)
                num_bins_x = np.asarray(len(bin_edges_x)-1, dtype=np.int32)
                num_bins_y = np.asarray(len(bin_edges_y)-1, dtype=np.int32)

                #num_loops = np.asarray(num_loops, dtype=np.int32)

                #num_bins_s1 = np.asarray(config_xe1t.l_s1_settings[0], dtype=np.int32)
                #num_bins_log = np.asarray(config_xe1t.l_log_settings[0], dtype=np.int32)
                #a_hist_2d = np.zeros(config_xe1t.l_s1_settings[0]*config_xe1t.l_log_settings[0], dtype=np.float32)
                
                a_s1 = np.zeros(num_mc_events, dtype=np.float32)
                a_s2 = np.zeros(num_mc_events, dtype=np.float32)


                tArgs = (local_rng_states, drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_parameters_for_gpu['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_parameters_for_gpu['w_value']), drv.In(d_parameters_for_gpu['alpha']), drv.In(d_parameters_for_gpu['zeta']), drv.In(d_parameters_for_gpu['beta']), drv.In(d_parameters_for_gpu['gamma']), drv.In(d_parameters_for_gpu['delta']), drv.In(d_parameters_for_gpu['kappa']), drv.In(d_parameters_for_gpu['eta']), drv.In(d_parameters_for_gpu['lambda']), drv.In(d_parameters_for_gpu['g1']), drv.In(d_parameters_for_gpu['extraction_efficiency']), drv.In(d_parameters_for_gpu['gas_gain_mean']), drv.In(d_parameters_for_gpu['gas_gain_width']), drv.In(d_parameters_for_gpu['dpe_prob']), drv.In(d_parameters_for_gpu['s1_bias_par']), drv.In(d_parameters_for_gpu['s1_smearing_par']), drv.In(d_parameters_for_gpu['s2_bias_par']), drv.In(d_parameters_for_gpu['s2_smearing_par']), drv.In(d_parameters_for_gpu['acceptance_par']), drv.In(num_pts_s1bs), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(num_pts_s1pf), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(num_pts_s1cuts), d_plotting_information['gpu_s1cuts_s1s'], d_plotting_information['gpu_s1cuts_mean_acc'], drv.In(num_pts_s2cuts), d_plotting_information['gpu_s2cuts_s2s'], d_plotting_information['gpu_s2cuts_mean_acc'], drv.In(num_bins_r2), d_plotting_information['gpu_bin_edges_r2'], drv.In(num_bins_z), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(num_bins_x), d_plotting_information['gpu_bin_edges_x'], drv.In(num_bins_y), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1), drv.InOut(a_s2))

                gpu_observables_func(*tArgs, **d_gpu_scale)
                
                h_current = root.TH2F(s_key, s_key, config_xe1t.l_s1_settings_pl[0], bin_edges_s1_th2, config_xe1t.l_s2_settings_pl[0], bin_edges_s2_th2)
                h_current.Add(f_constant)
                
                count_events = 0
                
                for i in xrange(len(a_s1)):
                    if (not np.isnan(a_s1[i])) and (not np.isnan(a_s2[i])) and (a_s2[i] > 0):
                        h_current.Fill(a_s1[i], np.log10(a_s2[i]))
                        count_events += 1
                
                #print '\nEvents per day: %f\n' % (count_events*integrated_rate * fiducial_volume_mass / float(num_mc_events))
                        
                h_current.Scale(integrated_rate * fiducial_volume_mass / float(num_mc_events))
                
                d_histograms[s_key] = h_current
                d_histograms[s_key].Write()

                #d_histograms[s_key] = np.reshape(a_hist_2d, (config_xe1t.l_log_settings[0], config_xe1t.l_s1_settings[0]))

                # flipping and transposing for drawing
                #d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'] = np.reshape(a_hist_2d, (config_xe1t.l_log_settings[0], config_xe1t.l_s1_settings[0])).T
                #d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'] = np.rot90(d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'])
                #d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'] = np.flipud(d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'])

                #sum_mc = np.sum(d_histograms[s_key], dtype=np.float32)
                #if sum_mc == 0:
                #    print 'Sum was zero!'
                #    ctx.pop()
                #    sys.exit()

                #d_histograms[s_key] /= sum_mc




f_hist.Close()




"""

fig_grid, l_ax_grid = plt.subplots(4, 1, figsize=figure_sizes)
count_ax = 0

for i, sigma_level in enumerate(l_sigma_levels):
    if sigma_level == 0:
        continue

    zero_mask = d_grid_parameter_histograms[par_name][0]['mc_hist'] == 0.
    current_hist = np.zeros(d_grid_parameter_histograms[par_name][0]['mc_hist'].shape)

    # only fill arrays where you won't get divide errors
    current_hist[~zero_mask] = ((d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'] - d_grid_parameter_histograms[par_name][0]['mc_hist']) / d_grid_parameter_histograms[par_name][0]['mc_hist'])[~zero_mask]

    current_chi2 = np.sum(((d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'] - d_grid_parameter_histograms[par_name][0]['mc_hist'])**2 / d_grid_parameter_histograms[par_name][0]['mc_hist'])[~zero_mask])

    #l_ax_grid[count_ax].pcolormesh(a_bin_edges_s1, a_bin_edges_log, d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'], cmap='bwr')
    pcolor_cax = l_ax_grid[count_ax].pcolormesh(a_bin_edges_s1, a_bin_edges_log, current_hist, vmin=-0.3, vmax=0.3, cmap='bwr')

    if sigma_level < 0:
        s_sign = '-'
    else:
        s_sign = '+'

    l_ax_grid[count_ax].set_title('$\mu %s %d \sigma$' % (s_sign, abs(sigma_level)))
    l_ax_grid[count_ax].set_xlabel('$S1 [PE]$')
    l_ax_grid[count_ax].set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')

    l_ax_grid[count_ax].text(0.75, 0.95, '$ \chi^2 = %.3e $' % (current_chi2), transform=l_ax_grid[count_ax].transAxes, fontsize=10,
        verticalalignment='top')


    count_ax += 1

fig_grid.tight_layout()

fig_grid.subplots_adjust(right=0.8)
cbar_ax = fig_grid.add_axes([0.85, 0.15, 0.05, 0.7])
fig_grid.colorbar(pcolor_cax, cax=cbar_ax)




s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)
    
    
fig_grid.savefig('%s%s_variation.png' % (s_path_for_save, par_to_examine))

"""




# end GPU context
ctx.pop()
print 'Successfully ended GPU context!'





#plt.show()



