#!/usr/bin/python

import sys
import neriX_analysis
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

from rootpy import stl
from rootpy.io import File
from rootpy.tree import Tree, TreeModel, TreeChain
from rootpy.plotting import Hist, Hist2D, Canvas, Graph, func


from math import floor

from sklearn import neighbors
from sklearn import grid_search
from sklearn import preprocessing


import cPickle as pickle



l_s1_cuts = [0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100]
transparency = 0.2
l_quantiles_for_2d = [50]
l_quantiles_for_lines_in_plot = [1, 5, 50]

# num_steps goes into how many random draws you will take from kept sampler
num_steps = 1#10
# num_steps_to_include is how large of sampler you keep
num_steps_to_include = 1000
num_mc_events = int(1e5)
device_number = 0

if(len(sys.argv) != 2):
	print 'Usage is python compare_data_fit.py <num walkers>'
	sys.exit(1)

num_walkers = int(sys.argv[1])


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
l_plots = ['plots', dir_specifier_name, '%s_kV_%s_deg' % (s_cathode_settings, s_degree_settings)]

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
a_percentile_values = np.percentile(a_samples, l_quantiles_for_2d, axis=0)


l_s1_settings = config_xe1t.l_s1_settings
l_log_settings = config_xe1t.l_log_settings
l_s2_settings = config_xe1t.l_s2_settings

mc_bin_number_multiplier = 2
fig_size_2d = (11, 13)
figure_sizes = (11, 2.0*(len(l_s1_cuts)-1))
d_subplot_space_2d = {'wspace':0.2, 'hspace':0.3}
d_subplot_space = {'wspace':0.2, 'hspace':0.6}


d_plotting_information = {}



# need to prepare GPU for MC simulations
import cuda_full_observables_production
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray

drv.init()
dev = drv.Device(device_number)
ctx = dev.make_context()
print 'Device Name: %s\n' % (dev.name())



d_mc_energy = pickle.load(open('%sambe_mc.p' % config_xe1t.path_to_fit_inputs, 'r'))
d_mc_positions = pickle.load(open('%smc_maps.p' % config_xe1t.path_to_fit_inputs, 'r'))
d_er_band = pickle.load(open('%ser_band.p' % config_xe1t.path_to_fit_inputs, 'r'))
d_ac_bkg = pickle.load(open('%sac_bkg.p' % config_xe1t.path_to_fit_inputs, 'r'))

random.seed()
    
    
# -----------------------------------------
#  Energy
# -----------------------------------------

bin_width = d_mc_energy['a_energy_bins'][1] - d_mc_energy['a_energy_bins'][0]

cdf = np.cumsum(d_mc_energy['a_energy_hist'])
cdf = cdf / cdf[-1]
values = np.random.rand(num_mc_events)
value_bins = np.searchsorted(cdf, values)
random_from_cdf = d_mc_energy['a_energy_bins'][value_bins]

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

bin_width = d_mc_positions['z_bin_edges'][1] - d_mc_positions['z_bin_edges'][0]

cdf = np.cumsum(d_mc_positions['z_map'])
cdf = cdf / cdf[-1]
values = np.random.rand(num_mc_events)
value_bins = np.searchsorted(cdf, values)
random_from_cdf = d_mc_positions['z_bin_edges'][value_bins]

a_mc_z = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_random_num = np.random.random()*bin_width + random_from_cdf[i]
    
    a_mc_z[i] = current_random_num





# -----------------------------------------
#  X, Y
# -----------------------------------------

bin_width_x = d_mc_positions['x_bin_edges'][1] - d_mc_positions['x_bin_edges'][0]
bin_width_y = d_mc_positions['y_bin_edges'][1] - d_mc_positions['y_bin_edges'][0]

cdf = np.cumsum(d_mc_positions['xy_map'].T.ravel())
cdf = cdf / cdf[-1]
values = np.random.rand(num_mc_events)
value_bins = np.searchsorted(cdf, values)
x_idx, y_idx = np.unravel_index(value_bins, (len(d_mc_positions['x_bin_edges'])-1, len(d_mc_positions['y_bin_edges'])-1))

a_mc_x = np.zeros(num_mc_events, dtype=np.float32)
a_mc_y = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_random_num_x = np.random.random()*bin_width_x + d_mc_positions['x_bin_edges'][x_idx[i]]
    current_random_num_y = np.random.random()*bin_width_y + d_mc_positions['y_bin_edges'][y_idx[i]]
    
    
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



# -----------------------------------------
#  Get array of AC Bkg S1 and S2
# -----------------------------------------


bin_width_s1 = d_ac_bkg['ac_bkg_s1_edges'][1] - d_ac_bkg['ac_bkg_s1_edges'][0]
bin_width_log = d_ac_bkg['ac_bkg_log_edges'][1] - d_ac_bkg['ac_bkg_log_edges'][0]

cdf = np.cumsum(d_ac_bkg['ac_bkg_hist'].ravel())
cdf = cdf / cdf[-1]
values = np.random.rand(num_mc_events)
value_bins = np.searchsorted(cdf, values)
s1_idx, log_idx = np.unravel_index(value_bins, (len(d_ac_bkg['ac_bkg_s1_edges'])-1, len(d_ac_bkg['ac_bkg_log_edges'])-1))

a_ac_bkg_s1 = np.zeros(num_mc_events, dtype=np.float32)
a_ac_bkg_log = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_random_num_s1 = np.random.random()*bin_width_s1 + d_ac_bkg['ac_bkg_s1_edges'][s1_idx[i]]
    current_random_num_log = np.random.random()*bin_width_log + d_ac_bkg['ac_bkg_log_edges'][log_idx[i]]
    
    
    a_ac_bkg_s1[i] = current_random_num_s1
    a_ac_bkg_log[i] = current_random_num_log





# ------------------------------------------------
# ------------------------------------------------
# Pull required arrays for splines
# ------------------------------------------------
# ------------------------------------------------

d_bias_smearing = pickle.load(open('%ss1_s2_bias_and_smearing.p' % (config_xe1t.path_to_fit_inputs), 'r'))
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


d_acceptances = pickle.load(open('%sacceptances_ambe.p' % (config_xe1t.path_to_fit_inputs), 'r'))
# PF acceptance
a_s1pf_s1s = np.asarray(d_acceptances['pf_s1']['x_values'], dtype=np.float32)
a_s1pf_lb_acc = np.asarray(d_acceptances['pf_s1']['y_values_lower'], dtype=np.float32)
a_s1pf_mean_acc = np.asarray(d_acceptances['pf_s1']['y_values_mean'], dtype=np.float32)
a_s1pf_ub_acc = np.asarray(d_acceptances['pf_s1']['y_values_upper'], dtype=np.float32)


# cut acceptances
cut_acceptance_s1_intercept = config_xe1t.cut_acceptance_s1_intercept
cut_acceptance_s1_intercept_uncertainty = config_xe1t.cut_acceptance_s1_intercept_uncertainty

cut_acceptance_s1_slope = config_xe1t.cut_acceptance_s1_slope
cut_acceptance_s1_slope_uncertainty = config_xe1t.cut_acceptance_s1_slope_uncertainty

cut_acceptance_s2_intercept = config_xe1t.cut_acceptance_s2_intercept
cut_acceptance_s2_intercept_uncertainty = config_xe1t.cut_acceptance_s2_intercept_uncertainty

cut_acceptance_s2_slope = config_xe1t.cut_acceptance_s2_slope
cut_acceptance_s2_slope_uncertainty = config_xe1t.cut_acceptance_s2_slope_uncertainty



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

d_plotting_information['gpu_ac_bkg_s1'] = pycuda.gpuarray.to_gpu(a_ac_bkg_s1)
d_plotting_information['gpu_ac_bkg_log'] = pycuda.gpuarray.to_gpu(a_ac_bkg_log)

d_plotting_information['gpu_bin_edges_r2'] = pycuda.gpuarray.to_gpu(bin_edges_r2)

d_plotting_information['gpu_bin_edges_z'] = pycuda.gpuarray.to_gpu(bin_edges_z)

d_plotting_information['gpu_s1_correction_map'] = pycuda.gpuarray.to_gpu(s1_correction_map)

d_plotting_information['gpu_bin_edges_x'] = pycuda.gpuarray.to_gpu(bin_edges_x)

d_plotting_information['gpu_bin_edges_y'] = pycuda.gpuarray.to_gpu(bin_edges_y)

d_plotting_information['gpu_s2_correction_map'] = pycuda.gpuarray.to_gpu(s2_correction_map)

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




# get random seeds setup
local_gpu_setup_kernel = pycuda.compiler.SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('setup_kernel')
local_rng_states = drv.mem_alloc(np.int32(num_blocks*block_dim)*pycuda.characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
local_gpu_setup_kernel(np.int32(int(num_blocks*block_dim)), local_rng_states, np.uint64(0), np.uint64(0), grid=(int(num_blocks), 1), block=(int(block_dim), 1, 1))


# get observables function
gpu_observables_func = SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production_with_arrays')



# get s1 and s2 data arrays
d_plotting_information['d_s1_s2_data'] = pickle.load(open('%sambe_data.p' % (config_xe1t.path_to_fit_inputs), 'r'))
d_plotting_information['a_s1_data'] = d_plotting_information['d_s1_s2_data']['s1']
d_plotting_information['a_s2_data'] = d_plotting_information['d_s1_s2_data']['s2']

# get number of data pts
d_plotting_information['num_data_pts'] = len(d_plotting_information['a_s1_data'])


# create figure and give settings for space
fig_s1_log, (ax_s1_projection, ax_s1_log_data, ax_s1_log_mc) = plt.subplots(3, sharex=False, sharey=False, figsize=fig_size_2d)
fig_s1_log.subplots_adjust(**d_subplot_space_2d)


# create 2d histogram of data
hist_s1_log_data = ax_s1_log_data.hist2d(d_plotting_information['a_s1_data'], np.log10(d_plotting_information['a_s2_data']/d_plotting_information['a_s1_data']), bins=[l_s1_settings[0], l_log_settings[0]], range=[[l_s1_settings[1], l_s1_settings[2]], [l_log_settings[1], l_log_settings[2]]], cmap='Blues')
fig_s1_log.colorbar(hist_s1_log_data[3], ax=ax_s1_log_data)





# -----------------------------------------------
# -----------------------------------------------
# run MC
# -----------------------------------------------
# -----------------------------------------------


for quantile_number, a_fit_parameters in enumerate(a_percentile_values):

    num_trials = np.asarray(num_mc_events, dtype=np.int32)
    mean_field = np.asarray(config_xe1t.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)

    w_value = np.asarray(a_fit_parameters[0], dtype=np.float32)
    alpha = np.asarray(a_fit_parameters[1], dtype=np.float32)
    zeta = np.asarray(a_fit_parameters[2], dtype=np.float32)
    beta = np.asarray(a_fit_parameters[3], dtype=np.float32)
    gamma = np.asarray(a_fit_parameters[4], dtype=np.float32)
    delta = np.asarray(a_fit_parameters[5], dtype=np.float32)
    kappa = np.asarray(a_fit_parameters[6], dtype=np.float32)
    eta = np.asarray(a_fit_parameters[7], dtype=np.float32)
    lamb = np.asarray(a_fit_parameters[8], dtype=np.float32)

    g1_value = np.asarray(a_fit_parameters[9], dtype=np.float32)
    extraction_efficiency = np.asarray(a_fit_parameters[10], dtype=np.float32)
    gas_gain_value = np.asarray(a_fit_parameters[11], dtype=np.float32)
    gas_gain_width = np.asarray(a_fit_parameters[12], dtype=np.float32)
    dpe_prob = np.asarray(a_fit_parameters[13], dtype=np.float32)

    s1_bias_par = np.asarray(a_fit_parameters[14], dtype=np.float32)
    s1_smearing_par = np.asarray(a_fit_parameters[15], dtype=np.float32)
    s2_bias_par = np.asarray(a_fit_parameters[16], dtype=np.float32)
    s2_smearing_par = np.asarray(a_fit_parameters[17], dtype=np.float32)
    acceptance_par = np.asarray(a_fit_parameters[18], dtype=np.float32)
    
    current_cut_acceptance_s1_intercept = np.asarray(cut_acceptance_s1_intercept + a_fit_parameters[19]*cut_acceptance_s1_intercept_uncertainty, dtype=np.float32)
    current_cut_acceptance_s1_slope = np.asarray(cut_acceptance_s1_slope + a_fit_parameters[19]*cut_acceptance_s1_slope_uncertainty, dtype=np.float32)
    
    current_cut_acceptance_s2_intercept = np.asarray(cut_acceptance_s2_intercept + a_fit_parameters[19]*cut_acceptance_s2_intercept_uncertainty, dtype=np.float32)
    current_cut_acceptance_s2_slope = np.asarray(cut_acceptance_s2_slope + a_fit_parameters[19]*cut_acceptance_s2_slope_uncertainty, dtype=np.float32)

    #print 'Fixing bkg to 0'
    #prob_bkg = np.asarray(0, dtype=np.float32)
    prob_bkg = np.asarray(a_fit_parameters[20], dtype=np.float32)
    prob_ac_bkg = np.asarray(a_fit_parameters[21], dtype=np.float32)
    scale_par = np.asarray(a_fit_parameters[22], dtype=np.float32)
    
    
    
    
    num_pts_s1bs = np.asarray(len(a_s1bs_s1s), dtype=np.int32)
    num_pts_s2bs = np.asarray(len(a_s2bs_s2s), dtype=np.int32)
    num_pts_s1pf = np.asarray(len(a_s1pf_s1s), dtype=np.int32)
    
    num_bins_r2 = np.asarray(len(bin_edges_r2)-1, dtype=np.int32)
    num_bins_z = np.asarray(len(bin_edges_z)-1, dtype=np.int32)
    num_bins_x = np.asarray(len(bin_edges_x)-1, dtype=np.int32)
    num_bins_y = np.asarray(len(bin_edges_y)-1, dtype=np.int32)
    


    a_s1_mc = np.full(num_mc_events, 0, dtype=np.float32)
    a_s2_mc = np.full(num_mc_events, 0, dtype=np.float32)
    a_weights = np.full(num_mc_events, 0, dtype=np.float32)

    tArgs = (local_rng_states, drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(prob_bkg), drv.In(prob_ac_bkg), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], d_plotting_information['gpu_ac_bkg_s1'], d_plotting_information['gpu_ac_bkg_log'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(dpe_prob), drv.In(s1_bias_par), drv.In(s1_smearing_par), drv.In(s2_bias_par), drv.In(s2_smearing_par), drv.In(acceptance_par), drv.In(num_pts_s1bs), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(num_pts_s1pf), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(current_cut_acceptance_s1_intercept), drv.In(current_cut_acceptance_s1_slope), drv.In(current_cut_acceptance_s2_intercept), drv.In(current_cut_acceptance_s2_slope), drv.In(num_bins_r2), d_plotting_information['gpu_bin_edges_r2'], drv.In(num_bins_z), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(num_bins_x), d_plotting_information['gpu_bin_edges_x'], drv.In(num_bins_y), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weights))

    gpu_observables_func(*tArgs, **d_gpu_scale)



hist_s1_log_mc = ax_s1_log_mc.hist2d(a_s1_mc, np.log10(a_s2_mc/a_s1_mc), bins=[l_s1_settings[0], l_log_settings[0]], range=[[l_s1_settings[1], l_s1_settings[2]], [l_log_settings[1], l_log_settings[2]]], cmap='Blues', weights=a_weights)
fig_s1_log.colorbar(hist_s1_log_mc[3], ax=ax_s1_log_mc)


df_data = pd.DataFrame({'s1':d_plotting_information['a_s1_data'], 's2':d_plotting_information['a_s2_data']})
df_data['log'] = np.log10(df_data['s2']/df_data['s1'])

df_mc = pd.DataFrame({'s1':a_s1_mc, 's2':a_s2_mc})
df_mc['log'] = np.log10(df_mc['s2']/df_mc['s1'])



# add alii to dictionary
d_plotting_information['fig_s1_log'] = fig_s1_log
d_plotting_information['ax_s1_projection'] = ax_s1_projection
d_plotting_information['ax_s1_log_data'] = ax_s1_log_data
d_plotting_information['ax_s1_log_mc'] = ax_s1_log_mc
d_plotting_information['hist_s1_log_data'] = hist_s1_log_data
d_plotting_information['df_data'] = df_data
d_plotting_information['df_mc'] = df_mc



# get from last histogram
s1_edges = hist_s1_log_data[1]
log_edges = hist_s1_log_data[2]
s2_edges = np.linspace(l_s2_settings[1], l_s2_settings[2], l_s2_settings[0]+1)

s1_edges_mc = np.linspace(l_s1_settings[1], l_s1_settings[2], mc_bin_number_multiplier*l_s1_settings[0]+1)
s1_mc_bin_width = (l_s1_settings[2]-l_s1_settings[1])/float(l_s1_settings[0]*mc_bin_number_multiplier)
s1_bin_centers_mc = np.linspace(l_s1_settings[1]+s1_mc_bin_width/2., l_s1_settings[2]-s1_mc_bin_width/2., mc_bin_number_multiplier*l_s1_settings[0])
s2_edges_mc = np.linspace(l_s2_settings[1], l_s2_settings[2], mc_bin_number_multiplier*l_s2_settings[0]+1)
s2_mc_bin_width = (l_s2_settings[2]-l_s2_settings[1])/float(l_s2_settings[0]*mc_bin_number_multiplier)
s2_bin_centers_mc = np.linspace(l_s2_settings[1]+s2_mc_bin_width/2., l_s2_settings[2]-s2_mc_bin_width/2., mc_bin_number_multiplier*l_s2_settings[0])
log_edges_mc = np.linspace(l_log_settings[1], l_log_settings[2], mc_bin_number_multiplier*l_log_settings[0]+1)
log_mc_bin_width = (l_log_settings[2]-l_log_settings[1])/float(l_log_settings[0]*mc_bin_number_multiplier)
log_bin_centers_mc = np.linspace(l_log_settings[1]+log_mc_bin_width/2., l_log_settings[2]-log_mc_bin_width/2., mc_bin_number_multiplier*l_log_settings[0])

df_histograms = {}
df_histograms['s1'] = np.zeros((num_walkers*num_steps, len(s1_edges_mc)-1), dtype=float)

d_quantile_lines = {}
d_quantile_lines['s2'] = {}
d_quantile_lines['log'] = {}

df_histograms['s2'] = {}
df_histograms['log'] = {}
d_s2_data_slices = {}
d_log_data_slices = {}
for s1_cut_num in xrange(len(l_s1_cuts)-1):
    current_set_s1_cuts = (l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])
    
    d_quantile_lines['s2'][current_set_s1_cuts] = {}
    d_quantile_lines['log'][current_set_s1_cuts] = {}
    
    df_histograms['s2'][current_set_s1_cuts] = np.zeros((num_walkers*num_steps, len(s1_edges_mc)-1), dtype=float)
    df_histograms['log'][current_set_s1_cuts] = np.zeros((num_walkers*num_steps, len(s1_edges_mc)-1), dtype=float)
    
    
    # get data slices
    cut_df = df_data[(df_data['s1'] > l_s1_cuts[s1_cut_num]) & (df_data['s1'] < l_s1_cuts[s1_cut_num+1])]
    cut_df_mc = df_mc[(df_mc['s1'] > l_s1_cuts[s1_cut_num]) & (df_mc['s1'] < l_s1_cuts[s1_cut_num+1])]
    
    for current_quantile in l_quantiles_for_lines_in_plot:
        d_quantile_lines['s2'][current_set_s1_cuts][current_quantile] = np.percentile(cut_df_mc['s2'], current_quantile)
        d_quantile_lines['log'][current_set_s1_cuts][current_quantile] = np.percentile(cut_df_mc['log'], current_quantile)
    
    d_s2_data_slices[current_set_s1_cuts] = cut_df['s2']
    d_log_data_slices[current_set_s1_cuts] = cut_df['log']



print '\nStarting bands in S1 and S2\n'
l_dfs = [0 for i in xrange(num_walkers*num_steps)]

for i in tqdm.tqdm(xrange(num_walkers*num_steps)):
    # create dictionary to hold relevant information
    l_dfs[i] = {}


    #a_fit_parameters = a_samples[-(np.random.randint(1, num_steps_to_include*num_walkers) % total_length_sampler), :]
    #print 'debugging so no random int'
    a_fit_parameters = a_samples[-i, :]

    # load parameters into proper variables
    num_trials = np.asarray(num_mc_events, dtype=np.int32)
    mean_field = np.asarray(config_xe1t.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)
    
    

    w_value = np.asarray(a_fit_parameters[0], dtype=np.float32)
    alpha = np.asarray(a_fit_parameters[1], dtype=np.float32)
    zeta = np.asarray(a_fit_parameters[2], dtype=np.float32)
    beta = np.asarray(a_fit_parameters[3], dtype=np.float32)
    gamma = np.asarray(a_fit_parameters[4], dtype=np.float32)
    delta = np.asarray(a_fit_parameters[5], dtype=np.float32)
    kappa = np.asarray(a_fit_parameters[6], dtype=np.float32)
    eta = np.asarray(a_fit_parameters[7], dtype=np.float32)
    lamb = np.asarray(a_fit_parameters[8], dtype=np.float32)

    g1_value = np.asarray(a_fit_parameters[9], dtype=np.float32)
    extraction_efficiency = np.asarray(a_fit_parameters[10], dtype=np.float32)
    gas_gain_value = np.asarray(a_fit_parameters[11], dtype=np.float32)
    gas_gain_width = np.asarray(a_fit_parameters[12], dtype=np.float32)
    dpe_prob = np.asarray(a_fit_parameters[13], dtype=np.float32)

    s1_bias_par = np.asarray(a_fit_parameters[14], dtype=np.float32)
    s1_smearing_par = np.asarray(a_fit_parameters[15], dtype=np.float32)
    s2_bias_par = np.asarray(a_fit_parameters[16], dtype=np.float32)
    s2_smearing_par = np.asarray(a_fit_parameters[17], dtype=np.float32)
    acceptance_par = np.asarray(a_fit_parameters[18], dtype=np.float32)

    current_cut_acceptance_s1_intercept = np.asarray(cut_acceptance_s1_intercept + a_fit_parameters[19]*cut_acceptance_s1_intercept_uncertainty, dtype=np.float32)
    current_cut_acceptance_s1_slope = np.asarray(cut_acceptance_s1_slope + a_fit_parameters[19]*cut_acceptance_s1_slope_uncertainty, dtype=np.float32)
    
    current_cut_acceptance_s2_intercept = np.asarray(cut_acceptance_s2_intercept + a_fit_parameters[19]*cut_acceptance_s2_intercept_uncertainty, dtype=np.float32)
    current_cut_acceptance_s2_slope = np.asarray(cut_acceptance_s2_slope + a_fit_parameters[19]*cut_acceptance_s2_slope_uncertainty, dtype=np.float32)

    #print 'Fixing bkg to 0'
    #prob_bkg = np.asarray(0, dtype=np.float32)
    prob_bkg = np.asarray(a_fit_parameters[20], dtype=np.float32)
    prob_ac_bkg = np.asarray(a_fit_parameters[21], dtype=np.float32)
    scale_par = np.asarray(a_fit_parameters[22], dtype=np.float32)
    
    num_pts_s1bs = np.asarray(len(a_s1bs_s1s), dtype=np.int32)
    num_pts_s2bs = np.asarray(len(a_s2bs_s2s), dtype=np.int32)
    num_pts_s1pf = np.asarray(len(a_s1pf_s1s), dtype=np.int32)
    
    num_bins_r2 = np.asarray(len(bin_edges_r2)-1, dtype=np.int32)
    num_bins_z = np.asarray(len(bin_edges_z)-1, dtype=np.int32)
    num_bins_x = np.asarray(len(bin_edges_x)-1, dtype=np.int32)
    num_bins_y = np.asarray(len(bin_edges_y)-1, dtype=np.int32)
    


    a_s1_mc = np.full(num_mc_events, -2, dtype=np.float32)
    a_s2_mc = np.full(num_mc_events, -2, dtype=np.float32)
    a_weights = np.full(num_mc_events, 0, dtype=np.float32)

    tArgs = (local_rng_states, drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(prob_bkg), drv.In(prob_ac_bkg), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], d_plotting_information['gpu_ac_bkg_s1'], d_plotting_information['gpu_ac_bkg_log'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(dpe_prob), drv.In(s1_bias_par), drv.In(s1_smearing_par), drv.In(s2_bias_par), drv.In(s2_smearing_par), drv.In(acceptance_par), drv.In(num_pts_s1bs), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(num_pts_s1pf), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(current_cut_acceptance_s1_intercept), drv.In(current_cut_acceptance_s1_slope), drv.In(current_cut_acceptance_s2_intercept), drv.In(current_cut_acceptance_s2_slope), drv.In(num_bins_r2), d_plotting_information['gpu_bin_edges_r2'], drv.In(num_bins_z), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(num_bins_x), d_plotting_information['gpu_bin_edges_x'], drv.In(num_bins_y), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weights))

    gpu_observables_func(*tArgs, **d_gpu_scale)
    
    
    l_dfs[i]['main_df'] = pd.DataFrame({'s1':a_s1_mc, 's2':a_s2_mc, 'weights':a_weights})
    l_dfs[i]['main_df'] = l_dfs[i]['main_df'][(l_dfs[i]['main_df']['s1'] > 0) & (l_dfs[i]['main_df']['s2'] > 0) & (~l_dfs[i]['main_df']['s1'].isnull()) & (~l_dfs[i]['main_df']['s2'].isnull())]
    
    l_dfs[i]['main_df']['log'] = np.log10(l_dfs[i]['main_df']['s2']/l_dfs[i]['main_df']['s1'])
    
    l_dfs[i]['s1_hist'], _ = np.histogram(l_dfs[i]['main_df']['s1'], s1_edges_mc, weights=l_dfs[i]['main_df']['weights'])
    l_dfs[i]['s1_hist'] = np.asarray(l_dfs[i]['s1_hist'], dtype=float)
    
    #print l_dfs[i]['s1_hist']
    
    # scale_factor for histograms
    #print scale_par, d_plotting_information['num_data_pts'], float(np.sum(l_dfs[i]['s1_hist']))
    scaling_factor_for_histogram = scale_par*d_plotting_information['num_data_pts']/float(np.sum(l_dfs[i]['s1_hist']))*mc_bin_number_multiplier
    
    l_dfs[i]['s1_hist'] *= scaling_factor_for_histogram
    
    df_histograms['s1'][i, :] = l_dfs[i]['s1_hist']
    

    l_dfs[i]['s2_hist_after_s1_cuts'] = {}
    l_dfs[i]['log_hist_after_s1_cuts'] = {}
    for s1_cut_num in xrange(len(l_s1_cuts)-1):
        # make cut
        cut_df = l_dfs[i]['main_df'][(l_dfs[i]['main_df']['s1'] > l_s1_cuts[s1_cut_num]) & (l_dfs[i]['main_df']['s1'] < l_s1_cuts[s1_cut_num+1])]
        
        # add into dictionary
        l_dfs[i]['s2_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])],  _ = np.histogram(cut_df['s2'], s2_edges_mc, weights=cut_df['weights'])
        l_dfs[i]['s2_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])] = np.asarray(l_dfs[i]['s2_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])], dtype=float)
        l_dfs[i]['s2_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])] *= scaling_factor_for_histogram
        
        l_dfs[i]['log_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])],  _ = np.histogram(cut_df['log'], log_edges_mc, weights=cut_df['weights'])
        l_dfs[i]['log_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])] = np.asarray(l_dfs[i]['log_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])], dtype=float)
        l_dfs[i]['log_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])] *= scaling_factor_for_histogram

        df_histograms['s2'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])][i, :] = l_dfs[i]['s2_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])]
        df_histograms['log'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])][i, :] = l_dfs[i]['log_hist_after_s1_cuts'][(l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])]



# now use d_histograms to fill d_quantiles
d_data_histograms = {}
d_quantiles = {}
l_quantiles = [16, 50, 84]
assert len(l_quantiles) == 3, 'Must use exactly three quantiles for analysis'

print '\nGetting quantiles for S1 Histograms...\n'

d_quantiles['s1'] = {}
for quantile in l_quantiles:
    d_quantiles['s1'][quantile] = np.zeros(len(s1_edges_mc)-1)
    for bin_number in xrange(len(s1_edges_mc)-1):
        #print bin_number
        #print np.percentile(df_histograms['s1'][:, bin_number], quantile)
        d_quantiles['s1'][quantile][bin_number] = np.percentile(df_histograms['s1'][:, bin_number], quantile)
d_data_histograms['s1'], _ = np.histogram(df_data['s1'], s1_edges)


print '\nGetting quantiles for S2 Histograms...\n'

d_quantiles['s2'] = {}
d_data_histograms['s2'] = {}
d_quantiles['log'] = {}
d_data_histograms['log'] = {}
for s1_cut_num in xrange(len(l_s1_cuts)-1):
    current_set_s1_cuts = (l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])
    d_quantiles['s2'][current_set_s1_cuts] = {}
    d_quantiles['log'][current_set_s1_cuts] = {}
    for quantile in l_quantiles:
        d_quantiles['s2'][current_set_s1_cuts][quantile] = np.zeros(len(s1_edges_mc)-1)
        d_quantiles['log'][current_set_s1_cuts][quantile] = np.zeros(len(s1_edges_mc)-1)
        for bin_number in xrange(len(s1_edges_mc)-1):
            d_quantiles['s2'][current_set_s1_cuts][quantile][bin_number] = np.percentile(df_histograms['s2'][current_set_s1_cuts][:, bin_number], quantile)
            d_quantiles['log'][current_set_s1_cuts][quantile][bin_number] = np.percentile(df_histograms['log'][current_set_s1_cuts][:, bin_number], quantile)
    d_data_histograms['s2'][current_set_s1_cuts], _ = np.histogram(d_s2_data_slices[current_set_s1_cuts], s2_edges)
    d_data_histograms['log'][current_set_s1_cuts], _ = np.histogram(d_log_data_slices[current_set_s1_cuts], log_edges)





# remove all events outside bounds
df_mc = df_mc[(df_mc['s1'] > l_s1_settings[1]) & (df_mc['s1'] < l_s1_settings[2]) & (df_mc['log'] > l_log_settings[1]) & (df_mc['log'] < l_log_settings[2])]


# grab median lines for each
assert l_s1_settings[0] == l_log_settings[0]

a_median_line_s1 = [0 for i in xrange(l_s1_settings[0])]
a_median_line_data_log = [0 for i in xrange(l_s1_settings[0])]
a_median_line_mc_log = [0 for i in xrange(l_s1_settings[0])]

l_missed_pts = []

for i in xrange(l_s1_settings[0]):
    a_median_line_s1[i] = (s1_edges[i+1] + s1_edges[i]) / 2.

    current_df_data = df_data[(df_data['s1'] > s1_edges[i]) & (df_data['s1'] < s1_edges[i+1])]
    current_df_mc = df_mc[(df_mc['s1'] > s1_edges[i]) & (df_mc['s1'] < s1_edges[i+1])]

    if len(current_df_data['log']) == 0:
        #a_median_line_data_log[i] = a_median_line_data_log[i-1]
        l_missed_pts.append(i)
    else:
        #print current_df_data['log']
        #print np.median(current_df_data['log'])
        #print '\n\n'
        a_median_line_data_log[i] = np.median(current_df_data['log'])

    if len(current_df_mc['log']) == 0:
        a_median_line_mc_log[i] = a_median_line_mc_log[i-1]
    else:
        a_median_line_mc_log[i] = np.median(current_df_mc['log'])


l_missed_pts.reverse()
for pt in l_missed_pts:
    a_median_line_s1.pop(pt)
    a_median_line_data_log.pop(pt)
    a_median_line_mc_log.pop(pt)


# draw on data hist and get handles
line_handle_data, = ax_s1_log_data.plot(a_median_line_s1, a_median_line_data_log, color='r', label='Data Median')
line_handle_mc, = ax_s1_log_data.plot(a_median_line_s1, a_median_line_mc_log, color='magenta', label='MC Median')

# draw on mc hist
ax_s1_log_mc.plot(a_median_line_s1, a_median_line_data_log, color='r', label='Data Median')
ax_s1_log_mc.plot(a_median_line_s1, a_median_line_mc_log, color='magenta', label='MC Median')


# add labels to plots
ax_s1_log_data.set_title(r'NR Band Data - $V_c$ = %.3f kV' % (cathode_setting), fontsize=12)
ax_s1_log_data.set_xlim(s1_edges_mc[0], s1_edges_mc[-1])
ax_s1_log_data.set_xlabel('S1 [PE]')
ax_s1_log_data.set_ylabel(r'$log_{10}(\frac{S2}{S1})$')

ax_s1_log_mc.set_title(r'NR Band Best Fit MC - $V_c$ = %.3f kV' % (cathode_setting), fontsize=12)
ax_s1_log_mc.set_xlim(s1_edges_mc[0], s1_edges_mc[-1])
ax_s1_log_mc.set_xlabel('S1 [PE]')
ax_s1_log_mc.set_ylabel(r'$log_{10}(\frac{S2}{S1})$')


ax_s1_log_data.legend(handles=[line_handle_data, line_handle_mc], loc='best')






# grab S1 information for plot
a_s1_x_values, a_s1_y_values, a_s1_x_err_low, a_s1_x_err_high, a_s1_y_err_low, a_s1_y_err_high = neriX_analysis.prepare_hist_arrays_for_plotting(d_data_histograms['s1'], s1_edges)

ax_s1_projection.errorbar(a_s1_x_values, a_s1_y_values, xerr=[a_s1_x_err_low, a_s1_x_err_high], yerr=[a_s1_y_err_low, a_s1_y_err_high], linestyle='', color='black')

ax_s1_projection.plot(s1_bin_centers_mc, d_quantiles['s1'][l_quantiles[1]], color='b', linestyle='--')
ax_s1_projection.fill_between(s1_bin_centers_mc, d_quantiles['s1'][l_quantiles[0]], d_quantiles['s1'][l_quantiles[2]], color='b', alpha=transparency)

ax_s1_projection.set_xlim(s1_edges_mc[0], s1_edges_mc[-1])
ax_s1_projection.set_xlabel('S1 [PE]')
ax_s1_projection.set_ylabel('Counts')

#print len(s1_bin_centers_mc)
#print s1_bin_centers_mc


# produce 1D plots of S1 and slices of S2
fig_s2s, l_s2_axes = plt.subplots(len(l_s1_cuts)-1, figsize=figure_sizes)
fig_log, l_log_axes = plt.subplots(len(l_s1_cuts)-1, figsize=figure_sizes)

fig_s2s_logy, l_s2_logy_axes = plt.subplots(len(l_s1_cuts)-1, figsize=figure_sizes)
fig_log_logy, l_log_logy_axes = plt.subplots(len(l_s1_cuts)-1, figsize=figure_sizes)

#fig_s2s.tight_layout()
#fig_log.tight_layout()
fig_s2s.subplots_adjust(**d_subplot_space)
fig_log.subplots_adjust(**d_subplot_space)

d_s2_plots = {}
d_log_plots = {}
d_s2_logy_plots = {}
d_log_logy_plots = {}
for i, s1_cut_num in enumerate(xrange(len(l_s1_cuts)-1)):
    current_set_s1_cuts = (l_s1_cuts[s1_cut_num], l_s1_cuts[s1_cut_num+1])
    
    d_s2_plots[current_set_s1_cuts] = {}
    d_log_plots[current_set_s1_cuts] = {}
    d_s2_logy_plots[current_set_s1_cuts] = {}
    d_log_logy_plots[current_set_s1_cuts] = {}

    d_s2_plots[current_set_s1_cuts]['axis'] = l_s2_axes[i]
    d_log_plots[current_set_s1_cuts]['axis'] = l_log_axes[i]
    d_s2_logy_plots[current_set_s1_cuts]['axis'] = l_s2_logy_axes[i]
    d_log_logy_plots[current_set_s1_cuts]['axis'] = l_log_logy_axes[i]

    # get values for actual plotting
    d_s2_plots[current_set_s1_cuts]['x_values'], d_s2_plots[current_set_s1_cuts]['y_values'], d_s2_plots[current_set_s1_cuts]['x_err_low'], d_s2_plots[current_set_s1_cuts]['x_err_high'], d_s2_plots[current_set_s1_cuts]['y_err_low'], d_s2_plots[current_set_s1_cuts]['y_err_high'] = neriX_analysis.prepare_hist_arrays_for_plotting(d_data_histograms['s2'][current_set_s1_cuts], s2_edges)
    d_log_plots[current_set_s1_cuts]['x_values'], d_log_plots[current_set_s1_cuts]['y_values'], d_log_plots[current_set_s1_cuts]['x_err_low'], d_log_plots[current_set_s1_cuts]['x_err_high'], d_log_plots[current_set_s1_cuts]['y_err_low'], d_log_plots[current_set_s1_cuts]['y_err_high'] = neriX_analysis.prepare_hist_arrays_for_plotting(d_data_histograms['log'][current_set_s1_cuts], log_edges)

    # now actual plot onto axis
    d_s2_plots[current_set_s1_cuts]['axis'].errorbar(d_s2_plots[current_set_s1_cuts]['x_values'], d_s2_plots[current_set_s1_cuts]['y_values'], xerr=[d_s2_plots[current_set_s1_cuts]['x_err_high'], d_s2_plots[current_set_s1_cuts]['x_err_high']], yerr=[d_s2_plots[current_set_s1_cuts]['y_err_low'], d_s2_plots[current_set_s1_cuts]['y_err_high']], linestyle='', color='black')
    d_log_plots[current_set_s1_cuts]['axis'].errorbar(d_log_plots[current_set_s1_cuts]['x_values'], d_log_plots[current_set_s1_cuts]['y_values'], xerr=[d_log_plots[current_set_s1_cuts]['x_err_high'], d_log_plots[current_set_s1_cuts]['x_err_high']], yerr=[d_log_plots[current_set_s1_cuts]['y_err_low'], d_log_plots[current_set_s1_cuts]['y_err_high']], linestyle='', color='black')
    d_s2_logy_plots[current_set_s1_cuts]['axis'].errorbar(d_s2_plots[current_set_s1_cuts]['x_values'], d_s2_plots[current_set_s1_cuts]['y_values'], xerr=[d_s2_plots[current_set_s1_cuts]['x_err_high'], d_s2_plots[current_set_s1_cuts]['x_err_high']], yerr=[d_s2_plots[current_set_s1_cuts]['y_err_low'], d_s2_plots[current_set_s1_cuts]['y_err_high']], linestyle='', color='black')
    d_log_logy_plots[current_set_s1_cuts]['axis'].errorbar(d_log_plots[current_set_s1_cuts]['x_values'], d_log_plots[current_set_s1_cuts]['y_values'], xerr=[d_log_plots[current_set_s1_cuts]['x_err_high'], d_log_plots[current_set_s1_cuts]['x_err_high']], yerr=[d_log_plots[current_set_s1_cuts]['y_err_low'], d_log_plots[current_set_s1_cuts]['y_err_high']], linestyle='', color='black')
    
    # plot median
    d_s2_plots[current_set_s1_cuts]['axis'].plot(s2_bin_centers_mc, d_quantiles['s2'][current_set_s1_cuts][l_quantiles[1]], color='b', linestyle='--')
    d_log_plots[current_set_s1_cuts]['axis'].plot(log_bin_centers_mc, d_quantiles['log'][current_set_s1_cuts][l_quantiles[1]], color='b', linestyle='--')
    d_s2_logy_plots[current_set_s1_cuts]['axis'].plot(s2_bin_centers_mc, d_quantiles['s2'][current_set_s1_cuts][l_quantiles[1]], color='b', linestyle='--')
    d_log_logy_plots[current_set_s1_cuts]['axis'].plot(log_bin_centers_mc, d_quantiles['log'][current_set_s1_cuts][l_quantiles[1]], color='b', linestyle='--')
    
    # plot shaded region
    d_s2_plots[current_set_s1_cuts]['axis'].fill_between(s2_bin_centers_mc, d_quantiles['s2'][current_set_s1_cuts][l_quantiles[0]], d_quantiles['s2'][current_set_s1_cuts][l_quantiles[2]], color='b', alpha=transparency)
    d_log_plots[current_set_s1_cuts]['axis'].fill_between(log_bin_centers_mc, d_quantiles['log'][current_set_s1_cuts][l_quantiles[0]], d_quantiles['log'][current_set_s1_cuts][l_quantiles[2]], color='b', alpha=transparency)
    d_s2_logy_plots[current_set_s1_cuts]['axis'].fill_between(s2_bin_centers_mc, d_quantiles['s2'][current_set_s1_cuts][l_quantiles[0]], d_quantiles['s2'][current_set_s1_cuts][l_quantiles[2]], color='b', alpha=transparency)
    d_log_logy_plots[current_set_s1_cuts]['axis'].fill_between(log_bin_centers_mc, d_quantiles['log'][current_set_s1_cuts][l_quantiles[0]], d_quantiles['log'][current_set_s1_cuts][l_quantiles[2]], color='b', alpha=transparency)
    
    # add lines representing quantiles in question (vertical lines)
    for current_quantile in l_quantiles_for_lines_in_plot:
        d_s2_plots[current_set_s1_cuts]['axis'].axvline(d_quantile_lines['s2'][current_set_s1_cuts][current_quantile], color='r', linestyle='--')
        d_log_plots[current_set_s1_cuts]['axis'].axvline(d_quantile_lines['log'][current_set_s1_cuts][current_quantile], color='r', linestyle='--')
        d_s2_logy_plots[current_set_s1_cuts]['axis'].axvline(d_quantile_lines['s2'][current_set_s1_cuts][current_quantile], color='r', linestyle='--')
        d_log_logy_plots[current_set_s1_cuts]['axis'].axvline(d_quantile_lines['log'][current_set_s1_cuts][current_quantile], color='r', linestyle='--')

    d_s2_plots[current_set_s1_cuts]['axis'].set_xlabel('S2 [PE]')
    d_s2_plots[current_set_s1_cuts]['axis'].set_ylabel('Counts')
    d_log_plots[current_set_s1_cuts]['axis'].set_xlabel(r'$Log_{10}(\frac{S2}{S1})$')
    d_log_plots[current_set_s1_cuts]['axis'].set_ylabel('Counts')
    d_s2_logy_plots[current_set_s1_cuts]['axis'].set_xlabel('S2 [PE]')
    d_s2_logy_plots[current_set_s1_cuts]['axis'].set_ylabel('Counts')
    d_log_logy_plots[current_set_s1_cuts]['axis'].set_xlabel(r'$Log_{10}(\frac{S2}{S1})$')
    d_log_logy_plots[current_set_s1_cuts]['axis'].set_ylabel('Counts')


    # for logy plots, make y-axis logarithmic
    d_s2_logy_plots[current_set_s1_cuts]['axis'].set_ylim(0.1, max(max(d_s2_plots[current_set_s1_cuts]['y_values']), max(d_quantiles['s2'][current_set_s1_cuts][l_quantiles[2]]))*1.2)
    d_log_logy_plots[current_set_s1_cuts]['axis'].set_ylim(0.1, max(max(d_log_plots[current_set_s1_cuts]['y_values']), max(d_quantiles['log'][current_set_s1_cuts][l_quantiles[2]]))*1.2)
    d_s2_logy_plots[current_set_s1_cuts]['axis'].set_yscale('log')
    d_log_logy_plots[current_set_s1_cuts]['axis'].set_yscale('log')


    # create text box
    d_s2_plots[current_set_s1_cuts]['axis'].text(0.75, 0.95, '$ %d < S1 < %d $' % current_set_s1_cuts, transform=d_s2_plots[current_set_s1_cuts]['axis'].transAxes, fontsize=10, verticalalignment='top')
    d_log_plots[current_set_s1_cuts]['axis'].text(0.75, 0.95, '$ %d < S1 < %d $' % current_set_s1_cuts, transform=d_log_plots[current_set_s1_cuts]['axis'].transAxes, fontsize=10, verticalalignment='top')
    d_s2_logy_plots[current_set_s1_cuts]['axis'].text(0.75, 0.95, '$ %d < S1 < %d $' % current_set_s1_cuts, transform=d_s2_logy_plots[current_set_s1_cuts]['axis'].transAxes, fontsize=10, verticalalignment='top')
    d_log_logy_plots[current_set_s1_cuts]['axis'].text(0.75, 0.95, '$ %d < S1 < %d $' % current_set_s1_cuts, transform=d_log_logy_plots[current_set_s1_cuts]['axis'].transAxes, fontsize=10, verticalalignment='top')









s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)
    
d_median_line = {}
d_median_line['s1'] = a_median_line_s1
d_median_line['data_log_s2_s1_median'] = a_median_line_data_log
d_median_line['mc_log_s2_s1_median'] = a_median_line_mc_log

pickle.dump(d_median_line, open('%snr_band_median_line.p' % (s_path_for_save), 'w'))


fig_s1_log.savefig('%ss_data_mc_comparison_s1_and_2d_%s_%.3f_kV.png' % (s_path_for_save, dir_specifier_name, cathode_setting))
fig_s2s.savefig('%ss_data_mc_comparison_s2_slices_%s_%.3f_kV.png' % (s_path_for_save, dir_specifier_name, cathode_setting))
fig_log.savefig('%ss_data_mc_comparison_log_slices_%s_%.3f_kV.png' % (s_path_for_save, dir_specifier_name, cathode_setting))
fig_s2s_logy.savefig('%ss_data_mc_comparison_s2_slices_logy_%s_%.3f_kV.png' % (s_path_for_save, dir_specifier_name, cathode_setting))
fig_log_logy.savefig('%ss_data_mc_comparison_log_slices_logy_%s_%.3f_kV.png' % (s_path_for_save, dir_specifier_name, cathode_setting))






# end GPU context
ctx.pop()
print 'Successfully ended GPU context!'





#plt.show()



