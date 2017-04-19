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

from scipy.interpolate import spline

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

plt.rc('text', usetex=True)

l_s1_cuts = [0, 10, 40, 100]
transparency = 0.2
l_quantiles_for_2d = [50]

# num_steps goes into how many random draws you will take from kept sampler
num_steps = 10
# num_steps_to_include is how large of sampler you keep
num_mc_events = int(2e4)
num_steps_to_include = 1
device_number = 0

if(len(sys.argv) != 2):
	print 'Usage is python compare_data_fit.py <num walkers>'
	sys.exit(1)

num_walkers = int(sys.argv[1])

fiducial_volume_mass = 1000. # kg
roi_lb_cut_s1 = 3.
print '\n\nGiving rate assuming %0.f kg FV\n\n\n' % (fiducial_volume_mass)

lax_version = 'lax_0.11.1'
print '\n\nCurrently using %s for fitting\n\n\n' % (lax_version)

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


l_s1_settings = [10*8, 0, 70]
l_log_settings = config_xe1t.l_log_settings
l_s2_settings = config_xe1t.l_s2_settings

l_energy_settings = [70, 0, 70]

s1_edges = np.linspace(l_s1_settings[1], l_s1_settings[2], l_s1_settings[0]+1)
log_edges = np.linspace(l_log_settings[1], l_log_settings[2], l_log_settings[0]+1)
s2_edges = np.linspace(l_s2_settings[1], l_s2_settings[2], l_s2_settings[0]+1)

bin_edges_s1_th2 = np.linspace(config_xe1t.l_s1_settings_pl[1], config_xe1t.l_s1_settings_pl[2], config_xe1t.l_s1_settings_pl[0]+1)
bin_edges_s1_th2 = np.asarray(bin_edges_s1_th2, dtype=np.float32)
bin_edges_s2_th2 = np.linspace(np.log10(config_xe1t.l_s2_settings_pl[1]), np.log10(config_xe1t.l_s2_settings_pl[2]), config_xe1t.l_s2_settings_pl[0]+1)
bin_edges_s2_th2 = np.asarray(bin_edges_s2_th2, dtype=np.float32)


mc_bin_number_multiplier = 5
figure_sizes = (11, 7)
d_subplot_space = {'wspace':0.2, 'hspace':0.3}


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


d_mc_energy = pickle.load(open('%swimp_mc.p' % config_xe1t.path_to_fit_inputs, 'r'))
d_er_band = pickle.load(open('%ser_band.p' % config_xe1t.path_to_fit_inputs, 'r'))
d_ac_bkg = pickle.load(open('%sac_bkg.p' % config_xe1t.path_to_fit_inputs, 'r'))

random.seed()
    
    

# -----------------------------------------
#  Energy
# -----------------------------------------


a_mc_energy = current_random_num = np.random.uniform(l_energy_settings[1], l_energy_settings[2], size=num_mc_events)
a_mc_energy = np.asarray(a_mc_energy, dtype=np.float32)
    


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


d_acceptances = pickle.load(open('%sacceptances_wimps.p' % (config_xe1t.path_to_fit_inputs), 'r'))
# PF acceptance
#print '\n\nTesting with 100% acceptance\n\n'
#d_acceptances['pf_s1']['y_values_lower'] = [1 for i in xrange(len(d_acceptances['pf_s1']['x_values']))]
#d_acceptances['pf_s1']['y_values_mean'] = [1 for i in xrange(len(d_acceptances['pf_s1']['x_values']))]
#d_acceptances['pf_s1']['y_values_upper'] = [1 for i in xrange(len(d_acceptances['pf_s1']['x_values']))]

a_s1pf_s1s = np.asarray(d_acceptances['pf_s1']['x_values'], dtype=np.float32)
a_s1pf_lb_acc = np.asarray(d_acceptances['pf_s1']['y_values_lower'], dtype=np.float32)
a_s1pf_mean_acc = np.asarray(d_acceptances['pf_s1']['y_values_mean'], dtype=np.float32)
a_s1pf_ub_acc = np.asarray(d_acceptances['pf_s1']['y_values_upper'], dtype=np.float32)





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
gpu_observables_func_arrays = SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production_with_arrays_no_fv_corrected_and_uncorrected_and_acceptances')





# -----------------------------------------------
# -----------------------------------------------
# run MC
# -----------------------------------------------
# -----------------------------------------------

s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)

l_dataframes = [0 for i in xrange(num_steps_to_include*num_walkers)]

# threshold is only s2 so need to apply s1 threshold
def s1_threshold(cs1):
    if cs1 > 3. and cs1 < 70.:
        return 1.
    else:
        return 0.

for i in tqdm.tqdm(xrange(num_steps_to_include*num_walkers)):
    a_fit_parameters = a_samples[-i,:]

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

    #print 'Fixing bkg to 0'
    prob_bkg = np.asarray(0, dtype=np.float32)
    #prob_bkg = np.asarray(a_fit_parameters[19], dtype=np.float32)
    #prob_ac_bkg = np.asarray(0, dtype=np.float32)
    scale_par = np.asarray(1, dtype=np.float32)
    
    
    
    
    num_pts_s1bs = np.asarray(len(a_s1bs_s1s), dtype=np.int32)
    num_pts_s2bs = np.asarray(len(a_s2bs_s2s), dtype=np.int32)
    num_pts_s1pf = np.asarray(len(a_s1pf_s1s), dtype=np.int32)
    
    num_bins_r2 = np.asarray(len(bin_edges_r2)-1, dtype=np.int32)
    num_bins_z = np.asarray(len(bin_edges_z)-1, dtype=np.int32)
    num_bins_x = np.asarray(len(bin_edges_x)-1, dtype=np.int32)
    num_bins_y = np.asarray(len(bin_edges_y)-1, dtype=np.int32)
    
    cut_acceptance_par = np.asarray(a_fit_parameters[19], dtype=np.float32)
    cut_acceptance_s1_intercept = np.asarray(config_xe1t.cut_acceptance_s1_intercept + cut_acceptance_par*config_xe1t.cut_acceptance_s1_intercept_uncertainty, dtype=np.float32)
    cut_acceptance_s1_slope = np.asarray(config_xe1t.cut_acceptance_s1_slope + cut_acceptance_par*config_xe1t.cut_acceptance_s1_slope_uncertainty, dtype=np.float32)
    
    # need to increase by 4% because of s2 tails cut
    cut_acceptance_s2_intercept = np.asarray((config_xe1t.cut_acceptance_s2_intercept + cut_acceptance_par*config_xe1t.cut_acceptance_s2_intercept_uncertainty)*1.04, dtype=np.float32)
    cut_acceptance_s2_slope = np.asarray(config_xe1t.cut_acceptance_s2_slope + cut_acceptance_par*config_xe1t.cut_acceptance_s2_slope_uncertainty, dtype=np.float32)

    num_bins_s1 = np.asarray(l_s1_settings[0], dtype=np.int32)
    num_bins_log = np.asarray(l_log_settings[0], dtype=np.int32)
    
    gpu_s1_edges = np.asarray(s1_edges, np.float32)
    gpu_log_edges = np.asarray(log_edges, np.float32)
    
    a_s1 = np.full(num_mc_events, -1, dtype=np.float32)
    a_s2 = np.full(num_mc_events, -2, dtype=np.float32)
    a_cs1 = np.full(num_mc_events, -1, dtype=np.float32)
    a_cs2 = np.full(num_mc_events, -2, dtype=np.float32)
    
    a_pax_acceptances = np.full(num_mc_events, -3, dtype=np.float32)
    a_s1_cut_acceptances = np.full(num_mc_events, -3, dtype=np.float32)
    a_s2_cut_acceptances = np.full(num_mc_events, -3, dtype=np.float32)
    a_threshold_acceptances = np.full(num_mc_events, -3, dtype=np.float32)
    
    tArgs = (local_rng_states, drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(prob_bkg), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(dpe_prob), drv.In(s1_bias_par), drv.In(s1_smearing_par), drv.In(s2_bias_par), drv.In(s2_smearing_par), drv.In(acceptance_par), drv.In(num_pts_s1bs), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(num_pts_s1pf), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(cut_acceptance_s1_intercept), drv.In(cut_acceptance_s1_slope), drv.In(cut_acceptance_s2_intercept), drv.In(cut_acceptance_s2_slope), drv.In(num_bins_r2), d_plotting_information['gpu_bin_edges_r2'], drv.In(num_bins_z), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(num_bins_x), d_plotting_information['gpu_bin_edges_x'], drv.In(num_bins_y), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1), drv.InOut(a_s2), drv.InOut(a_cs1), drv.InOut(a_cs2), drv.InOut(a_pax_acceptances), drv.InOut(a_s1_cut_acceptances), drv.InOut(a_s2_cut_acceptances), drv.InOut(a_threshold_acceptances))

    gpu_observables_func_arrays(*tArgs, **d_gpu_scale)

    l_dataframes[i] = pd.DataFrame({'energy':a_mc_energy, 's1':a_s1, 's2':a_s2, 'cs1':a_cs1, 'cs2':a_cs2, 'pax_acceptances':a_pax_acceptances, 's1_cut_acceptances':a_s1_cut_acceptances, 's2_cut_acceptances':a_s2_cut_acceptances, 'threshold_acceptances':a_threshold_acceptances})

    l_dataframes[i]['threshold_acceptances'] *= l_dataframes[i]['cs1'].apply(s1_threshold)

# end GPU context
ctx.pop()
print 'Successfully ended GPU context!'

bin_edges_energy = np.linspace(l_energy_settings[1], l_energy_settings[2], l_energy_settings[0]+1)
bin_centers_energy = (bin_edges_energy[1:] + bin_edges_energy[:-1]) / 2.

l_percentiles = [16., 50., 84.]

d_arrays = {}
d_arrays['s1_cut_only'] = {}
d_arrays['s2_cut_only'] = {}
d_arrays['pax_only'] = {}
d_arrays['threshold_only'] = {}
d_arrays['total_minus_threshold'] = {}
d_arrays['total'] = {}

l_keys = ['threshold_only', 's1_cut_only', 's2_cut_only', 'pax_only', 'total_minus_threshold', 'total']

d_arrays['s1_cut_only']['bin_centers'] = bin_centers_energy
d_arrays['s1_cut_only']['a_prob'] = np.zeros(l_energy_settings[0])
d_arrays['s1_cut_only']['a_lb'] = np.zeros(l_energy_settings[0])
d_arrays['s1_cut_only']['a_ub'] = np.zeros(l_energy_settings[0])
d_arrays['s1_cut_only']['color'] = 'g'
d_arrays['s1_cut_only']['linestyle'] = '-.'
d_arrays['s1_cut_only']['label'] = r'S1 Cut Signal Efficiency Only'

d_arrays['s2_cut_only']['bin_centers'] = bin_centers_energy
d_arrays['s2_cut_only']['a_prob'] = np.zeros(l_energy_settings[0])
d_arrays['s2_cut_only']['a_lb'] = np.zeros(l_energy_settings[0])
d_arrays['s2_cut_only']['a_ub'] = np.zeros(l_energy_settings[0])
d_arrays['s2_cut_only']['color'] = 'r'
d_arrays['s2_cut_only']['linestyle'] = '-.'
d_arrays['s2_cut_only']['label'] = r'S2 Cut Signal Efficiency Only'

d_arrays['pax_only']['bin_centers'] = bin_centers_energy
d_arrays['pax_only']['a_prob'] = np.zeros(l_energy_settings[0])
d_arrays['pax_only']['a_lb'] = np.zeros(l_energy_settings[0])
d_arrays['pax_only']['a_ub'] = np.zeros(l_energy_settings[0])
d_arrays['pax_only']['color'] = 'b'
d_arrays['pax_only']['linestyle'] = '-.'
d_arrays['pax_only']['label'] = r'PAX Signal Efficiency Only'

d_arrays['threshold_only']['bin_centers'] = bin_centers_energy
d_arrays['threshold_only']['a_prob'] = np.zeros(l_energy_settings[0])
d_arrays['threshold_only']['a_lb'] = np.zeros(l_energy_settings[0])
d_arrays['threshold_only']['a_ub'] = np.zeros(l_energy_settings[0])
d_arrays['threshold_only']['color'] = 'orange'
d_arrays['threshold_only']['linestyle'] = '-.'
d_arrays['threshold_only']['label'] = r'Threshold Signal Efficiency Only'

d_arrays['total_minus_threshold']['bin_centers'] = bin_centers_energy
d_arrays['total_minus_threshold']['a_prob'] = np.zeros(l_energy_settings[0])
d_arrays['total_minus_threshold']['a_lb'] = np.zeros(l_energy_settings[0])
d_arrays['total_minus_threshold']['a_ub'] = np.zeros(l_energy_settings[0])
d_arrays['total_minus_threshold']['color'] = 'magenta'
d_arrays['total_minus_threshold']['linestyle'] = '--'
d_arrays['total_minus_threshold']['label'] = r'Total Signal Efficiency (Excluding Thresholds)'

d_arrays['total']['bin_centers'] = bin_centers_energy
d_arrays['total']['a_prob'] = np.zeros(l_energy_settings[0])
d_arrays['total']['a_lb'] = np.zeros(l_energy_settings[0])
d_arrays['total']['a_ub'] = np.zeros(l_energy_settings[0])
d_arrays['total']['color'] = 'black'
d_arrays['total']['linestyle'] = '--'
d_arrays['total']['label'] = r'Total Signal Efficiency'


for i in tqdm.tqdm(xrange(l_energy_settings[0])):

    # make arrays to store events for median and bands
    a_num_total_events = np.zeros(num_walkers*num_steps_to_include)
    a_num_events_no_signal = np.zeros(num_walkers*num_steps_to_include)
    a_num_events_s1_cuts = np.zeros(num_walkers*num_steps_to_include)
    a_num_events_s2_cuts = np.zeros(num_walkers*num_steps_to_include)
    a_num_events_pax = np.zeros(num_walkers*num_steps_to_include)
    a_num_events_threshold = np.zeros(num_walkers*num_steps_to_include)
    a_num_events_total_minus_threshold = np.zeros(num_walkers*num_steps_to_include)
    a_num_events_total = np.zeros(num_walkers*num_steps_to_include)

    for j in xrange(num_steps_to_include*num_walkers):

        current_df = l_dataframes[j][(l_dataframes[i]['energy'] > bin_edges_energy[i]) & (l_dataframes[i]['energy'] < bin_edges_energy[i+1])]
        
        a_num_total_events[j] = float(len(current_df))

        # number for acceptance/s1/s2 less than zero means that was not physical event
        # (0 photons or s2 < 200 for example)
        temp_df = current_df[(current_df['s1_cut_acceptances'] < 0) | (current_df['pax_acceptances'] < 0) | (current_df['s2_cut_acceptances'] < 0)]
        a_num_events_no_signal[j] = len(temp_df)
        
        #print current_df[~(current_df['s1_cut_acceptances'] > 0)]['s1_cut_acceptances']
        
        # now check s1_cuts
        a_num_events_s1_cuts[j] = np.sum(np.random.binomial(1, current_df['s1_cut_acceptances']))

        # now check s2_cuts
        a_num_events_s2_cuts[j] = np.sum(np.random.binomial(1, current_df['s2_cut_acceptances']))

        # now check pax
        a_num_events_pax[j] = np.sum(np.random.binomial(1, current_df['pax_acceptances']))

        # now check threshold
        a_num_events_threshold[j] = np.sum(np.random.binomial(1, current_df['threshold_acceptances']))
        
        # now check total minus s1 threshold
        a_num_events_total_minus_threshold[j] = np.sum(np.random.binomial(1, current_df['s1_cut_acceptances']*current_df['s2_cut_acceptances']*current_df['pax_acceptances']))

        # now check total
        a_num_events_total[j] = np.sum(np.random.binomial(1, current_df['s1_cut_acceptances']*current_df['s2_cut_acceptances']*current_df['pax_acceptances']*current_df['threshold_acceptances']))

    d_arrays['s1_cut_only']['a_lb'][i], d_arrays['s1_cut_only']['a_prob'][i], d_arrays['s1_cut_only']['a_ub'][i] = np.percentile(a_num_events_s1_cuts / a_num_total_events, l_percentiles)
    d_arrays['s2_cut_only']['a_lb'][i], d_arrays['s2_cut_only']['a_prob'][i], d_arrays['s2_cut_only']['a_ub'][i] = np.percentile(a_num_events_s2_cuts / a_num_total_events, l_percentiles)
    d_arrays['pax_only']['a_lb'][i], d_arrays['pax_only']['a_prob'][i], d_arrays['pax_only']['a_ub'][i] = np.percentile(a_num_events_pax / a_num_total_events, l_percentiles)
    d_arrays['threshold_only']['a_lb'][i], d_arrays['threshold_only']['a_prob'][i], d_arrays['threshold_only']['a_ub'][i] = np.percentile(a_num_events_threshold / a_num_total_events, l_percentiles)
    d_arrays['total_minus_threshold']['a_lb'][i], d_arrays['total_minus_threshold']['a_prob'][i], d_arrays['total_minus_threshold']['a_ub'][i] = np.percentile(a_num_events_total_minus_threshold / a_num_total_events, l_percentiles)
    d_arrays['total']['a_lb'][i], d_arrays['total']['a_prob'][i], d_arrays['total']['a_ub'][i] = np.percentile(a_num_events_total / a_num_total_events, l_percentiles)



fig_acceptances, ax_acceptances = plt.subplots(1)

for label in l_keys:
    ax_acceptances.plot(d_arrays[label]['bin_centers'], d_arrays[label]['a_prob'], color=d_arrays[label]['color'], linestyle=d_arrays[label]['linestyle'], label=d_arrays[label]['label'])
    ax_acceptances.fill_between(d_arrays[label]['bin_centers'], d_arrays[label]['a_lb'], d_arrays[label]['a_ub'], color=d_arrays[label]['color'], alpha=0.1)

ax_acceptances.set_title(r'SR0 Signal Efficiencies')
ax_acceptances.set_xlabel(r'Energy [keV]')
ax_acceptances.set_ylabel(r'Signal Efficiency')


ax_acceptances.set_xlim(bin_edges_energy[0], bin_edges_energy[-1])
ax_acceptances.set_ylim(0, 1.05)

ax_acceptances.legend(loc='best', fontsize=10)
#ax_acceptances.set_xscale('log')
#ax_acceptances.set_yscale('log')

fig_acceptances.savefig('%ssignal_efficiencies_%.3f_kV.png' % (s_path_for_save, cathode_setting))

#print d_main['cs2']

pickle.dump(d_arrays, open('%ssignal_efficiency_bands_dict_%.3f_kV.p' % (s_path_for_save, cathode_setting), 'w'))

#plt.show()



