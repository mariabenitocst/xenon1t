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



l_s1_cuts = [0, 10, 40, 100]
transparency = 0.2
l_quantiles_for_2d = [50]

# num_steps goes into how many random draws you will take from kept sampler
num_steps = 10
# num_steps_to_include is how large of sampler you keep
num_steps_to_include = 1000
num_mc_events = int(1e6)
num_loops = 10

if(len(sys.argv) != 2):
	print 'Usage is python compare_data_fit.py <num walkers>'
	sys.exit(1)

num_walkers = int(sys.argv[1])

fiducial_volume_mass = 1000. # kg
roi_lb_cut_s1 = 3.
print '\n\nGiving rate assuming %0.f kg FV\n\n\n' % (fiducial_volume_mass)

lax_version = 'lax_0.9.2'
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
dev = drv.Device(1)
ctx = dev.make_context()
print 'Device Name: %s\n' % (dev.name())


d_mc_energy = pickle.load(open('%swimp_mc.p' % config_xe1t.path_to_fit_inputs, 'r'))
d_er_band = pickle.load(open('%ser_band.p' % config_xe1t.path_to_fit_inputs, 'r'))

random.seed()
    
    
f_rn = root.TFile('./radiogenic_neutrons_spectra/RadiogenicNeutrons_SR0.root')
h_energy = f_rn.hEdTotal_1tFVcylSR0
h_pos = f_rn.h2dTotal_1tFVcylSR0

# ['energy_kev', 'events/ton/yr/keV']
    
# -----------------------------------------
#  Energy
# -----------------------------------------

# from script in Marco's email
expected_rate = h_energy.Integral(1, 300)*365*fiducial_volume_mass # events/yr


a_mc_energy = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_random_num = h_energy.GetRandom()
    
    a_mc_energy[i] = current_random_num



# -----------------------------------------
#  X, Y, Z
# -----------------------------------------

a_mc_x = np.zeros(num_mc_events, dtype=np.float32)
a_mc_y = np.zeros(num_mc_events, dtype=np.float32)
a_mc_z = np.zeros(num_mc_events, dtype=np.float32)
for i in tqdm.tqdm(xrange(num_mc_events)):
    current_r = np.asarray(1e50, dtype=np.float64)
    current_z = np.asarray(0, dtype=np.float64)
    current_phi = np.random.uniform(0, 2*np.pi)
    
    h_pos.GetRandom2(current_r, current_z)
    # given in r^2 so must correct
    current_r = current_r**0.5

    current_random_num_x = current_r * np.cos(current_phi)
    current_random_num_y = current_r * np.sin(current_phi)
    
    # must divide by 10 to go from mm -> cm
    a_mc_x[i] = current_random_num_x/10.
    a_mc_y[i] = current_random_num_y/10.
    a_mc_z[i] = current_z/10.



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


d_cut_acceptances = pickle.load(open('%scut_acceptances.p' % (config_xe1t.path_to_fit_inputs), 'r'))
# PF acceptance
#print '\n\nTesting with 100% acceptance\n\n'
#d_cut_acceptances['s1']['acceptance'] = [1 for i in xrange(len(d_cut_acceptances['s1']['spline_pts']))]
#d_cut_acceptances['s2']['acceptance'] = [1 for i in xrange(len(d_cut_acceptances['s2']['spline_pts']))]

a_s1cuts_s1s = np.asarray(d_cut_acceptances['s1']['spline_pts'], dtype=np.float32)
a_s1cuts_mean_acc = np.asarray(d_cut_acceptances['s1']['acceptance'], dtype=np.float32)

a_s2cuts_s2s = np.asarray(d_cut_acceptances['s2']['spline_pts'], dtype=np.float32)
a_s2cuts_mean_acc = np.asarray(d_cut_acceptances['s2']['acceptance'], dtype=np.float32)




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
gpu_observables_func = SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production_with_log_hist_no_fv')
gpu_observables_func_arrays = SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production_with_arrays_no_fv')



# get s1 and s2 data arrays
d_plotting_information['d_s1_s2_data'] = pickle.load(open('%sambe_data.p' % (config_xe1t.path_to_fit_inputs), 'r'))
d_plotting_information['a_s1_data'] = d_plotting_information['d_s1_s2_data']['s1']
d_plotting_information['a_s2_data'] = d_plotting_information['d_s1_s2_data']['s2']

# get number of data pts
d_plotting_information['num_data_pts'] = len(d_plotting_information['a_s1_data'])


# create figure and give settings for space
fig_s1_log, ax_s1_log_mc = plt.subplots(1, sharex=False, sharey=False, figsize=figure_sizes)
fig_s1_log.subplots_adjust(**d_subplot_space)






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

f_constant = root.TF2('constant', '0.0001', bin_edges_s1_th2[0], bin_edges_s1_th2[-1], bin_edges_s2_th2[0], bin_edges_s2_th2[-1])
f_hist = root.TFile('%sradiogenic_neutrons_bkg_%s.root' % (s_path_for_save, lax_version), 'RECREATE')

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

    print 'Fixing bkg to 0'
    prob_bkg = np.asarray(0, dtype=np.float32)
    #prob_bkg = np.asarray(a_fit_parameters[19], dtype=np.float32)
    scale_par = np.asarray(1, dtype=np.float32)
    
    
    
    
    num_pts_s1bs = np.asarray(len(a_s1bs_s1s), dtype=np.int32)
    num_pts_s2bs = np.asarray(len(a_s2bs_s2s), dtype=np.int32)
    num_pts_s1pf = np.asarray(len(a_s1pf_s1s), dtype=np.int32)
    num_pts_s1cuts = np.asarray(len(a_s1cuts_s1s), dtype=np.int32)
    num_pts_s2cuts = np.asarray(len(a_s2cuts_s2s), dtype=np.int32)
    
    num_bins_r2 = np.asarray(len(bin_edges_r2)-1, dtype=np.int32)
    num_bins_z = np.asarray(len(bin_edges_z)-1, dtype=np.int32)
    num_bins_x = np.asarray(len(bin_edges_x)-1, dtype=np.int32)
    num_bins_y = np.asarray(len(bin_edges_y)-1, dtype=np.int32)
    


    num_bins_s1 = np.asarray(l_s1_settings[0], dtype=np.int32)
    num_bins_log = np.asarray(l_log_settings[0], dtype=np.int32)
    
    gpu_s1_edges = np.asarray(s1_edges, np.float32)
    gpu_log_edges = np.asarray(log_edges, np.float32)
    
    num_loops = np.asarray(num_loops, dtype=np.int32)
    
    a_hist_2d = np.zeros(int(num_bins_s1)*int(num_bins_log), dtype=np.float32)
    
    
    tArgs = (local_rng_states, drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(prob_bkg), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(dpe_prob), drv.In(s1_bias_par), drv.In(s1_smearing_par), drv.In(s2_bias_par), drv.In(s2_smearing_par), drv.In(acceptance_par), drv.In(num_pts_s1bs), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(num_pts_s1pf), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(num_pts_s1cuts), d_plotting_information['gpu_s1cuts_s1s'], d_plotting_information['gpu_s1cuts_mean_acc'], drv.In(num_pts_s2cuts), d_plotting_information['gpu_s2cuts_s2s'], d_plotting_information['gpu_s2cuts_mean_acc'], drv.In(num_bins_r2), d_plotting_information['gpu_bin_edges_r2'], drv.In(num_bins_z), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(num_bins_x), d_plotting_information['gpu_bin_edges_x'], drv.In(num_bins_y), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.In(num_bins_s1), drv.In(gpu_s1_edges), drv.In(num_bins_log), drv.In(gpu_log_edges), drv.InOut(a_hist_2d), drv.In(num_loops))

    gpu_observables_func(*tArgs, **d_gpu_scale)


    scale_factor = float(expected_rate) / float(num_mc_events*num_loops)

    scale_factor_arrays = scale_factor*num_loops

    # now use arrays function
    
    a_s1_mc = np.full(num_mc_events, -1, dtype=np.float32)
    a_s2_mc = np.full(num_mc_events, -1, dtype=np.float32)

    tArgs = (local_rng_states, drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(prob_bkg), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(w_value), drv.In(alpha), drv.In(zeta), drv.In(beta), drv.In(gamma), drv.In(delta), drv.In(kappa), drv.In(eta), drv.In(lamb), drv.In(g1_value), drv.In(extraction_efficiency), drv.In(gas_gain_value), drv.In(gas_gain_width), drv.In(dpe_prob), drv.In(s1_bias_par), drv.In(s1_smearing_par), drv.In(s2_bias_par), drv.In(s2_smearing_par), drv.In(acceptance_par), drv.In(num_pts_s1bs), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(num_pts_s2bs), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(num_pts_s1pf), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(num_pts_s1cuts), d_plotting_information['gpu_s1cuts_s1s'], d_plotting_information['gpu_s1cuts_mean_acc'], drv.In(num_pts_s2cuts), d_plotting_information['gpu_s2cuts_s2s'], d_plotting_information['gpu_s2cuts_mean_acc'], drv.In(num_bins_r2), d_plotting_information['gpu_bin_edges_r2'], drv.In(num_bins_z), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(num_bins_x), d_plotting_information['gpu_bin_edges_x'], drv.In(num_bins_y), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc))
    

    gpu_observables_func_arrays(*tArgs, **d_gpu_scale)
    
    s_key = 'radiogenic_neutrons_bkg'
    h_current = root.TH2F(s_key, s_key, config_xe1t.l_s1_settings_pl[0], bin_edges_s1_th2, config_xe1t.l_s2_settings_pl[0], bin_edges_s2_th2)
    h_current.Add(f_constant)
    
    count_events = 0

    for i in xrange(len(a_s1_mc)):
        if (not np.isnan(a_s1_mc[i])) and (not np.isnan(a_s2_mc[i])) and (a_s2_mc[i] > 0):
            if (a_s1_mc[i] > bin_edges_s1_th2[0] and a_s1_mc[i] < bin_edges_s1_th2[-1] and np.log10(a_s2_mc[i]) > bin_edges_s2_th2[0] and np.log10(a_s2_mc[i]) < bin_edges_s2_th2[-1]):
                h_current.Fill(a_s1_mc[i], np.log10(a_s2_mc[i]))
                count_events += 1

    print '\nEvents per year: %f\n' % (count_events*scale_factor_arrays)

    # histogram should be in events per day
    h_current.Scale(scale_factor_arrays/365.)
    h_current.Write()







a_hist_2d = np.reshape(a_hist_2d, (int(num_bins_log), int(num_bins_s1))).T

for i, bin_edge in enumerate(s1_edges):
    # cut at 3 PE for dark matter range
    if bin_edge >= roi_lb_cut_s1:
        break

cut_bin = i
assert cut_bin >= 0


#print cut_bin
#print s1_edges
#print a_hist_2d.shape


scale_factor = float(expected_rate) / float(num_mc_events*num_loops)


df_data = pd.DataFrame({'s1':d_plotting_information['a_s1_data'], 's2':d_plotting_information['a_s2_data']})
df_data['log'] = np.log10(df_data['s2']/df_data['s1'])


#print scale_factor
a_hist_2d = np.multiply(a_hist_2d, scale_factor)
#print a_hist_2d
a_roi_hist2d = a_hist_2d[cut_bin:, :]

cax_s1_log_mc = ax_s1_log_mc.pcolor(s1_edges, log_edges, a_hist_2d.T, cmap='Blues')
ax_s1_log_mc.axvline(x=roi_lb_cut_s1, color='r', linestyle='--')
fig_s1_log.colorbar(cax_s1_log_mc, label=r'$\frac{Events}{Year}$', format='%.2e')





ax_s1_log_mc.set_title(r'Radiogenic Neutron Background - $V_c$ = %.3f kV' % (cathode_setting), fontsize=12)
ax_s1_log_mc.text(0.45, 0.85, r'$\mathrm{Integrated\ Rate} = %.2e \ \frac{\mathrm{Events}}{\mathrm{Year}}$' % (np.sum(a_roi_hist2d)), transform=ax_s1_log_mc.transAxes, fontsize=14)
ax_s1_log_mc.set_xlim(s1_edges[0], s1_edges[-1])
ax_s1_log_mc.set_xlabel('S1 [PE]')
ax_s1_log_mc.set_ylabel(r'$log_{10}(\frac{S2}{S1})$')






    
f_hist.Close()

fig_s1_log.savefig('%sradiogenic_neutrons_%.3f_kV.png' % (s_path_for_save, cathode_setting))






# end GPU context
ctx.pop()
print 'Successfully ended GPU context!'





#plt.show()



