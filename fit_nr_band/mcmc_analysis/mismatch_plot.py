#!/usr/bin/python

import sys
import neriX_analysis
import os, re
import random

sys.path.insert(0, '..')
import config_xe1t, nr_analysis_xe1t

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

import pycuda.driver as drv

cathode_setting = config_xe1t.l_allowed_cathode_settings[0]
degree_setting = config_xe1t.l_allowed_degree_settings[0]

l_s1_cuts = [0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100]
transparency = 0.2
l_quantiles_for_lines_in_plot = [1, 5, 50]

# num_steps_to_include is how large of sampler you keep
num_steps_to_include = 1
num_mc_events = int(1e6)
device_number = 0

if(len(sys.argv) != 2):
	print 'Usage is python compare_data_fit.py <num walkers>'
	sys.exit(1)

num_walkers = int(sys.argv[1])

l_s1_settings = config_xe1t.l_s1_settings
l_log_settings = config_xe1t.l_log_settings
l_s2_settings = config_xe1t.l_s2_settings

mc_bin_number_multiplier = 2
fig_size_2d = (11, 13)
figure_sizes = (11, 2.0*(len(l_s1_cuts)-1))
d_subplot_space_2d = {'wspace':0.2, 'hspace':0.3}
d_subplot_space = {'wspace':0.2, 'hspace':0.6}

s_identifier = 'sb_ms'
if s_identifier == 'sb':
    s_source = 'ambe'
elif s_identifier == 'sbf':
    s_source = 'ambe_f'
elif s_identifier == 'sb_ms':
    s_source = 'ambe_ms'

current_analysis = nr_analysis_xe1t.nr_analysis_xe1t(s_source, 'lax_0.11.1', num_mc_events, num_walkers, num_steps_to_include)

d_data_information = {}

# get s1 and s2 data arrays
d_data_information['d_s1_s2_data'] = pickle.load(open('%sambe_data.p' % (config_xe1t.path_to_fit_inputs), 'r'))
d_data_information['a_s1_data'] = d_data_information['d_s1_s2_data']['s1']
d_data_information['a_s2_data'] = d_data_information['d_s1_s2_data']['s2']

# get number of data pts
d_data_information['num_data_pts'] = len(d_data_information['a_s1_data'])


d_plotting_information = current_analysis.prepare_gpu()
#print d_plotting_information['gpu_s1pf_lb_acc']


# create figure and give settings for space
fig_s1_log, (ax_s1_projection, ax_2d_comparison) = plt.subplots(2, sharex=False, sharey=False, figsize=fig_size_2d)
fig_s1_log.subplots_adjust(**d_subplot_space_2d)


# create 2d histogram of data
hist_s1_log_data, bin_edges_s1, bin_edges_log = np.histogram2d(d_data_information['a_s1_data'], np.log10(d_data_information['a_s2_data']/d_data_information['a_s1_data']), bins=[l_s1_settings[0], l_log_settings[0]], range=[[l_s1_settings[1], l_s1_settings[2]], [l_log_settings[1], l_log_settings[2]]])





# -----------------------------------------------
# -----------------------------------------------
# run MC
# -----------------------------------------------
# -----------------------------------------------



num_trials = np.asarray(num_mc_events, dtype=np.int32)
mean_field = np.asarray(config_xe1t.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)

d_sampler_values = current_analysis.get_best_fit_parameters()

scale_factor = current_analysis.get_scale_factor()

#print 'fixing acceptance to zero'
#d_sampler_values['acceptance_par'] = np.asarray(0, dtype=np.float32)

a_s1_mc = np.full(num_mc_events, -1, dtype=np.float32)
a_s2_mc = np.full(num_mc_events, -2, dtype=np.float32)
a_weights = np.full(num_mc_events, 0, dtype=np.float32)


if s_identifier == 'sb':
    t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_sampler_values['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_sampler_values['w_value']), drv.In(d_sampler_values['alpha']), drv.In(d_sampler_values['zeta']), drv.In(d_sampler_values['beta']), drv.In(d_sampler_values['gamma']), drv.In(d_sampler_values['delta']), drv.In(d_sampler_values['kappa']), drv.In(d_sampler_values['eta']), drv.In(d_sampler_values['lamb']), drv.In(d_sampler_values['g1_value']), drv.In(d_sampler_values['extraction_efficiency']), drv.In(d_sampler_values['gas_gain_value']), drv.In(d_sampler_values['gas_gain_width']), drv.In(d_sampler_values['dpe_prob']), drv.In(d_sampler_values['s1_bias_par']), drv.In(d_sampler_values['s1_smearing_par']), drv.In(d_sampler_values['s2_bias_par']), drv.In(d_sampler_values['s2_smearing_par']), drv.In(d_sampler_values['acceptance_par']), drv.In(d_sampler_values['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s1pf']), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(d_sampler_values['current_cut_acceptance_s1_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s1_slope']), drv.In(d_sampler_values['current_cut_acceptance_s2_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s2_slope']), drv.In(d_sampler_values['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_sampler_values['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_sampler_values['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_sampler_values['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weights))
elif s_identifier == 'sbf':
    t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_sampler_values['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_sampler_values['w_value']), drv.In(d_sampler_values['alpha']), drv.In(d_sampler_values['zeta']), drv.In(d_sampler_values['beta']), drv.In(d_sampler_values['gamma']), drv.In(d_sampler_values['delta']), drv.In(d_sampler_values['kappa']), drv.In(d_sampler_values['eta']), drv.In(d_sampler_values['lamb']), drv.In(d_sampler_values['g1_value']), drv.In(d_sampler_values['extraction_efficiency']), drv.In(d_sampler_values['gas_gain_value']), drv.In(d_sampler_values['gas_gain_width']), drv.In(d_sampler_values['dpe_prob']), drv.In(d_sampler_values['s1_bias_par']), drv.In(d_sampler_values['s1_smearing_par']), drv.In(d_sampler_values['s2_bias_par']), drv.In(d_sampler_values['s2_smearing_par']), drv.In(d_sampler_values['acceptance_par_0']), drv.In(d_sampler_values['acceptance_par_1']), drv.In(d_sampler_values['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_sampler_values['current_cut_acceptance_s1_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s1_slope']), drv.In(d_sampler_values['current_cut_acceptance_s2_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s2_slope']), drv.In(d_sampler_values['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_sampler_values['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_sampler_values['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_sampler_values['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weights))
elif s_identifier == 'sb_ms':
    t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_sampler_values['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_sampler_values['w_value']), drv.In(d_sampler_values['alpha']), drv.In(d_sampler_values['zeta']), drv.In(d_sampler_values['beta']), drv.In(d_sampler_values['gamma']), drv.In(d_sampler_values['delta']), drv.In(d_sampler_values['kappa']), drv.In(d_sampler_values['eta']), drv.In(d_sampler_values['lamb']), drv.In(d_sampler_values['g1_value']), drv.In(d_sampler_values['extraction_efficiency']), drv.In(d_sampler_values['gas_gain_value']), drv.In(d_sampler_values['gas_gain_width']), drv.In(d_sampler_values['dpe_prob']), drv.In(d_sampler_values['s1_bias_par']), drv.In(d_sampler_values['s1_smearing_par']), drv.In(d_sampler_values['s2_bias_par']), drv.In(d_sampler_values['s2_smearing_par']), drv.In(d_sampler_values['acceptance_par']), drv.In(d_sampler_values['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s1pf']), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(d_sampler_values['current_cut_acceptance_s1_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s1_slope']), drv.In(d_sampler_values['current_cut_acceptance_s2_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s2_slope']), drv.In(d_sampler_values['ms_par_0']), drv.In(d_sampler_values['ms_par_1']), drv.In(d_sampler_values['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_sampler_values['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_sampler_values['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_sampler_values['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weights))

current_analysis.call_gpu_func(t_args)


hist_s1_log_mc, _, _ = np.histogram2d(a_s1_mc, np.log10(a_s2_mc/a_s1_mc), bins=[l_s1_settings[0], l_log_settings[0]], range=[[l_s1_settings[1], l_s1_settings[2]], [l_log_settings[1], l_log_settings[2]]], weights=a_weights)


hist_s1_log_mc *= d_sampler_values['scale_par']*d_data_information['num_data_pts']/float(num_mc_events)


# make histogram showing Pearson Chi2 stat (with sign)
# for each bin
hist_comparison = np.zeros(hist_s1_log_mc.shape)
for index, value in np.ndenumerate(hist_s1_log_data):
    if value <= 0:
        continue

    hist_comparison[index] = np.sign(hist_s1_log_data[index] - hist_s1_log_mc[index]) * (hist_s1_log_data[index] - hist_s1_log_mc[index])**2. / hist_s1_log_data[index]


a_s1_hist_data = np.sum(hist_s1_log_data, axis=1)
a_s1_hist_mc = np.sum(hist_s1_log_mc, axis=1)
a_s1_comparison = np.zeros(a_s1_hist_data.shape)

for index in xrange(len(a_s1_hist_data)):
    if a_s1_hist_data[index] == 0:
        continue

    a_s1_comparison[index] = np.sign(a_s1_hist_data[index] - a_s1_hist_mc[index]) * (a_s1_hist_data[index] - a_s1_hist_mc[index])**2. / a_s1_hist_data[index]



pcolor_cax = ax_2d_comparison.pcolormesh(bin_edges_s1, bin_edges_log, hist_comparison.T, vmin=-np.max(hist_comparison)*1.1, vmax=np.max(hist_comparison)*1.1, cmap='bwr')
#cbar_ax = fig_s1_log.add_axes([0.85, 0.15, 0.05, 0.7])
fig_s1_log.colorbar(pcolor_cax, ax=ax_2d_comparison)


ax_s1_projection.plot((bin_edges_s1[1:]+bin_edges_s1[:-1])/2., a_s1_comparison, linestyle='', color='black', marker='o')

ax_s1_projection.set_xlabel('S1 [PE]')
ax_s1_projection.set_ylabel('Signed Residual')
ax_s1_projection.set_ylim(-5, 20)

ax_2d_comparison.set_xlabel('S1 [PE]')
ax_2d_comparison.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')

plt.show()

fig_s1_log.savefig('%ss_data_mc_mismatch_%s.png' % (current_analysis.get_path_to_plots(), current_analysis.get_save_name_beginning()))



# end GPU context
current_analysis.end_gpu_context()




#plt.show()



