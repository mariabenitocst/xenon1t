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
num_steps_to_include = 10
num_mc_events = int(4e5)
device_number = 0
b_save_array = False

s_identifier = 'sb_ms'
if s_identifier == 'sb':
    s_source = 'ambe'
elif s_identifier == 'sbf':
    s_source = 'ambe_f'
elif s_identifier == 'sb_ms':
    s_source = 'ambe_ms'

print '\n\nMaking plots assuming identifier %s\n\n' % (s_identifier)

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
fig_s1_log, (ax_s1_projection, ax_s1_log_data, ax_s1_log_mc) = plt.subplots(3, sharex=False, sharey=False, figsize=fig_size_2d)
fig_s1_log.subplots_adjust(**d_subplot_space_2d)


# create 2d histogram of data
hist_s1_log_data = ax_s1_log_data.hist2d(d_data_information['a_s1_data'], np.log10(d_data_information['a_s2_data']/d_data_information['a_s1_data']), bins=[l_s1_settings[0], l_log_settings[0]], range=[[l_s1_settings[1], l_s1_settings[2]], [l_log_settings[1], l_log_settings[2]]], cmap='Blues')
fig_s1_log.colorbar(hist_s1_log_data[3], ax=ax_s1_log_data)





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


hist_s1_log_mc = ax_s1_log_mc.hist2d(a_s1_mc, np.log10(a_s2_mc/a_s1_mc), bins=[l_s1_settings[0], l_log_settings[0]], range=[[l_s1_settings[1], l_s1_settings[2]], [l_log_settings[1], l_log_settings[2]]], cmap='Blues', weights=a_weights)
fig_s1_log.colorbar(hist_s1_log_mc[3], ax=ax_s1_log_mc)


df_data = pd.DataFrame({'s1':d_data_information['a_s1_data'], 's2':d_data_information['a_s2_data']})
df_data['log'] = np.log10(df_data['s2']/df_data['s1'])

df_mc = pd.DataFrame({'s1':a_s1_mc, 's2':a_s2_mc})
df_mc['log'] = np.log10(df_mc['s2']/df_mc['s1'])

if b_save_array:
    d_mc = {}
    d_mc['s1'] = a_s1_mc
    d_mc['s2'] = a_s2_mc
    d_mc['weight'] = a_weights
    pickle.dump(d_mc, open('./mc_output/mc_ambe_arrays.p', 'w'))
    del d_mc

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
df_histograms['s1'] = np.zeros((num_walkers*num_steps_to_include, len(s1_edges_mc)-1), dtype=float)

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
    
    df_histograms['s2'][current_set_s1_cuts] = np.zeros((num_walkers*num_steps_to_include, len(s1_edges_mc)-1), dtype=float)
    df_histograms['log'][current_set_s1_cuts] = np.zeros((num_walkers*num_steps_to_include, len(s1_edges_mc)-1), dtype=float)
    
    
    # get data slices
    cut_df = df_data[(df_data['s1'] > l_s1_cuts[s1_cut_num]) & (df_data['s1'] < l_s1_cuts[s1_cut_num+1])]
    cut_df_mc = df_mc[(df_mc['s1'] > l_s1_cuts[s1_cut_num]) & (df_mc['s1'] < l_s1_cuts[s1_cut_num+1])]
    
    for current_quantile in l_quantiles_for_lines_in_plot:
        d_quantile_lines['s2'][current_set_s1_cuts][current_quantile] = np.percentile(cut_df_mc['s2'], current_quantile)
        d_quantile_lines['log'][current_set_s1_cuts][current_quantile] = np.percentile(cut_df_mc['log'], current_quantile)
    
    d_s2_data_slices[current_set_s1_cuts] = cut_df['s2']
    d_log_data_slices[current_set_s1_cuts] = cut_df['log']



print '\nStarting bands in S1 and S2\n'
l_dfs = [0 for i in xrange(num_walkers*num_steps_to_include)]

for i, d_sampler_values in tqdm.tqdm(enumerate(current_analysis.yield_unfixed_parameters())):
    # create dictionary to hold relevant information
    l_dfs[i] = {}

    #print 'fixing acceptance to zero'
    #d_sampler_values['acceptance_par'] = np.asarray(0, dtype=np.float32)

    a_s1_mc = np.full(num_mc_events, -10, dtype=np.float32)
    a_s2_mc = np.full(num_mc_events, -20, dtype=np.float32)
    a_weights = np.full(num_mc_events, 0, dtype=np.float32)

    if s_identifier == 'sb':
        t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_sampler_values['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_sampler_values['w_value']), drv.In(d_sampler_values['alpha']), drv.In(d_sampler_values['zeta']), drv.In(d_sampler_values['beta']), drv.In(d_sampler_values['gamma']), drv.In(d_sampler_values['delta']), drv.In(d_sampler_values['kappa']), drv.In(d_sampler_values['eta']), drv.In(d_sampler_values['lamb']), drv.In(d_sampler_values['g1_value']), drv.In(d_sampler_values['extraction_efficiency']), drv.In(d_sampler_values['gas_gain_value']), drv.In(d_sampler_values['gas_gain_width']), drv.In(d_sampler_values['dpe_prob']), drv.In(d_sampler_values['s1_bias_par']), drv.In(d_sampler_values['s1_smearing_par']), drv.In(d_sampler_values['s2_bias_par']), drv.In(d_sampler_values['s2_smearing_par']), drv.In(d_sampler_values['acceptance_par']), drv.In(d_sampler_values['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s1pf']), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(d_sampler_values['current_cut_acceptance_s1_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s1_slope']), drv.In(d_sampler_values['current_cut_acceptance_s2_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s2_slope']), drv.In(d_sampler_values['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_sampler_values['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_sampler_values['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_sampler_values['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weights))
    elif s_identifier == 'sbf':
        t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_sampler_values['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_sampler_values['w_value']), drv.In(d_sampler_values['alpha']), drv.In(d_sampler_values['zeta']), drv.In(d_sampler_values['beta']), drv.In(d_sampler_values['gamma']), drv.In(d_sampler_values['delta']), drv.In(d_sampler_values['kappa']), drv.In(d_sampler_values['eta']), drv.In(d_sampler_values['lamb']), drv.In(d_sampler_values['g1_value']), drv.In(d_sampler_values['extraction_efficiency']), drv.In(d_sampler_values['gas_gain_value']), drv.In(d_sampler_values['gas_gain_width']), drv.In(d_sampler_values['dpe_prob']), drv.In(d_sampler_values['s1_bias_par']), drv.In(d_sampler_values['s1_smearing_par']), drv.In(d_sampler_values['s2_bias_par']), drv.In(d_sampler_values['s2_smearing_par']), drv.In(d_sampler_values['acceptance_par_0']), drv.In(d_sampler_values['acceptance_par_1']), drv.In(d_sampler_values['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_sampler_values['current_cut_acceptance_s1_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s1_slope']), drv.In(d_sampler_values['current_cut_acceptance_s2_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s2_slope']), drv.In(d_sampler_values['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_sampler_values['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_sampler_values['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_sampler_values['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weights))
    elif s_identifier == 'sb_ms':
        t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_sampler_values['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_sampler_values['w_value']), drv.In(d_sampler_values['alpha']), drv.In(d_sampler_values['zeta']), drv.In(d_sampler_values['beta']), drv.In(d_sampler_values['gamma']), drv.In(d_sampler_values['delta']), drv.In(d_sampler_values['kappa']), drv.In(d_sampler_values['eta']), drv.In(d_sampler_values['lamb']), drv.In(d_sampler_values['g1_value']), drv.In(d_sampler_values['extraction_efficiency']), drv.In(d_sampler_values['gas_gain_value']), drv.In(d_sampler_values['gas_gain_width']), drv.In(d_sampler_values['dpe_prob']), drv.In(d_sampler_values['s1_bias_par']), drv.In(d_sampler_values['s1_smearing_par']), drv.In(d_sampler_values['s2_bias_par']), drv.In(d_sampler_values['s2_smearing_par']), drv.In(d_sampler_values['acceptance_par']), drv.In(d_sampler_values['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s1pf']), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(d_sampler_values['current_cut_acceptance_s1_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s1_slope']), drv.In(d_sampler_values['current_cut_acceptance_s2_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s2_slope']), drv.In(d_sampler_values['ms_par_0']), drv.In(d_sampler_values['ms_par_1']), drv.In(d_sampler_values['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_sampler_values['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_sampler_values['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_sampler_values['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weights))

    current_analysis.call_gpu_func(t_args)
    
    
    l_dfs[i]['main_df'] = pd.DataFrame({'s1':a_s1_mc, 's2':a_s2_mc, 'weights':a_weights})
    l_dfs[i]['main_df'] = l_dfs[i]['main_df'][(l_dfs[i]['main_df']['s1'] > 0) & (l_dfs[i]['main_df']['s2'] > 0) & (~l_dfs[i]['main_df']['s1'].isnull()) & (~l_dfs[i]['main_df']['s2'].isnull())]
    
    l_dfs[i]['main_df']['log'] = np.log10(l_dfs[i]['main_df']['s2']/l_dfs[i]['main_df']['s1'])
    
    l_dfs[i]['s1_hist'], _ = np.histogram(l_dfs[i]['main_df']['s1'], s1_edges_mc, weights=l_dfs[i]['main_df']['weights'])
    l_dfs[i]['s1_hist'] = np.asarray(l_dfs[i]['s1_hist'], dtype=float)
    
    #print l_dfs[i]['s1_hist']
    
    # scale_factor for histograms
    #print scale_par, d_plotting_information['num_data_pts'], float(np.sum(l_dfs[i]['s1_hist']))
    scaling_factor_for_histogram = d_sampler_values['scale_par']*d_data_information['num_data_pts']/float(num_mc_events)*mc_bin_number_multiplier
    
    l_dfs[i]['s1_hist'] *= scaling_factor_for_histogram
    
    df_histograms['s1'][i, :] = l_dfs[i]['s1_hist']
    #print 'new'
    #print df_histograms['s1'][i, :]
    #print l_dfs[i]['s1_hist']
    

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
        #print df_histograms['s1'][:, bin_number]
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










    
d_median_line = {}
d_median_line['s1'] = a_median_line_s1
d_median_line['data_log_s2_s1_median'] = a_median_line_data_log
d_median_line['mc_log_s2_s1_median'] = a_median_line_mc_log

pickle.dump(d_median_line, open('%snr_band_median_line.p' % (current_analysis.get_path_to_plots()), 'w'))


fig_s1_log.savefig('%ss_data_mc_comparison_s1_and_2d_%s.png' % (current_analysis.get_path_to_plots(), current_analysis.get_save_name_beginning()))
fig_s2s.savefig('%ss_data_mc_comparison_s2_slices_%s.png' % (current_analysis.get_path_to_plots(), current_analysis.get_save_name_beginning()))
fig_log.savefig('%ss_data_mc_comparison_log_slices_%s.png' % (current_analysis.get_path_to_plots(), current_analysis.get_save_name_beginning()))
fig_s2s_logy.savefig('%ss_data_mc_comparison_s2_slices_logy_%s.png' % (current_analysis.get_path_to_plots(), current_analysis.get_save_name_beginning()))
fig_log_logy.savefig('%ss_data_mc_comparison_log_slices_logy_%s.png' % (current_analysis.get_path_to_plots(), current_analysis.get_save_name_beginning()))






# end GPU context
current_analysis.end_gpu_context()




#plt.show()



