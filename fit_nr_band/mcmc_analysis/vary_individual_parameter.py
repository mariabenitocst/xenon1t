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

import nr_analysis_xe1t

from math import floor

import scipy.optimize as op

import cPickle as pickle

import pycuda.driver as drv

# -----------------------------------------
#  Setup and initialization
# -----------------------------------------

l_grid_parameters = ['kappa']
num_mc_events = int(2e6)
num_loops = 4

d_parameter_to_index = {'kappa':6,
                        'alpha':1,
                        'gamma':4,
                        'w_value':0,
                        'zeta':2,
                        'delta':5,
                        'eta':7,
                        'lamb':8,
                        'g1_value':9,
                        'extraction_efficiency':10,
                        'acceptance_par':18}

cathode_setting = config_xe1t.l_allowed_cathode_settings[0]
degree_setting = config_xe1t.l_allowed_degree_settings[0]

s_identifier = 'sb_ms'

if s_identifier == 'sb':
    num_dim = 22
    l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'cut_acceptance_par'] + ['prob_bkg', 'scale_par']
elif s_identifier == 'sbf':
    num_dim = 23
    l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par_0', 'acceptance_par_1', 'cut_acceptance_par'] + ['prob_bkg', 'scale_par']
elif s_identifier == 'sb_ms':
    num_dim = 24
    l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lamb', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'cut_acceptance_par', 'ms_par_0', 'ms_par_1'] + ['prob_bkg', 'scale_par']
    s_source = 'ambe_ms'

l_sigma_levels = [-2, -1, 0, 1, 2]
assert 0 in l_sigma_levels

d_sigma_to_percentile = {-10:0.00001, -2:2.5, -1:16, 0:50, 1:84, 2:97.5, 10:99.999999}



assert len(l_grid_parameters) == 1
par_to_examine = l_grid_parameters[0]

# num_steps_to_include is how large of sampler you keep
num_steps_to_include = 1000

figure_sizes = (11, 13)

# get s1 and s2 data arrays
d_data_information = {}
d_data_information['d_s1_s2_data'] = pickle.load(open('%sambe_data.p' % (config_xe1t.path_to_fit_inputs), 'r'))
d_data_information['a_s1_data'] = d_data_information['d_s1_s2_data']['s1']
d_data_information['a_s2_data'] = d_data_information['d_s1_s2_data']['s2']

# get number of data pts
d_data_information['num_data_pts'] = len(d_data_information['a_s1_data'])


if(len(sys.argv) != 2):
	print 'Usage is python compare_data_fit.py <num walkers>'
	sys.exit(1)

num_walkers = int(sys.argv[1])

current_analysis = nr_analysis_xe1t.nr_analysis_xe1t(s_source, 'lax_0.11.1', num_mc_events, num_walkers, num_steps_to_include)
d_plotting_information = current_analysis.prepare_gpu()

# -----------------------------------------
#  Find maxima for each parameter
#  individually
# -----------------------------------------

d_grid_parameter_percentiles = {}

a_partial_sampler = current_analysis.get_mcmc_samples()

for par_name in l_grid_parameters:
    d_grid_parameter_percentiles[par_name] = {}
    
    for sigma_level in l_sigma_levels:
        d_grid_parameter_percentiles[par_name][sigma_level] = np.percentile(a_partial_sampler[:, d_parameter_to_index[par_name]], [d_sigma_to_percentile[sigma_level]])




# -----------------------------------------------
# -----------------------------------------------
# run MC
# -----------------------------------------------
# -----------------------------------------------


num_trials = np.asarray(num_mc_events, dtype=np.int32)
mean_field = np.asarray(config_xe1t.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)
d_sampler_values = current_analysis.get_best_fit_parameters()

a_s1_bin_edges = np.linspace(config_xe1t.l_s1_settings[1], config_xe1t.l_s1_settings[2], config_xe1t.l_s1_settings[0]+1, dtype=np.float32)
a_log_bin_edges = config_xe1t.a_log_bin_edges

d_grid_parameter_histograms = {}
for par_name in l_par_names:
    d_grid_parameter_histograms[par_name] = {}

for sigma_level in l_sigma_levels:
    d_grid_parameter_histograms[par_name][sigma_level] = {}


    
    d_parameters_for_gpu = {}
    
    for i, par_name in enumerate(l_par_names):
        if (par_name == par_to_examine):
            print sigma_level
            print d_sampler_values[par_name]
            d_sampler_values[par_name] = np.asarray(d_grid_parameter_percentiles[par_name][sigma_level], dtype=np.float32)
            print d_sampler_values[par_name]
    


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

    d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'], _, _ = np.histogram2d(a_s1_mc, np.log10(a_s2_mc/a_s1_mc), bins=[config_xe1t.l_s1_settings[0], config_xe1t.l_log_settings[0]], range=[[config_xe1t.l_s1_settings[1], config_xe1t.l_s1_settings[2]], [config_xe1t.l_log_settings[1], config_xe1t.l_log_settings[2]]], weights=a_weights)

    d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'] = d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'].T


    sum_mc = np.sum(d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'], dtype=np.float32)
    if sum_mc == 0:
        print 'Sum was zero!'
        ctx.pop()
        sys.exit()

    d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'] *= d_data_information['num_data_pts']*current_analysis.get_scale_factor()


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

    #l_ax_grid[count_ax].pcolormesh(a_s1_bin_edges, a_log_bin_edges, d_grid_parameter_histograms[par_name][sigma_level]['mc_hist'], cmap='bwr')
    pcolor_cax = l_ax_grid[count_ax].pcolormesh(a_s1_bin_edges, a_log_bin_edges, current_hist, vmin=-0.8, vmax=0.8, cmap='bwr')

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




    
fig_grid.savefig('%sparameter_variations/%s_variation.png' % (current_analysis.get_path_to_plots(), par_to_examine))



# end GPU context
current_analysis.end_gpu_context()



#plt.show()



