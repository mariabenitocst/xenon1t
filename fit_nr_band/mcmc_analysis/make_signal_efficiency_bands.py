#!/usr/bin/python
import ROOT as root

import sys
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

import pycuda.driver as drv

num_mc_events = int(2e6)
num_walkers = 256
num_steps_to_include = 1

# name notes for the saved pickle file
name_notes = ''

cathode_setting = config_xe1t.l_allowed_cathode_settings[0]
degree_setting = config_xe1t.l_allowed_degree_settings[0]

current_analysis = nr_analysis_xe1t.nr_analysis_xe1t('uniform_nr', 'lax_0.11.1', num_mc_events, num_walkers, num_steps_to_include, b_conservative_acceptance_posterior=True)

d_plotting_information = current_analysis.prepare_gpu()



# -----------------------------------------------
# -----------------------------------------------
# run MC
# -----------------------------------------------
# -----------------------------------------------


num_trials = np.asarray(num_mc_events, dtype=np.int32)
mean_field = np.asarray(config_xe1t.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)

l_dataframes = [0 for i in xrange(num_steps_to_include*num_walkers)]


def s1_threshold(cs1):
    if cs1 > 3. and cs1 < 70.:
        return 1.
    else:
        return 0.


for i, d_sampler_values in tqdm.tqdm(enumerate(current_analysis.yield_unfixed_parameters())):

    a_s1 = np.full(num_mc_events, -1, dtype=np.float32)
    a_s2 = np.full(num_mc_events, -2, dtype=np.float32)
    a_cs1 = np.full(num_mc_events, -1, dtype=np.float32)
    a_cs2 = np.full(num_mc_events, -2, dtype=np.float32)

    a_pax_acceptances = np.full(num_mc_events, -3, dtype=np.float32)
    a_s1_cut_acceptances = np.full(num_mc_events, -3, dtype=np.float32)
    a_s2_cut_acceptances = np.full(num_mc_events, -3, dtype=np.float32)
    a_threshold_acceptances = np.full(num_mc_events, -3, dtype=np.float32)

    t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_sampler_values['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_sampler_values['w_value']), drv.In(d_sampler_values['alpha']), drv.In(d_sampler_values['zeta']), drv.In(d_sampler_values['beta']), drv.In(d_sampler_values['gamma']), drv.In(d_sampler_values['delta']), drv.In(d_sampler_values['kappa']), drv.In(d_sampler_values['eta']), drv.In(d_sampler_values['lamb']), drv.In(d_sampler_values['g1_value']), drv.In(d_sampler_values['extraction_efficiency']), drv.In(d_sampler_values['gas_gain_value']), drv.In(d_sampler_values['gas_gain_width']), drv.In(d_sampler_values['dpe_prob']), drv.In(d_sampler_values['s1_bias_par']), drv.In(d_sampler_values['s1_smearing_par']), drv.In(d_sampler_values['s2_bias_par']), drv.In(d_sampler_values['s2_smearing_par']), drv.In(d_sampler_values['acceptance_par']), drv.In(d_sampler_values['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s1pf']), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(d_sampler_values['current_cut_acceptance_s1_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s1_slope']), drv.In(d_sampler_values['current_cut_acceptance_s2_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s2_slope']), drv.In(d_sampler_values['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_sampler_values['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_sampler_values['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_sampler_values['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1), drv.InOut(a_s2), drv.InOut(a_cs1), drv.InOut(a_cs2), drv.InOut(a_pax_acceptances), drv.InOut(a_s1_cut_acceptances), drv.InOut(a_s2_cut_acceptances), drv.InOut(a_threshold_acceptances))
    
    current_analysis.call_gpu_func(t_args)

    l_dataframes[i] = pd.DataFrame({'energy':current_analysis.get_mc_energies(), 's1':a_s1, 's2':a_s2, 'cs1':a_cs1, 'cs2':a_cs2, 'pax_acceptances':a_pax_acceptances, 's1_cut_acceptances':a_s1_cut_acceptances, 's2_cut_acceptances':a_s2_cut_acceptances, 'threshold_acceptances':a_threshold_acceptances})
    
    l_dataframes[i] = l_dataframes[i][l_dataframes[i]['cs1'] > 0]

    l_dataframes[i]['threshold_acceptances'] *= l_dataframes[i]['cs1'].apply(s1_threshold)

    #print l_dataframes[i]


    

current_analysis.end_gpu_context()



l_energy_settings = [70, 0, 70]

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

        current_df = l_dataframes[j][(l_dataframes[j]['energy'] > bin_edges_energy[i]) & (l_dataframes[j]['energy'] < bin_edges_energy[i+1])]
        
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
    print d_arrays[label]['a_prob']
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

plt.show()


pickle.dump(d_arrays, open('./mc_output/signal_efficiency_bands.p', 'w'))


