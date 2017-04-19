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

num_mc_events = 2e4
num_walkers = 256
num_steps_to_include = 10

# name notes for the saved pickle file
name_notes = ''

cathode_setting = config_xe1t.l_allowed_cathode_settings[0]
degree_setting = config_xe1t.l_allowed_degree_settings[0]

current_analysis = nr_analysis_xe1t.nr_analysis_xe1t('cnns', 'lax_0.11.1', num_mc_events, num_walkers, num_steps_to_include)

d_plotting_information = current_analysis.prepare_gpu()



# -----------------------------------------------
# -----------------------------------------------
# run MC
# -----------------------------------------------
# -----------------------------------------------


num_trials = np.asarray(num_mc_events, dtype=np.int32)
mean_field = np.asarray(config_xe1t.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)


d_sampler_values = current_analysis.get_best_fit_parameters()


scale_factor = current_analysis.get_scale_factor()

a_s1_mc = np.full(num_mc_events, -1, dtype=np.float32)
a_s2_mc = np.full(num_mc_events, -2, dtype=np.float32)
a_weight = np.full(num_mc_events, -3, dtype=np.float32)

t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_sampler_values['prob_bkg']), drv.In(d_sampler_values['prob_ac_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], d_plotting_information['gpu_ac_bkg_s1'], d_plotting_information['gpu_ac_bkg_log'], drv.In(d_sampler_values['w_value']), drv.In(d_sampler_values['alpha']), drv.In(d_sampler_values['zeta']), drv.In(d_sampler_values['beta']), drv.In(d_sampler_values['gamma']), drv.In(d_sampler_values['delta']), drv.In(d_sampler_values['kappa']), drv.In(d_sampler_values['eta']), drv.In(d_sampler_values['lamb']), drv.In(d_sampler_values['g1_value']), drv.In(d_sampler_values['extraction_efficiency']), drv.In(d_sampler_values['gas_gain_value']), drv.In(d_sampler_values['gas_gain_width']), drv.In(d_sampler_values['dpe_prob']), drv.In(d_sampler_values['s1_bias_par']), drv.In(d_sampler_values['s1_smearing_par']), drv.In(d_sampler_values['s2_bias_par']), drv.In(d_sampler_values['s2_smearing_par']), drv.In(d_sampler_values['acceptance_par']), drv.In(d_sampler_values['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_sampler_values['num_pts_s1pf']), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(d_sampler_values['current_cut_acceptance_s1_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s1_slope']), drv.In(d_sampler_values['current_cut_acceptance_s2_intercept']), drv.In(d_sampler_values['current_cut_acceptance_s2_slope']), drv.In(d_sampler_values['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_sampler_values['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_sampler_values['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_sampler_values['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1_mc), drv.InOut(a_s2_mc), drv.InOut(a_weight))
    

current_analysis.call_gpu_func(t_args)
    

current_analysis.end_gpu_context()

df_mc = pd.DataFrame({'s1':a_s1_mc, 's2':a_s2_mc, 'weight':a_weight})
df_mc = df_mc[(df_mc['s1'] > 0) & (df_mc['s2'] > 0) & (df_mc['weight'] > 0)]


d_mc = {'d_mc':df_mc.to_dict(), 'scale_factor':current_analysis.get_scale_factor()}

pickle.dump(d_mc, open('%s%s%s.p' % (config_xe1t.path_to_mc_outputs, current_analysis.get_save_name_beginning(), name_notes), 'w'))


