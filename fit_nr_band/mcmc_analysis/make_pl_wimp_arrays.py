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


# -----------------------------------------
#  Setup and initialization
# -----------------------------------------

l_grid_parameters = ['gamma', 'alpha', 'lambda', 'extraction_efficiency']
num_mc_events = int(2e5)

fiducial_volume_mass = 1000. # kg
print '\n\nGiving rate assuming %0.f kg FV\n\n\n' % (fiducial_volume_mass)

lax_version = 'lax_0.11.1'
print '\n\nCurrently using %s for fitting\n\n\n' % (lax_version)


l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1', 'extraction_efficiency', 'gas_gain_mean', 'gas_gain_width', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par'] + ['prob_bkg', 'scale_par']
num_dim = len(l_par_names)

l_sigma_levels = [-1, -0.5, 0, 0.5, 1]
assert 0 in l_sigma_levels

d_sigma_to_percentile = {-10:0.00001, -2:2.5, -1:16, -0.5:31, 0:50, 0.5:69, 1:84, 2:97.5, 10:99.999999}

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
num_steps_to_include = 10


if(len(sys.argv) != 3):
	print 'Usage is python make_pl_histograms_wimps.py <num walkers> <wimp mass>'
	sys.exit(1)

num_walkers = int(sys.argv[1])
wimp_mass = int(sys.argv[2]) # GeV
s_wimp_mass = '%04d' % (wimp_mass)


dir_specifier_name = 'run_0_band'

cathode_setting = config_xe1t.l_allowed_cathode_settings[0]
degree_setting = config_xe1t.l_allowed_degree_settings[0]


name_of_results_directory = config_xe1t.results_directory_name
l_plots = ['plots', dir_specifier_name, '%.3f_kV_%d_deg' % (cathode_setting, degree_setting), 'wimp_spectra']


current_analysis = nr_analysis_xe1t.nr_analysis_xe1t('wimp', 'lax_0.11.1', num_mc_events, num_walkers, num_steps_to_include, wimp_mass=wimp_mass)



# -----------------------------------------
#  Find maxima for each parameter
#  individually
# -----------------------------------------

d_grid_parameter_percentiles = {}
a_samples = current_analysis.a_samples

for par_name in l_grid_parameters:
    d_grid_parameter_percentiles[par_name] = {}
    
    for sigma_level in l_sigma_levels:
        d_grid_parameter_percentiles[par_name][sigma_level] = np.percentile(a_samples[:, d_parameter_to_index[par_name]], [d_sigma_to_percentile[sigma_level]])




l_s1_settings = config_xe1t.l_s1_settings
l_log_settings = config_xe1t.l_log_settings
l_s2_settings = config_xe1t.l_s2_settings

figure_sizes = (11, 13)




a_selected_sampler = []

for par_name in l_grid_parameters:
    a_selected_sampler.append(a_samples[:, d_parameter_to_index[par_name]])

a_selected_sampler = np.asarray(a_selected_sampler)

a_means = np.mean(a_selected_sampler, axis=1)
a_cov_matrix = np.cov(a_selected_sampler)

bin_edges_s1_th2 = np.linspace(config_xe1t.l_s1_settings_pl[1], config_xe1t.l_s1_settings_pl[2], config_xe1t.l_s1_settings_pl[0]+1)
bin_edges_s1_th2 = np.asarray(bin_edges_s1_th2, dtype=np.float32)
bin_edges_s2_th2 = np.linspace(np.log10(config_xe1t.l_s2_settings_pl[1]), np.log10(config_xe1t.l_s2_settings_pl[2]), config_xe1t.l_s2_settings_pl[0]+1)
bin_edges_s2_th2 = np.asarray(bin_edges_s2_th2, dtype=np.float32)

#print bin_edges_s1_th2
#print bin_edges_s2_th2



# -----------------------------------------------
# -----------------------------------------------
# run MC
# -----------------------------------------------
# -----------------------------------------------

d_plotting_information = current_analysis.prepare_gpu()
d_parameters_for_gpu = current_analysis.get_best_fit_parameters()

d_histograms = {}
d_arrays_to_pickle = {}

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

f_constant = root.TF2('constant', '0.0000001', bin_edges_s1_th2[0], bin_edges_s1_th2[-1], bin_edges_s2_th2[0], bin_edges_s2_th2[-1])
    
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


                num_trials = np.asarray(num_mc_events, dtype=np.int32)
                mean_field = np.asarray(config_xe1t.d_cathode_voltages_to_field[cathode_setting], dtype=np.float32)


                a_s1 = np.zeros(num_mc_events, dtype=np.float32)
                a_s2 = np.zeros(num_mc_events, dtype=np.float32)
                a_weights = np.zeros(num_mc_events, dtype=np.float32)


                t_args = (current_analysis.get_rng_states(), drv.In(num_trials), drv.In(mean_field), d_plotting_information['gpu_energies'], d_plotting_information['gpu_x_positions'], d_plotting_information['gpu_y_positions'], d_plotting_information['gpu_z_positions'], d_plotting_information['gpu_e_survival_prob'], drv.In(d_parameters_for_gpu['prob_bkg']), d_plotting_information['gpu_er_band_s1'], d_plotting_information['gpu_er_band_log'], drv.In(d_parameters_for_gpu['w_value']), drv.In(d_parameters_for_gpu['alpha']), drv.In(d_parameters_for_gpu['zeta']), drv.In(d_parameters_for_gpu['beta']), drv.In(d_parameters_for_gpu['gamma']), drv.In(d_parameters_for_gpu['delta']), drv.In(d_parameters_for_gpu['kappa']), drv.In(d_parameters_for_gpu['eta']), drv.In(d_parameters_for_gpu['lambda']), drv.In(d_parameters_for_gpu['g1_value']), drv.In(d_parameters_for_gpu['extraction_efficiency']), drv.In(d_parameters_for_gpu['gas_gain_value']), drv.In(d_parameters_for_gpu['gas_gain_width']), drv.In(d_parameters_for_gpu['dpe_prob']), drv.In(d_parameters_for_gpu['s1_bias_par']), drv.In(d_parameters_for_gpu['s1_smearing_par']), drv.In(d_parameters_for_gpu['s2_bias_par']), drv.In(d_parameters_for_gpu['s2_smearing_par']), drv.In(d_parameters_for_gpu['acceptance_par']), drv.In(d_parameters_for_gpu['num_pts_s1bs']), d_plotting_information['gpu_s1bs_s1s'], d_plotting_information['gpu_s1bs_lb_bias'], d_plotting_information['gpu_s1bs_ub_bias'], d_plotting_information['gpu_s1bs_lb_smearing'], d_plotting_information['gpu_s1bs_ub_smearing'], drv.In(d_parameters_for_gpu['num_pts_s2bs']), d_plotting_information['gpu_s2bs_s2s'], d_plotting_information['gpu_s2bs_lb_bias'], d_plotting_information['gpu_s2bs_ub_bias'], d_plotting_information['gpu_s2bs_lb_smearing'], d_plotting_information['gpu_s2bs_ub_smearing'], drv.In(d_parameters_for_gpu['num_pts_s1pf']), d_plotting_information['gpu_s1pf_s1s'], d_plotting_information['gpu_s1pf_lb_acc'], d_plotting_information['gpu_s1pf_mean_acc'], d_plotting_information['gpu_s1pf_ub_acc'], drv.In(d_parameters_for_gpu['current_cut_acceptance_s1_intercept']), drv.In(d_parameters_for_gpu['current_cut_acceptance_s1_slope']), drv.In(d_parameters_for_gpu['current_cut_acceptance_s2_intercept']), drv.In(d_parameters_for_gpu['current_cut_acceptance_s2_slope']), drv.In(d_parameters_for_gpu['num_bins_r2']), d_plotting_information['gpu_bin_edges_r2'], drv.In(d_parameters_for_gpu['num_bins_z']), d_plotting_information['gpu_bin_edges_z'], d_plotting_information['gpu_s1_correction_map'], drv.In(d_parameters_for_gpu['num_bins_x']), d_plotting_information['gpu_bin_edges_x'], drv.In(d_parameters_for_gpu['num_bins_y']), d_plotting_information['gpu_bin_edges_y'], d_plotting_information['gpu_s2_correction_map'], drv.InOut(a_s1), drv.InOut(a_s2), drv.InOut(a_weights))

                current_analysis.call_gpu_func(t_args)
                
                h_current = root.TH2F(s_key, s_key, config_xe1t.l_s1_settings_pl[0], bin_edges_s1_th2, config_xe1t.l_s2_settings_pl[0], bin_edges_s2_th2)
                h_current.Add(f_constant)
                
                count_events = 0
                
                for i in xrange(len(a_s1)):
                    if (not np.isnan(a_s1[i])) and (not np.isnan(a_s2[i])) and (a_s2[i] > 0):
                        h_current.Fill(a_s1[i], np.log10(a_s2[i]), a_weights[i])
                        count_events += a_weights[i]
                
                #print '\nEvents per day: %f\n' % (count_events*current_analysis.get_scale_factor())
                        
                h_current.Scale(current_analysis.get_scale_factor())
                
                d_histograms[s_key] = h_current
                d_histograms[s_key].Write()

                d_arrays_to_pickle[s_key] = {'s1':a_s1, 's2':a_s2, 'weights':a_weights}




f_hist.Close()


#pickle.dump(d_arrays_to_pickle, open('./mc_output/%s_arrays.p' % (current_analysis.get_save_name_beginning()), 'w'))





# end GPU context
current_analysis.end_gpu_context()

