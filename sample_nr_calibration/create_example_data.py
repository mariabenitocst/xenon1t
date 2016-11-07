#!/usr/bin/python
#import pickle
#print pickle.Pickler.dispatch
import dill
#print pickle.Pickler.dispatch

import ROOT as root
import sys, os

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

import emcee, corner, click
import neriX_analysis, neriX_datasets, neriX_config
from rootpy.plotting import Hist2D, Hist, Legend, Canvas
from rootpy.tree import Tree
from rootpy.io import File
from rootpy import stl
import numpy as np
import tqdm, time, copy_reg, types, pickle
from root_numpy import tree2array, array2tree
import scipy.optimize as op
import scipy.special
from scipy.stats import norm, poisson
from scipy.special import erf
from math import floor

import example_config

import astroML.density_estimation

import cuda_example_observables_production
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray
import pycuda.autoinit

gpu_example_observables_production_arrays = SourceModule(cuda_example_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_example_observables_production_lindhard_arrays')
setup_kernel = SourceModule(cuda_example_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('setup_kernel')


# grab parameters from config file
d_data_parameters = example_config.d_data_parameters
"""
'name':'nerix-like_nr',
'num_events':6000*10,
'l_energies':[3, 6, 9, 12, 15, 18],
'energy_width':0.05,
'mean_field':500, # V/cm
'w_value':13.7,
'alpha':1.24,
'zeta':0.0472,
'beta':239,
'gamma':0.01385,
'delta':0.0620,
'kappa':0.1394,
'eta':3.3,
'lambda':1.14,
'g1':[0.129, 0.003],
'extraction_efficiency':[0.984, 0.002],
'gas_gain_value':[21.29, 0.53],
'gas_gain_width':[8.01, 0.29],
'spe_res':[0.6, 0.01],
's1_eff_par0':[1.96, 0.008],
's1_eff_par1':[0.46, 0.0053],
's2_eff_pars':[91.2, 432.1],
's2_eff_cov_matrix':[[455.2, 134.8], [134.8, 812.8]]
"""

l_save_directory = ['example_data', d_data_parameters['name']]


def energy_pdf(x_values):
    pdf_output = np.zeros(len(x_values))

    for i in xrange(len(d_data_parameters['l_energies'])):
        pdf_output += norm.pdf(x_values, loc=d_data_parameters['l_energies'][i], scale=d_data_parameters['l_energies'][i]*d_data_parameters['energy_width'])

    return pdf_output



num_mc_events = int(d_data_parameters['num_events'])
d_gpu_scale = {}
block_dim = 1024
d_gpu_scale['block'] = (block_dim,1,1)
numBlocks = floor(num_mc_events / float(block_dim))
d_gpu_scale['grid'] = (int(numBlocks), 1)
num_mc_events = int(numBlocks*block_dim)

seed = int(time.time())
rng_states = drv.mem_alloc(num_mc_events*pycuda.characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
setup_kernel(np.int32(num_mc_events), rng_states, np.uint64(seed), np.uint64(0), **d_gpu_scale)
print 'Cuda random states setup...\n'


l_energy_arrays = [0 for i in xrange(len(d_data_parameters['l_energies']))]
for i in xrange(len(l_energy_arrays)):
    l_energy_arrays[i] = np.random.normal(loc=d_data_parameters['l_energies'][i], scale=d_data_parameters['l_energies'][i]*d_data_parameters['energy_width'], size=int(d_data_parameters['num_events']/len(l_energy_arrays)))

# concatenate arrays and make sure they are
# float32 for use on GPU
a_energies = np.concatenate(l_energy_arrays)
a_energies = np.asarray(a_energies, dtype=np.float32)


nb_energy = 100
lb_energy = 1
ub_energy = 25

nb_s1 = 40
lb_s1 = 0
ub_s1 = 35

nb_s2 = 40
lb_s2 = 0
ub_s2 = 2500

nb_logs2s1 = 40
lb_logs2s1 = 1.5
ub_logs2s1 = 3


x_pts_energy = np.linspace(lb_energy, ub_energy, nb_energy)
y_pts_energy = energy_pdf(x_pts_energy)

f_energy, (ax_energy_pdf, ax_energy_sample, ax_s1_s2) = plt.subplots(1, 3, figsize=[18, 6])

h_energy, be_energy = np.histogram(a_energies, bins=nb_energy, range=[lb_energy, ub_energy])

ax_energy_pdf.plot(x_pts_energy, y_pts_energy)

ax_energy_pdf.set_title('Assumed PDF for NR Source')
ax_energy_pdf.set_xlabel('Energy [keV]')
ax_energy_pdf.set_ylabel(r'PDF [$\frac{1}{keV}$]')


# use analysis function to create histogram with
# Poisson errors
a_x_values_energy, a_y_values_energy, a_x_err_low_energy, a_x_err_high_energy, a_y_err_low_energy, a_y_err_high_energy = neriX_analysis.prepare_hist_arrays_for_plotting(h_energy, be_energy)
ax_energy_sample.errorbar(a_x_values_energy, a_y_values_energy, xerr=[a_x_err_low_energy, a_x_err_high_energy], yerr=[a_y_err_low_energy, a_y_err_high_energy], color='b')

ax_energy_sample.set_title('Sample of PDF for NR Source')
ax_energy_sample.set_xlabel('Energy [keV]')
ax_energy_sample.set_ylabel(r'Counts')


# random state done
a_num_trials = np.asarray(num_mc_events, dtype=np.int32)
a_mean_field = np.asarray(d_data_parameters['mean_field'], dtype=np.float32)
# energy done
a_w_value = np.asarray(d_data_parameters['w_value'], dtype=np.float32)
a_alpha = np.asarray(d_data_parameters['alpha'], dtype=np.float32)
a_zeta = np.asarray(d_data_parameters['zeta'], dtype=np.float32)
a_beta = np.asarray(d_data_parameters['beta'], dtype=np.float32)
a_gamma = np.asarray(d_data_parameters['gamma'], dtype=np.float32)
a_delta = np.asarray(d_data_parameters['delta'], dtype=np.float32)
a_kappa = np.asarray(d_data_parameters['kappa'], dtype=np.float32)
a_eta = np.asarray(d_data_parameters['eta'], dtype=np.float32)
a_lambda = np.asarray(d_data_parameters['lambda'], dtype=np.float32)
a_g1 = np.asarray(d_data_parameters['g1'], dtype=np.float32)
a_extraction_efficiency = np.asarray(d_data_parameters['extraction_efficiency'], dtype=np.float32)
a_gas_gain_value = np.asarray(d_data_parameters['gas_gain_value'], dtype=np.float32)
a_gas_gain_width = np.asarray(d_data_parameters['gas_gain_width'], dtype=np.float32)
a_spe_res = np.asarray(d_data_parameters['spe_res'], dtype=np.float32)
a_s1_eff_par0 = np.asarray(d_data_parameters['s1_eff_par0'][0], dtype=np.float32)
a_s1_eff_par1 = np.asarray(d_data_parameters['s1_eff_par1'][0], dtype=np.float32)
a_s2_eff_par0 = np.asarray(d_data_parameters['s2_eff_pars'][0], dtype=np.float32)
a_s2_eff_par1 = np.asarray(d_data_parameters['s2_eff_pars'][1], dtype=np.float32)

a_s1 = np.full(num_mc_events, -1, dtype=np.float32)
a_s2 = np.full(num_mc_events, -1, dtype=np.float32)

l_gpu_args = [rng_states, drv.In(a_num_trials), drv.In(a_mean_field), drv.In(a_energies), drv.In(a_w_value), drv.In(a_alpha), drv.In(a_zeta), drv.In(a_beta), drv.In(a_gamma), drv.In(a_delta), drv.In(a_kappa), drv.In(a_eta), drv.In(a_lambda), drv.In(a_g1), drv.In(a_extraction_efficiency), drv.In(a_gas_gain_value), drv.In(a_gas_gain_width), drv.In(a_spe_res), drv.In(a_s1_eff_par0), drv.In(a_s1_eff_par1), drv.In(a_s2_eff_par0), drv.In(a_s2_eff_par1), drv.InOut(a_s1), drv.InOut(a_s2)]

gpu_example_observables_production_arrays(*l_gpu_args, **d_gpu_scale)

a_indices_of_lost_events = np.where(a_s1 == -1.)
a_s1 = np.delete(a_s1, a_indices_of_lost_events)
a_s2 = np.delete(a_s2, a_indices_of_lost_events)

print '\n\nPercentage of events after acceptance: %.3f\n\n' % (len(a_s1)/float(num_mc_events))

ax_s1_s2.hist2d(a_s1, a_s2, bins=[nb_s1, nb_s2], range=[[lb_s1, ub_s1], [lb_s2, ub_s2]])
#ax_s1_s2.hist2d(a_s1, np.log10(a_s2/a_s1), bins=[nb_s1, nb_logs2s1], range=[[lb_s1, ub_s1], [lb_logs2s1, ub_logs2s1]])

ax_s1_s2.set_title('Observables for NR Source')
ax_s1_s2.set_xlabel('S1 [PE]')
ax_s1_s2.set_ylabel(r'S2 [PE]')

neriX_analysis.save_figure(l_save_directory, f_energy, 'produced_data_%s' % (d_data_parameters['name']), batch_mode=True)



# make plots of the S1 and S2 acceptances

def s1_acceptance(s1, center, shape):
    return 1. / (1. + np.exp(-(s1-center)/shape))

def s2_acceptance(s1, center, shape):
    return max([0, 1. - np.exp(-(s1-center)/shape)])


f_acceptance, (ax_s1_acceptance, ax_s2_acceptance) = plt.subplots(1, 2, figsize=[12, 6])

a_x_s1, a_s1_acc_values, a_s1_acc_err_low, a_s1_acc_err_high = neriX_analysis.create_1d_fit_confidence_band(s1_acceptance, [d_data_parameters['s1_eff_par0'][0], d_data_parameters['s1_eff_par1'][0]], [[d_data_parameters['s1_eff_par0'][1]**2, 0], [0, d_data_parameters['s1_eff_par1'][1]**2]], 0, 6)

ax_s1_acceptance.plot(a_x_s1, a_s1_acc_values, color='g', linestyle='--')
ax_s1_acceptance.fill_between(a_x_s1, a_s1_acc_values-a_s1_acc_err_low, a_s1_acc_values+a_s1_acc_err_high, color='b', alpha=0.2)

ax_s1_acceptance.set_title('S1 Acceptance')
ax_s1_acceptance.set_xlabel('S1 [PE]')
ax_s1_acceptance.set_ylabel('Fraction of Events Accepted')
ax_s1_acceptance.grid(True)



a_x_s2, a_s2_acc_values, a_s2_acc_err_low, a_s2_acc_err_high = neriX_analysis.create_1d_fit_confidence_band(s2_acceptance, d_data_parameters['s2_eff_pars'], d_data_parameters['s2_eff_cov_matrix'], 0, 2500)

ax_s2_acceptance.plot(a_x_s2, a_s2_acc_values, color='g', linestyle='--')
ax_s2_acceptance.fill_between(a_x_s2, a_s2_acc_values-a_s2_acc_err_low, a_s2_acc_values+a_s2_acc_err_high, color='b', alpha=0.2)

ax_s2_acceptance.set_title('S2 Acceptance')
ax_s2_acceptance.set_xlabel('S2 [PE]')
ax_s2_acceptance.set_ylabel('Fraction of Events Accepted')
ax_s2_acceptance.grid(True)


neriX_analysis.save_figure(l_save_directory, f_acceptance, 's1_and_s2_acceptance_%s' % (d_data_parameters['name']), batch_mode=True)


neriX_analysis.pickle_object(l_save_directory, a_s1, 'a_s1_%s' % (d_data_parameters['name']), batch_mode=True)
neriX_analysis.pickle_object(l_save_directory, a_s2, 'a_s2_%s' % (d_data_parameters['name']), batch_mode=True)





#plt.show()




