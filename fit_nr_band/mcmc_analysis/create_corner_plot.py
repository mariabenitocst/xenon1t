#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

import numpy as np
import corner, time, tqdm
import cPickle as pickle

import neriX_analysis

import config_xe1t

if len(sys.argv) != 2:
    print 'Use is python perform_full_matching.py <num walkers>'
    sys.exit()



d_degree_setting_to_energy_name = {2300:3,
                                   3000:5,
                                   3500:7,
                                   4500:10,
                                   5300:15,
                                   6200:20}


num_walkers = int(sys.argv[1])

directory_descriptor = 'run_0_band'

l_degree_settings_in_use = [-4]
s_degree_settings = ''
for degree_setting in l_degree_settings_in_use:
    s_degree_settings += '%s,' % (degree_setting)
s_degree_settings = s_degree_settings[:-1]


s_identifier = 'sb_ms'

l_cathode_settings_in_use = [12.]
s_cathode_settings = ''
for cathode_setting in l_cathode_settings_in_use:
    s_cathode_settings += '%.3f,' % (cathode_setting)
s_cathode_settings = s_cathode_settings[:-1]

nameOfResultsDirectory = config_xe1t.results_directory_name
l_plots = ['plots', directory_descriptor, '%s_kV_%s_deg' % (s_cathode_settings, s_degree_settings)]

dir_specifier_name = '%s_kV_%s_deg' % (s_cathode_settings, s_degree_settings)

nameOfResultsDirectory += '/%s' % (directory_descriptor)

sPathToFile = './%s/%s/sampler_dictionary.p' % (nameOfResultsDirectory, dir_specifier_name)

if os.path.exists(sPathToFile):
    dSampler = pickle.load(open(sPathToFile, 'r'))
    l_chains = []
    for sampler in dSampler[num_walkers]:
        l_chains.append(sampler['_chain'])

    a_full_sampler = np.concatenate(l_chains, axis=1)

    print 'Successfully loaded sampler!'
else:
    print sPathToFile
    print 'Could not find file!'
    sys.exit()


if s_identifier == 'sb':
    num_dim = 22
    l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'cut_acceptance_par'] + ['prob_bkg', 'scale_par']
elif s_identifier == 'sbf':
    num_dim = 23
    l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par_0', 'acceptance_par_1', 'cut_acceptance_par'] + ['prob_bkg', 'scale_par']
elif s_identifier == 'sb_ms':
    num_dim = 24
    l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'cut_acceptance_par', 'ms_par_0', 'ms_par_1'] + ['prob_bkg', 'scale_par']
    
    # contains all NEST parameteres
    d_comparisons = config_xe1t.nest_lindhard_model
    
    d_comparisons['values']['w_value'] = config_xe1t.w_value
    d_comparisons['uncertainty']['w_value'] = [-config_xe1t.w_value_uncertainty, config_xe1t.w_value_uncertainty]
    
    d_comparisons['values']['g1_value'] = config_xe1t.g1_value
    d_comparisons['uncertainty']['g1_value'] = [-config_xe1t.g1_uncertainty, config_xe1t.g1_uncertainty]

    d_comparisons['values']['extraction_efficiency_value'] = config_xe1t.extraction_efficiency_value
    d_comparisons['uncertainty']['extraction_efficiency_value'] = [-config_xe1t.extraction_efficiency_uncertainty, config_xe1t.extraction_efficiency_uncertainty]
    
    d_comparisons['values']['gas_gain_mean_value'] = config_xe1t.gas_gain_value
    d_comparisons['uncertainty']['gas_gain_mean_value'] = [-config_xe1t.gas_gain_uncertainty, config_xe1t.gas_gain_uncertainty]

    d_comparisons['values']['gas_gain_width_value'] = config_xe1t.gas_gain_width
    d_comparisons['uncertainty']['gas_gain_width_value'] = [-config_xe1t.gas_gain_width_uncertainty, config_xe1t.gas_gain_width_uncertainty]

    d_comparisons['values']['dpe_prob'] = config_xe1t.dpe_median
    d_comparisons['uncertainty']['dpe_prob'] = [-config_xe1t.dpe_std, config_xe1t.dpe_std]
    
    d_comparisons['values']['s1_bias_par'] = config_xe1t.bias_median
    d_comparisons['uncertainty']['s1_bias_par'] = [-config_xe1t.bias_std, config_xe1t.bias_std]

    d_comparisons['values']['s1_smearing_par'] = config_xe1t.bias_median
    d_comparisons['uncertainty']['s1_smearing_par'] = [-config_xe1t.bias_std, config_xe1t.bias_std]
    
    d_comparisons['values']['s2_bias_par'] = config_xe1t.bias_median
    d_comparisons['uncertainty']['s2_bias_par'] = [-config_xe1t.bias_std, config_xe1t.bias_std]
    
    d_comparisons['values']['s2_smearing_par'] = config_xe1t.bias_median
    d_comparisons['uncertainty']['s2_smearing_par'] = [-config_xe1t.bias_std, config_xe1t.bias_std]
    
    d_comparisons['values']['acceptance_par'] = 0
    d_comparisons['uncertainty']['acceptance_par'] = [-1, 1]

    d_comparisons['values']['cut_acceptance_par'] = 0
    d_comparisons['uncertainty']['cut_acceptance_par'] = [-1, 1]
    
    d_comparisons['values']['ms_par_0'] = config_xe1t.ms_par_0
    d_comparisons['uncertainty']['ms_par_0'] = [-config_xe1t.ms_par_0_unc, config_xe1t.ms_par_0_unc]
    
    d_comparisons['values']['ms_par_1'] = config_xe1t.ms_par_1
    d_comparisons['uncertainty']['ms_par_1'] = [-config_xe1t.ms_par_1_unc, config_xe1t.ms_par_1_unc]



assert num_dim == len(l_par_names)

num_steps = 1000

samples = a_full_sampler[:, -num_steps:, :].reshape((-1, num_dim))

start_time = time.time()
print 'Starting corner plot...\n'
fig = corner.corner(samples, labels=l_par_names, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3e', title_kwargs={"fontsize": 12})
print 'Corner plot took %.3f minutes.\n\n' % ((time.time()-start_time)/60.)


# as a test reduce chain size
#a_full_sampler = a_full_sampler[:, :int(a_full_sampler.shape[1]/2), :]

tot_number_events = a_full_sampler.shape[1]
batch_size = int(tot_number_events/40)
num_batches = int(tot_number_events/batch_size/2)
d_gr_stats = {}

if s_identifier == 'sb':
    l_free_pars = ['gamma', 'kappa', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'cut_acceptance_par', 'prob_bkg', 'scale_par']
elif s_identifier == 'sbf':
    l_free_pars = ['gamma', 'kappa', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par_0', 'acceptance_par_1', 'cut_acceptance_par', 'prob_bkg', 'scale_par']
elif s_identifier == 'sb_ms':
    l_free_pars = ['gamma', 'kappa', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'cut_acceptance_par', 'ms_par_0', 'ms_par_1', 'prob_bkg', 'scale_par']

l_colors = plt.get_cmap('jet')(np.linspace(0, 1.0, len(l_free_pars)))

for par_name in l_par_names:
    d_gr_stats[par_name] = [0 for i in xrange(num_batches)]

l_size_for_test = [2*i*batch_size for i in xrange(num_batches)]

# calculate Gelman-Rubin statistic
print '\nCalculating Gelman-Rubin Statistic for each parameter...\n'
for i in tqdm.tqdm(xrange(num_dim)):
    par_name = l_par_names[i]
    for j in xrange(1, num_batches+1):
        #print tot_number_events, 2*j*batch_size
        num_events_in_batch = float(j*batch_size)
    
        a_sampler = a_full_sampler[:, j*batch_size:2*j*batch_size, i]
        #print a_sampler[0,:]
        #print np.var(a_sampler[0,:], ddof=1)
        
        a_means = np.mean(a_sampler, axis=1)
        a_vars = np.var(a_sampler, axis=1, ddof=1)
        
        #print a_means
        #print a_vars
        
        mean_of_means = np.mean(a_means)
        #print len(a_vars), a_vars
        #print num_walkers, np.sum(a_vars), mean_of_means
        b_ = num_events_in_batch/(num_walkers-1.) * np.sum((a_means-mean_of_means)**2)
        w_ = 1./(num_walkers) * np.sum(a_vars)
        var_p = (num_events_in_batch-1)/num_events_in_batch*w_ + b_/num_events_in_batch
        v_ = var_p + b_/(num_events_in_batch*num_walkers)
        rg_stat = (v_/w_)**0.5
        
        #print num_events_in_batch, b_, w_, var_p, v_
        
        d_gr_stats[par_name][j-1] = rg_stat
        #print 'G-R statistic for %s in batch %d: %.3f\n' % (l_par_names[i], j, rg_stat)


f_gr, a_gr = plt.subplots(1)
l_legend_handles = []
for i, par_name in enumerate(l_free_pars):
    current_handle, = a_gr.plot(l_size_for_test, d_gr_stats[par_name], color=l_colors[i], linestyle='-', label=par_name)
    l_legend_handles.append(current_handle)

a_gr.plot(l_size_for_test, [1.1 for i in xrange(len(l_size_for_test))], linestyle='--', color='black')
a_gr.set_xlabel('MCMC Steps')
a_gr.set_ylabel('GR Statistic')
a_gr.set_ylim([1, 5])
a_gr.set_yscale('log')
a_gr.legend(handles=l_legend_handles, loc='upper center', fontsize=5)





l_quantiles = [16., 50., 84.]

l_posterior_median = []
l_posterior_lb = []
l_posterior_ub = []

l_prior_median = []
l_prior_lb = []
l_prior_ub = []

l_names = []

for i, par_name in enumerate(l_par_names):
    lb_par, med_par, ub_par = np.percentile(samples[:,i], l_quantiles)
    print '^  %s  |  $%.2e^{+%.2e}_{-%.2e}$  |' % (par_name, med_par, ub_par-med_par, med_par-lb_par)
    
    if par_name in d_comparisons['values']:
        l_posterior_median.append(med_par - d_comparisons['values'][par_name])
        l_posterior_lb.append(med_par-lb_par)
        l_posterior_ub.append(ub_par-med_par)
        
        l_prior_median.append(d_comparisons['values'][par_name]
         - d_comparisons['values'][par_name])
        l_prior_lb.append(-d_comparisons['uncertainty'][par_name][0])
        l_prior_ub.append(d_comparisons['uncertainty'][par_name][1])
        
        l_names.append(par_name)

"""
l_x_values = [i for i in xrange(len(l_names))]

f_comparisons, ax_comparisons = plt.subplots(1)

ax_comparisons.set_xticks(l_x_values, l_names)

ax_comparisons.errorbar(l_x_values, l_posterior_median, yerr=[l_posterior_lb, l_posterior_ub], color='b', label='Posterior')
ax_comparisons.errorbar(l_x_values, l_prior_median, yerr=[l_prior_lb, l_prior_ub], color='r', label='Prior')

ax_comparisons.set_xlabel('Parameter')
ax_comparisons.set_ylabel('Value minus prior median')
ax_comparisons.legend(loc='best')

plt.show()
"""

a_variation_samples = samples[:, [1,4,7,18]]
l_variation_labels = [l_par_names[1], l_par_names[4], l_par_names[7], l_par_names[18]]
fig_variation = corner.corner(a_variation_samples, labels=l_variation_labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3e', title_kwargs={"fontsize": 12})


# path for save
s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)

fig.savefig('%s%s_corner_plot_%s.png' % (s_path_for_save, s_identifier, dir_specifier_name))
f_gr.savefig('%s%s_gr_statistic_%s.png' % (s_path_for_save, s_identifier, dir_specifier_name))
fig_variation.savefig('%s%s_variation_corner_plot_%s.png' % (s_path_for_save, s_identifier, dir_specifier_name))









"""
if s_identifier == 'sbf':

    # define the necessary function
    def fit_exp_eff(x, shape, offset):
        return max(0, 1. - np.exp(-shape*(x - offset)))

    #print np.mean(samples[-num_walkers*num_steps:, -5:-3], axis=0)
    #print np.cov(samples[-num_walkers*num_steps:, -5:-3].T)

    a_x_values, a_y_values, a_y_err_low, a_y_err_high = neriX_analysis.create_1d_fit_confidence_band(fit_exp_eff, np.mean(samples[-num_steps:, -5:-3]), np.cov(samples[-num_steps:, -5:-3]), 1, 20)
    
    f_eff, ax_eff = plt.subplots(1)
    ax_eff.fill_between(a_x_values, a_y_err_low, a_y_err_high)
    plt.show()
"""

#raw_input('Enter to continue...')
