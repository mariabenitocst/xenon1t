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



num_dim = 20

l_par_names = ['w_value', 'alpha', 'zeta', 'beta', 'gamma', 'delta', 'kappa', 'eta', 'lambda', 'g1_value', 'extraction_efficiency_value', 'gas_gain_mean_value', 'gas_gain_width_value', 'dpe_prob', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par'] + ['scale_par']


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

l_free_pars = ['gamma', 'kappa', 's1_bias_par', 's1_smearing_par', 's2_bias_par', 's2_smearing_par', 'acceptance_par', 'scale_par']

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
a_gr.set_ylim([1, 5])
a_gr.set_yscale('log')
a_gr.legend(handles=l_legend_handles, loc='upper center', fontsize=5)


# path for save
s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)

fig.savefig('%ss_corner_plot_%s.png' % (s_path_for_save, dir_specifier_name))
f_gr.savefig('%ss_gr_statistic_%s.png' % (s_path_for_save, dir_specifier_name))



#raw_input('Enter to continue...')
