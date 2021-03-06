#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

import ROOT as root
from rootpy.plotting import Hist, Hist2D, Canvas, Legend
import numpy as np
import corner, time
import cPickle as pickle
import tqdm

if len(sys.argv) != 3:
    print 'Use is python create_corner_plot.py <filename> <num walkers>'
    sys.exit()

print '\n\nBy default look for all energies - change source if anything else is needed.\n'



filename = sys.argv[1]
num_walkers = int(sys.argv[2])

l_plots = ['plots', filename]

dir_specifier_name = filename

sPathToFile = './results/%s/sampler_dictionary.p' % (dir_specifier_name)

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



l_free_pars = ['p_hit_first_dynode', 'mean_electrons_per_dynode', 'width_electrons_per_dynode', 'p_e_freed', 'bkg_mean', 'bkg_std', 'mean_num_pe_mpe', 'scale_par']
l_colors = ['b', 'r', 'g', 'y', 'black', 'm', 'darkcyan', 'brown']
num_dim = len(l_free_pars)


num_steps = 1000

samples = a_full_sampler[:, -num_steps:, :].reshape((-1, num_dim))

print samples.shape
start_time = time.time()
print 'Starting corner plot...\n'
fig = corner.corner(samples, labels=l_free_pars, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3e', title_kwargs={"fontsize": 12})
print 'Corner plot took %.3f minutes.\n\n' % ((time.time()-start_time)/60.)




tot_number_events = a_full_sampler.shape[1]
batch_size = int(tot_number_events/40) # 40 usually
num_batches = int(tot_number_events/batch_size/2)
d_gr_stats = {}



for par_name in l_free_pars:
    d_gr_stats[par_name] = [0 for i in xrange(num_batches)]

l_size_for_test = [2*i*batch_size for i in xrange(num_batches)]

# calculate Gelman-Rubin statistic
print '\nCalculating Gelman-Rubin Statistic for each parameter...\n'
for i in tqdm.tqdm(xrange(num_dim)):
    par_name = l_free_pars[i]
    for j in xrange(1, num_batches+1):
        #print tot_number_events, 2*j*batch_size
        num_events_in_batch = float(j*batch_size)
    
        a_sampler = a_full_sampler[:, j*batch_size:2*j*batch_size, i]
        #print a_sampler[0,:]
        #print np.var(a_sampler[0,:], ddof=1)
        
        a_means = np.mean(a_sampler, axis=1)
        a_vars = np.var(a_sampler, axis=1, ddof=1)
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
        #print 'G-R statistic for %s in batch %d: %.3f\n' % (l_labels[i], j, rg_stat)


f_gr, a_gr = plt.subplots(1)
l_legend_handles = []
for i, par_name in enumerate(l_free_pars):
    current_handle, = a_gr.plot(l_size_for_test, d_gr_stats[par_name], color=l_colors[i], linestyle='-', label=par_name)
    l_legend_handles.append(current_handle)

a_gr.plot(l_size_for_test, [1.1 for i in xrange(len(l_size_for_test))], linestyle='--', color='black')
a_gr.set_ylim([1, 20])
a_gr.set_yscale('log')
a_gr.legend(handles=l_legend_handles, loc='best')





# path for save
s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)

fig.savefig('%s%s_corner_plot.png' % (s_path_for_save, dir_specifier_name))
f_gr.savefig('%s%s_gr_statistic.png' % (s_path_for_save, dir_specifier_name))



#raw_input('Enter to continue...')
