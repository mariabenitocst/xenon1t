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




num_walkers = 64
num_steps = 1000

d_files = {}
l_x_labels = ['nerix_160418_1523', 'nerix_160418_1531']

for filename in l_x_labels:
    d_files[filename] = {}
    sPathToFile = './results/%s/sampler_dictionary.p' % (filename)

    if os.path.exists(sPathToFile):
        dSampler = pickle.load(open(sPathToFile, 'r'))
        l_chains = []
        for sampler in dSampler[num_walkers]:
            l_chains.append(sampler['_chain'])

        a_full_sampler = np.concatenate(l_chains, axis=1)
        num_dim = a_full_sampler.shape[2]
        d_files[filename]['a_samples'] = a_full_sampler[:, -num_steps:, :].reshape((-1, num_dim))

        print 'Successfully loaded sampler!'
    else:
        print sPathToFile
        print 'Could not find file!'
        sys.exit()


# dimensions of important variables
dim_prob_pe_from_dynode = 0
dim_prob_bad_collection = 1
dim_ic_first_dynode = 5
dim_ic_bad_collection = 6
dim_mean_num_photons = -3
dim_prob_pe_from_photocathode = -2


num_x_pts = len(l_x_labels)
l_x = [i for i in xrange(num_x_pts)]



l_pfd_med = [0 for i in xrange(num_x_pts)]
l_pfd_lb = [0 for i in xrange(num_x_pts)]
l_pfd_ub = [0 for i in xrange(num_x_pts)]

for i in xrange(num_x_pts):
    #current_lb, current_med, current_ub = np.percentile(d_files[l_x_labels[i]]['a_samples'][:, dim_prob_pe_from_dynode]*d_files[l_x_labels[i]]['a_samples'][:, dim_mean_num_photons], [16., 50., 84.])
    current_lb, current_med, current_ub = np.percentile(d_files[l_x_labels[i]]['a_samples'][:, dim_prob_pe_from_dynode], [16., 50., 84.])
    l_pfd_med[i] = current_med
    l_pfd_lb[i] = current_med - current_lb
    l_pfd_ub[i] = current_ub - current_med

# plot prob that gamma hits first dynode
fig_pfd, ax_pfd = plt.subplots()
ax_pfd.errorbar(l_x, l_pfd_med, yerr=[l_pfd_lb, l_pfd_ub], fmt='b.')
ax_pfd.set_xlabel('Dataset')
ax_pfd.set_ylabel('Probability Incident Photon Hits First Dynode')
ax_pfd.set_xticks(l_x)
ax_pfd.set_xticklabels(l_x_labels)
ax_pfd.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)



l_icfd_med = [0 for i in xrange(num_x_pts)]
l_icfd_lb = [0 for i in xrange(num_x_pts)]
l_icfd_ub = [0 for i in xrange(num_x_pts)]

for i in xrange(num_x_pts):
    current_lb, current_med, current_ub = np.percentile(d_files[l_x_labels[i]]['a_samples'][:, dim_ic_first_dynode], [16., 50., 84.])
    l_icfd_med[i] = current_med
    l_icfd_lb[i] = current_med - current_lb
    l_icfd_ub[i] = current_ub - current_med

# plot prob that gamma hits first dynode
fig_icfd, ax_icfd = plt.subplots()
ax_icfd.errorbar(l_x, l_icfd_med, yerr=[l_icfd_lb, l_icfd_ub], fmt='b.')
ax_icfd.set_xlabel('Dataset')
ax_icfd.set_ylabel('Ionization Correction to PE from First Dynode')
ax_icfd.set_xticks(l_x)
ax_icfd.set_xticklabels(l_x_labels)
ax_icfd.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)




l_icbt_med = [0 for i in xrange(num_x_pts)]
l_icbt_lb = [0 for i in xrange(num_x_pts)]
l_icbt_ub = [0 for i in xrange(num_x_pts)]

for i in xrange(num_x_pts):
    current_lb, current_med, current_ub = np.percentile(d_files[l_x_labels[i]]['a_samples'][:, dim_ic_bad_collection], [16., 50., 84.])
    l_icbt_med[i] = current_med
    l_icbt_lb[i] = current_med - current_lb
    l_icbt_ub[i] = current_ub - current_med

# plot prob that gamma hits first dynode
fig_icbt, ax_icbt = plt.subplots()
ax_icbt.errorbar(l_x, l_icbt_med, yerr=[l_icbt_lb, l_icbt_ub], fmt='b.')
ax_icbt.set_xlabel('Dataset')
ax_icbt.set_ylabel('Ionization Correction to PE with Bad Trajectory')
ax_icbt.set_xticks(l_x)
ax_icbt.set_xticklabels(l_x_labels)
ax_icbt.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)





l_pbc_med = [0 for i in xrange(num_x_pts)]
l_pbc_lb = [0 for i in xrange(num_x_pts)]
l_pbc_ub = [0 for i in xrange(num_x_pts)]

for i in xrange(num_x_pts):
    current_lb, current_med, current_ub = np.percentile(1. - d_files[l_x_labels[i]]['a_samples'][:, dim_prob_bad_collection], [16., 50., 84.])
    l_pbc_med[i] = current_med
    l_pbc_lb[i] = current_med - current_lb
    l_pbc_ub[i] = current_ub - current_med

# plot prob that gamma hits first dynode
fig_pbc, ax_pbc = plt.subplots()
ax_pbc.errorbar(l_x, l_pbc_med, yerr=[l_pbc_lb, l_pbc_ub], fmt='b.')
ax_pbc.set_xlabel('Dataset')
ax_pbc.set_ylabel('Probability PE has Bad Trajectory')
ax_pbc.set_xticks(l_x)
ax_pbc.set_xticklabels(l_x_labels)
ax_pbc.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)





l_mnp_med = [0 for i in xrange(num_x_pts)]
l_mnp_lb = [0 for i in xrange(num_x_pts)]
l_mnp_ub = [0 for i in xrange(num_x_pts)]


for i in xrange(num_x_pts):
    current_lb, current_med, current_ub = np.percentile(d_files[l_x_labels[i]]['a_samples'][:, dim_prob_pe_from_photocathode]*d_files[l_x_labels[i]]['a_samples'][:, dim_mean_num_photons] + d_files[l_x_labels[i]]['a_samples'][:, dim_prob_pe_from_dynode]*d_files[l_x_labels[i]]['a_samples'][:, dim_mean_num_photons], [16., 50., 84.])
    l_mnp_med[i] = current_med
    l_mnp_lb[i] = current_med - current_lb
    l_mnp_ub[i] = current_ub - current_med


for i, s_dataset in enumerate(l_x_labels):
    print '\nMean number of PE for %s: $%.3f^{+%.3f}_{-%.3f}$' % (s_dataset, l_mnp_med[i], np.absolute(l_mnp_ub[i]), np.absolute(l_mnp_lb[i]))

# plot prob that gamma hits first dynode
fig_mnp, ax_mnp = plt.subplots()
ax_mnp.errorbar(l_x, l_mnp_med, yerr=[l_mnp_lb, l_mnp_ub], fmt='b.', label='Cascade Model')
ax_mnp.set_xlabel('Dataset')
ax_mnp.set_ylabel('Occupancy')
ax_mnp.set_xticks(l_x)
ax_mnp.set_xticklabels(l_x_labels)
ax_mnp.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)



l_mnfa_med = [0 for i in xrange(num_x_pts)]
l_mnfa_lb = [0 for i in xrange(num_x_pts)]
l_mnfa_ub = [0 for i in xrange(num_x_pts)]


for i in xrange(num_x_pts):
    current_lb, current_med, current_ub = np.percentile(d_files[l_x_labels[i]]['a_samples'][:, dim_prob_pe_from_photocathode]*d_files[l_x_labels[i]]['a_samples'][:, dim_mean_num_photons], [16., 50., 84.])
    l_mnfa_med[i] = current_med
    l_mnfa_lb[i] = current_med - current_lb
    l_mnfa_ub[i] = current_ub - current_med


for i, s_dataset in enumerate(l_x_labels):
    print '\nMean number of Fully Amplified PE for %s: $%.3f^{+%.3f}_{-%.3f}$' % (s_dataset, l_mnfa_med[i], np.absolute(l_mnfa_ub[i]), np.absolute(l_mnfa_lb[i]))

# plot prob that gamma hits first dynode
fig_mnfa, ax_mnfa = plt.subplots()
ax_mnfa.errorbar(l_x, l_mnfa_med, yerr=[l_mnfa_lb, l_mnfa_ub], fmt='b.', label='Cascade Model')
ax_mnfa.set_xlabel('Dataset')
ax_mnfa.set_ylabel('Mean Number of PE from Photocathode')
ax_mnfa.set_xticks(l_x)
ax_mnfa.set_xticklabels(l_x_labels)
ax_mnfa.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)





s_path_to_save = './plots/summary_plots/'
if not os.path.isdir(s_path_to_save):
    os.makedirs(s_path_to_save)

s_description = 'nerix_cascade_summary'

fig_pfd.savefig('%s%s_prob_hit_first_dynode.png' % (s_path_to_save, s_description))
fig_icfd.savefig('%s%s_ionization_correction_from_first_dynode.png' % (s_path_to_save, s_description))
fig_pbc.savefig('%s%s_prob_bad_trajectory.png' % (s_path_to_save, s_description))
fig_icbt.savefig('%s%s_ionization_correction_from_bad_trajectory.png' % (s_path_to_save, s_description))
fig_mnp.savefig('%s%s_occupancy.png' % (s_path_to_save, s_description))
fig_mnfa.savefig('%s%s_num_pe_from_photocathode.png' % (s_path_to_save, s_description))

plt.tight_layout()
#plt.show()

#raw_input('Enter to continue...')
