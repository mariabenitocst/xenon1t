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
d_files['0062_0061'] = {}
d_files['0062_0061']['voltage'] = 1600
d_files['0062_0061']['attenuation'] = 5e-6
d_files['0062_0061']['mi_spe_mean'] = 4.98e6
d_files['0062_0061']['mi_spe_mean_unc'] = 2e4
d_files['0062_0061']['mi_spe_width'] = 2.71e6
d_files['0062_0061']['mi_spe_width_unc'] = 2.4e4
d_files['0062_0061']['mi_mean_num_pe'] = 1.324
d_files['0062_0061']['mi_mean_num_pe_unc'] = 0.005
d_files['0062_0061']['cascade_spe_mean'] = 5.12e6
d_files['0062_0061']['cascade_spe_mean_unc'] = 3.06e4
d_files['0062_0061']['cascade_spe_width'] = 2.66e6
d_files['0062_0061']['cascade_spe_width_unc'] = 1.91e4

d_files['0066_0065'] = {}
d_files['0066_0065']['voltage'] = 1500
d_files['0066_0065']['attenuation'] = 5e-6
d_files['0066_0065']['mi_spe_mean'] = 3.10e6
d_files['0066_0065']['mi_spe_mean_unc'] = 1e4
d_files['0066_0065']['mi_spe_width'] = 1.56e6
d_files['0066_0065']['mi_spe_width_unc'] = 2e4
d_files['0066_0065']['mi_mean_num_pe'] = 1.289
d_files['0066_0065']['mi_mean_num_pe_unc'] = 0.005
d_files['0066_0065']['cascade_spe_mean'] = 3.17e6
d_files['0066_0065']['cascade_spe_mean_unc'] = 2.45e4
d_files['0066_0065']['cascade_spe_width'] = 1.56e6
d_files['0066_0065']['cascade_spe_width_unc'] = 1.33e4

d_files['0067_0068'] = {}
d_files['0067_0068']['voltage'] = 1400
d_files['0067_0068']['attenuation'] = 5e-6
d_files['0067_0068']['mi_spe_mean'] = 1.88e6
d_files['0067_0068']['mi_spe_mean_unc'] = 7e3
d_files['0067_0068']['mi_spe_width'] = 8.64e5
d_files['0067_0068']['mi_spe_width_unc'] = 9.7e3
d_files['0067_0068']['mi_mean_num_pe'] = 1.257
d_files['0067_0068']['mi_mean_num_pe_unc'] = 0.005
d_files['0067_0068']['cascade_spe_mean'] = 1.87e6
d_files['0067_0068']['cascade_spe_mean_unc'] = 1.54e5
d_files['0067_0068']['cascade_spe_width'] = 9.05e5
d_files['0067_0068']['cascade_spe_width_unc'] = 9.13e4

d_files['0071_0072'] = {}
d_files['0071_0072']['voltage'] = 1700
d_files['0071_0072']['attenuation'] = 5e-6
d_files['0071_0072']['mi_spe_mean'] = 7.88e6
d_files['0071_0072']['mi_spe_mean_unc'] = 2e4
d_files['0071_0072']['mi_spe_width'] = 4.49e6
d_files['0071_0072']['mi_spe_width_unc'] = 5e4
d_files['0071_0072']['mi_mean_num_pe'] = 1.351
d_files['0071_0072']['mi_mean_num_pe_unc'] = 0.005
d_files['0071_0072']['cascade_spe_mean'] = 8.03e6
d_files['0071_0072']['cascade_spe_mean_unc'] = 4.24e4
d_files['0071_0072']['cascade_spe_width'] = 4.26e6
d_files['0071_0072']['cascade_spe_width_unc'] = 2.60e4

d_files['0073_0074'] = {}
d_files['0073_0074']['voltage'] = 1700
d_files['0073_0074']['attenuation'] = 1e-5
d_files['0073_0074']['mi_spe_mean'] = 7.90e6
d_files['0073_0074']['mi_spe_mean_unc'] = 2e4
d_files['0073_0074']['mi_spe_width'] = 4.51e6
d_files['0073_0074']['mi_spe_width_unc'] = 5e4
d_files['0073_0074']['mi_mean_num_pe'] = 2.395
d_files['0073_0074']['mi_mean_num_pe_unc'] = 0.008
d_files['0073_0074']['cascade_spe_mean'] = 7.88e6
d_files['0073_0074']['cascade_spe_mean_unc'] = 7.97e4
d_files['0073_0074']['cascade_spe_width'] = 4.34e6
d_files['0073_0074']['cascade_spe_width_unc'] = 4.76e4



for filename in d_files:
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

l_x_labels = ['0067_0068', '0066_0065', '0062_0061', '0071_0072', '0073_0074']
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

l_mnp_med_mi = [0 for i in xrange(num_x_pts)]
l_mnp_unc_mi = [0 for i in xrange(num_x_pts)]

for i in xrange(num_x_pts):
    current_lb, current_med, current_ub = np.percentile(d_files[l_x_labels[i]]['a_samples'][:, dim_prob_pe_from_photocathode]*d_files[l_x_labels[i]]['a_samples'][:, dim_mean_num_photons] + d_files[l_x_labels[i]]['a_samples'][:, dim_prob_pe_from_dynode]*d_files[l_x_labels[i]]['a_samples'][:, dim_mean_num_photons], [16., 50., 84.])
    l_mnp_med[i] = current_med
    l_mnp_lb[i] = current_med - current_lb
    l_mnp_ub[i] = current_ub - current_med

    l_mnp_med_mi[i] = d_files[l_x_labels[i]]['mi_mean_num_pe']
    l_mnp_unc_mi[i] = d_files[l_x_labels[i]]['mi_mean_num_pe_unc']

for i, s_dataset in enumerate(l_x_labels):
    print '\nMean number of PE for %s: $%.3f^{+%.3f}_{-%.3f}$' % (s_dataset, l_mnp_med[i], np.absolute(l_mnp_ub[i]), np.absolute(l_mnp_lb[i]))

# plot prob that gamma hits first dynode
fig_mnp, ax_mnp = plt.subplots()
ax_mnp.errorbar(l_x, l_mnp_med_mi, yerr=[l_mnp_unc_mi, l_mnp_unc_mi], fmt='r.', label='Model Independent')
ax_mnp.errorbar(l_x, l_mnp_med, yerr=[l_mnp_lb, l_mnp_ub], fmt='b.', label='Cascade Model')
ax_mnp.set_xlabel('Dataset')
ax_mnp.set_ylabel('Occupancy')
ax_mnp.set_xticks(l_x)
ax_mnp.set_xticklabels(l_x_labels)
ax_mnp.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)
ax_mnp.legend(loc='best')



l_mnfa_med = [0 for i in xrange(num_x_pts)]
l_mnfa_lb = [0 for i in xrange(num_x_pts)]
l_mnfa_ub = [0 for i in xrange(num_x_pts)]

l_mnfa_med_mi = [0 for i in xrange(num_x_pts)]
l_mnfa_unc_mi = [0 for i in xrange(num_x_pts)]

for i in xrange(num_x_pts):
    current_lb, current_med, current_ub = np.percentile(d_files[l_x_labels[i]]['a_samples'][:, dim_prob_pe_from_photocathode]*d_files[l_x_labels[i]]['a_samples'][:, dim_mean_num_photons]*d_files[l_x_labels[i]]['a_samples'][:, dim_prob_bad_collection], [16., 50., 84.])
    l_mnfa_med[i] = current_med
    l_mnfa_lb[i] = current_med - current_lb
    l_mnfa_ub[i] = current_ub - current_med

    l_mnfa_med_mi[i] = d_files[l_x_labels[i]]['mi_mean_num_pe']
    l_mnfa_unc_mi[i] = d_files[l_x_labels[i]]['mi_mean_num_pe_unc']

for i, s_dataset in enumerate(l_x_labels):
    print '\nMean number of Fully Amplified PE for %s: $%.3f^{+%.3f}_{-%.3f}$' % (s_dataset, l_mnfa_med[i], np.absolute(l_mnfa_ub[i]), np.absolute(l_mnfa_lb[i]))

# plot prob that gamma hits first dynode
fig_mnfa, ax_mnfa = plt.subplots()
ax_mnfa.errorbar(l_x, l_mnfa_med_mi, yerr=[l_mnfa_unc_mi, l_mnfa_unc_mi], fmt='r.', label='Model Independent')
ax_mnfa.errorbar(l_x, l_mnfa_med, yerr=[l_mnfa_lb, l_mnfa_ub], fmt='b.', label='Cascade Model')
ax_mnfa.set_xlabel('Dataset')
ax_mnfa.set_ylabel('Mean Number of PE from Photocathode')
ax_mnfa.set_xticks(l_x)
ax_mnfa.set_xticklabels(l_x_labels)
ax_mnfa.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)
ax_mnfa.legend(loc='best')





l_gain_med = [0 for i in xrange(num_x_pts)]
l_gain_unc = [0 for i in xrange(num_x_pts)]

l_gain_med_mi = [0 for i in xrange(num_x_pts)]
l_gain_unc_mi = [0 for i in xrange(num_x_pts)]

for i in xrange(num_x_pts):
    l_gain_med[i] = d_files[l_x_labels[i]]['cascade_spe_mean']
    l_gain_unc[i] = d_files[l_x_labels[i]]['cascade_spe_mean_unc']

    l_gain_med_mi[i] = d_files[l_x_labels[i]]['mi_spe_mean']
    l_gain_unc_mi[i] = d_files[l_x_labels[i]]['mi_spe_mean_unc']

for i, s_dataset in enumerate(l_x_labels):
    print '\nGain for %s: %.3e +/- %.3e' % (s_dataset, l_gain_med[i], np.absolute(l_gain_unc[i]))

fig_gain, ax_gain = plt.subplots()
ax_gain.errorbar(l_x, l_gain_med_mi, yerr=[l_gain_unc_mi, l_gain_unc_mi], fmt='r.', label='Model Independent')
ax_gain.errorbar(l_x, l_gain_med, yerr=[l_gain_unc, l_gain_unc], fmt='b.', label='Cascade Model')
ax_gain.set_xlabel('Dataset')
ax_gain.set_ylabel('Mean of SPE Response')
ax_gain.set_xticks(l_x)
ax_gain.set_xticklabels(l_x_labels)
ax_gain.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)
ax_gain.legend(loc='best')



l_std_med = [0 for i in xrange(num_x_pts)]
l_std_unc = [0 for i in xrange(num_x_pts)]

l_std_med_mi = [0 for i in xrange(num_x_pts)]
l_std_unc_mi = [0 for i in xrange(num_x_pts)]

for i in xrange(num_x_pts):
    l_std_med[i] = d_files[l_x_labels[i]]['cascade_spe_width']
    l_std_unc[i] = d_files[l_x_labels[i]]['cascade_spe_width_unc']

    l_std_med_mi[i] = d_files[l_x_labels[i]]['mi_spe_width']
    l_std_unc_mi[i] = d_files[l_x_labels[i]]['mi_spe_width_unc']

for i, s_dataset in enumerate(l_x_labels):
    print '\nStd for %s: %.3e +/- %.3e' % (s_dataset, l_std_med[i], np.absolute(l_std_unc[i]))

fig_std, ax_std = plt.subplots()
ax_std.errorbar(l_x, l_std_med_mi, yerr=[l_std_unc_mi, l_std_unc_mi], fmt='r.', label='Model Independent')
ax_std.errorbar(l_x, l_std_med, yerr=[l_std_unc, l_std_unc], fmt='b.', label='Cascade Model')
ax_std.set_xlabel('Dataset')
ax_std.set_ylabel('Standard Deviation of SPE Response')
ax_std.set_xticks(l_x)
ax_std.set_xticklabels(l_x_labels)
ax_std.set_xlim(l_x[0]-0.5, l_x[-1]+0.5)
ax_std.legend(loc='best')



s_path_to_save = './plots/summary_plots/'
if not os.path.isdir(s_path_to_save):
    os.makedirs(s_path_to_save)

s_description = 'cascade_summary'

fig_pfd.savefig('%s%s_prob_hit_first_dynode.png' % (s_path_to_save, s_description))
fig_icfd.savefig('%s%s_ionization_correction_from_first_dynode.png' % (s_path_to_save, s_description))
fig_pbc.savefig('%s%s_prob_bad_trajectory.png' % (s_path_to_save, s_description))
fig_icbt.savefig('%s%s_ionization_correction_from_bad_trajectory.png' % (s_path_to_save, s_description))
fig_mnp.savefig('%s%s_occupancy.png' % (s_path_to_save, s_description))
fig_mnfa.savefig('%s%s_num_pe_from_photocathode.png' % (s_path_to_save, s_description))
fig_gain.savefig('%s%s_gain.png' % (s_path_to_save, s_description))
fig_std.savefig('%s%s_std.png' % (s_path_to_save, s_description))

plt.tight_layout()
#plt.show()

#raw_input('Enter to continue...')
