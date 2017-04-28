#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.interpolate import spline

import numpy as np
import cPickle as pickle
import tqdm

import config_xe1t

dir_specifier_name = 'run_0_band'

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


l_plots = ['plots', dir_specifier_name, '%s_kV_%s_deg' % (s_cathode_settings, s_degree_settings)]


s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)


d_arrays = pickle.load(open('./mc_output/signal_efficiency_bands.p', 'r'))

l_keys = ['threshold_only', 's1_cut_only', 's2_cut_only', 'pax_only', 'total_minus_threshold', 'total']

fig_acceptances, ax_acceptances = plt.subplots(1)

for label in l_keys:
    spline_y = spline(d_arrays[label]['bin_centers'], d_arrays[label]['a_prob'], d_arrays[label]['bin_centers'])
    ax_acceptances.plot(d_arrays[label]['bin_centers'], spline_y, color=d_arrays[label]['color'], linestyle=d_arrays[label]['linestyle'], label=d_arrays[label]['label'])

ax_acceptances.set_title(r'SR0 Signal Efficiencies')
ax_acceptances.set_xlabel(r'Energy [keV]')
ax_acceptances.set_ylabel(r'Signal Efficiency')
ax_acceptances.set_xscale('log')
ax_acceptances.set_yscale('log')

ax_acceptances.set_xlim(d_arrays[label]['bin_centers'][0], d_arrays[label]['bin_centers'][-1])
ax_acceptances.set_ylim(0, 1.05)

ax_acceptances.legend(loc='best', fontsize=10)

fig_acceptances.savefig('%ssignal_efficiencies_%.3f_kV.png' % (s_path_for_save, cathode_setting))



