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
import config_xe1t
import numpy as np
import time, tqdm
import cPickle as pickle


s_path_to_corrections = './resources/FastLCEMap.pkl'
s_path_to_plots = './plots/supporting/lce_maps/'
s_path_to_pickle_save = './fit_inputs/'

d_correction_maps = pickle.load(open(s_path_to_corrections, 'rb'))

#print d_correction_maps['FastS2LCEXYMap'].keys()
# d_correction_maps: 'FastS1LCERZMap', 'FastS2LCEXYMap'
# keys in subdictionaries: ['map', 'ystep', 'xlower', 'xupper', 'xstep', 'ynbins', 'yupper', 'xnbins', 'ylower']

# S1 map is in R^2 vs Z
# S2 map is in X vs Y

#print d_correction_maps['FastS2LCEXYMap']['ynbins']
#print np.asarray(d_correction_maps['FastS2LCEXYMap']['map']).shape

# use pcolormesh to draw the map
# pcolormesh(X, Y, C, **kwargs)
# x and y should be the bin edges!
# C should be value at center


# --------------------------------------------
# --------------------------------------------
#  S1 Correction
# --------------------------------------------
# --------------------------------------------

s1_r2_bin_edges = np.linspace(d_correction_maps['FastS1LCERZMap']['xlower'], d_correction_maps['FastS1LCERZMap']['xupper'], d_correction_maps['FastS1LCERZMap']['xnbins'])
s1_z_bin_edges = np.linspace(d_correction_maps['FastS1LCERZMap']['ylower'], d_correction_maps['FastS1LCERZMap']['yupper'], d_correction_maps['FastS1LCERZMap']['ynbins'])

f_s1_correction, ax_s1_correction = plt.subplots(1)

s1_pcolor_data =  ax_s1_correction.pcolor(s1_r2_bin_edges, s1_z_bin_edges, d_correction_maps['FastS1LCERZMap']['map'])

ax_s1_correction.set_title('S1 Correction Map')
ax_s1_correction.set_xlabel('$R^2 [cm^2]$')
ax_s1_correction.set_ylabel('$Z [cm]$')
ax_s1_correction.set_ylim(s1_z_bin_edges[0], s1_z_bin_edges[-1])

f_s1_correction.colorbar(s1_pcolor_data, ax=ax_s1_correction)





# --------------------------------------------
# --------------------------------------------
#  S2 Correction
# --------------------------------------------
# --------------------------------------------

s2_x_bin_edges = np.linspace(d_correction_maps['FastS2LCEXYMap']['xlower'], d_correction_maps['FastS2LCEXYMap']['xupper'], d_correction_maps['FastS2LCEXYMap']['xnbins'])
s2_y_bin_edges = np.linspace(d_correction_maps['FastS2LCEXYMap']['ylower'], d_correction_maps['FastS2LCEXYMap']['yupper'], d_correction_maps['FastS2LCEXYMap']['ynbins'])

f_s2_correction, ax_s2_correction = plt.subplots(1)

s2_pcolor_data = ax_s2_correction.pcolormesh(s2_x_bin_edges, s2_y_bin_edges, d_correction_maps['FastS2LCEXYMap']['map'])

ax_s2_correction.set_title('S2 Correction Map')
ax_s2_correction.set_xlabel('$X [cm]$')
ax_s2_correction.set_ylabel('$Y [cm]$')

f_s2_correction.colorbar(s2_pcolor_data, ax=ax_s2_correction)

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

f_s1_correction.savefig('%ss1_pos_correction.png' % (s_path_to_plots))
f_s2_correction.savefig('%ss2_pos_correction.png' % (s_path_to_plots))

print s1_r2_bin_edges

d_corrections = {}
d_corrections['s1'] = {}
d_corrections['s1']['r2_bin_edges'] = s1_r2_bin_edges
d_corrections['s1']['z_bin_edges'] = s1_z_bin_edges
d_corrections['s1']['map'] = d_correction_maps['FastS1LCERZMap']['map']

d_corrections['s2'] = {}
d_corrections['s2']['x_bin_edges'] = s2_x_bin_edges
d_corrections['s2']['y_bin_edges'] = s2_y_bin_edges
d_corrections['s2']['map'] = d_correction_maps['FastS2LCEXYMap']['map']

pickle.dump(d_corrections, open('%ssignal_correction_maps.p' % (s_path_to_pickle_save), 'w'))

#plt.show()


