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

import corner
from sklearn import neighbors
from sklearn import grid_search
from sklearn import preprocessing


"""
s_path_to_input = './resources/FastLCEMap.pkl'
s_path_to_plots = './plots/supporting/lce_maps/'
s_path_to_pickle_save = './fit_inputs/'

d_mc_maps = pickle.load(open(s_path_to_input, 'rb'))
"""

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


# ---------------------------------------------
# ---------------------------------------------
#  Read in from MC
# ---------------------------------------------
# ---------------------------------------------

s_path_to_input = './resources/ambe_mc.p'
s_path_to_plots = './plots/supporting/mc_map/'
s_path_to_pickle_save = './fit_inputs/'

d_mc_maps_full = pickle.load(open(s_path_to_input, 'rb'))

print list(d_mc_maps_full)

# convert to cm
d_mc_maps_full['X'] = d_mc_maps_full['X']/10.
d_mc_maps_full['Y'] = d_mc_maps_full['Y']/10.
d_mc_maps_full['Z'] = d_mc_maps_full['Z']/10.
d_mc_maps_full['distance_to_source'] = ((d_mc_maps_full['X']-55.96)**2. + (d_mc_maps_full['Y']-43.72)**2. + (d_mc_maps_full['Z']+50.)**2.)**0.5

# AmBe optimized
d_mc_maps_full = d_mc_maps_full[(d_mc_maps_full['Ed'] < 100) & ((d_mc_maps_full['X']**2. + d_mc_maps_full['Y']**2.) < config_xe1t.max_r**2.) & (d_mc_maps_full['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < d_mc_maps_full['Z']) & (d_mc_maps_full['distance_to_source'] < 80.)]

# cylinder
#d_mc_maps_full = d_mc_maps_full[(d_mc_maps_full['Ed'] < 100) & ((d_mc_maps_full['X']**2. + d_mc_maps_full['Y']**2.) < config_xe1t.max_r**2.) & (d_mc_maps_full['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < d_mc_maps_full['Z'])]
#d_mc_maps_full = d_mc_maps_full[(d_mc_maps_full['Ed'] < 100) & ((d_mc_maps_full['X']**2. + d_mc_maps_full['Y']**2.) < config_xe1t.max_r**2.) & (d_mc_maps_full['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < d_mc_maps_full['Z'])]

d_mc_maps = {}
d_mc_maps['energy'] = np.asarray(d_mc_maps_full['Ed'], dtype=np.float32)
d_mc_maps['x'] = np.asarray(d_mc_maps_full['X'], dtype=np.float32)
d_mc_maps['y'] = np.asarray(d_mc_maps_full['Y'], dtype=np.float32)
d_mc_maps['z'] = np.asarray(d_mc_maps_full['Z'], dtype=np.float32)

a_samples = np.asarray([d_mc_maps['energy'], d_mc_maps['x'], d_mc_maps['y'], d_mc_maps['z']])

fig_corner = corner.corner(a_samples.T, labels=['Energy [keV]', 'X [cm]', 'Y [cm]', 'Z [cm]'])



num_bins_x = 100
num_bins_y = 100
num_bins_z = 100


a_x_y_map, a_x_bins, a_y_bins = np.histogram2d(d_mc_maps['x'], d_mc_maps['y'], bins=[num_bins_x, num_bins_y])

fig_xy, ax_xy = plt.subplots(1)

ax_xy.pcolormesh(a_x_bins, a_y_bins, a_x_y_map.T)

ax_xy.set_xlim(np.min(d_mc_maps['x']), np.max(d_mc_maps['x']))
ax_xy.set_ylim(np.min(d_mc_maps['y']), np.max(d_mc_maps['y']))
ax_xy.set_xlabel('X [cm]')
ax_xy.set_ylabel('Y [cm]')


fig_z, ax_z = plt.subplots(1)

a_z_map, a_z_bins, _ = ax_z.hist(d_mc_maps['z'], bins=num_bins_z)

ax_z.set_xlim(np.min(d_mc_maps['z']), np.max(d_mc_maps['z']))
ax_z.set_xlabel('Z [cm]')
ax_z.set_ylabel('Counts')


d_mc_maps['xy_map'] = a_x_y_map
d_mc_maps['x_bin_edges'] = a_x_bins
d_mc_maps['y_bin_edges'] = a_y_bins

d_mc_maps['z_map'] = a_z_map
d_mc_maps['z_bin_edges'] = a_z_bins

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_corner.savefig('%sxyz_corner_plot.png' % (s_path_to_plots))
fig_xy.savefig('%sxy_map.png' % (s_path_to_plots))
fig_z.savefig('%sz_map.png' % (s_path_to_plots))


pickle.dump(d_mc_maps, open('%smc_maps.p' % (s_path_to_pickle_save), 'w'))


#plt.show()


"""
num_draws_from_kde = 10000

scaler = preprocessing.StandardScaler()
scaler.fit(a_samples.T)
a_scaled_samples = scaler.transform(a_samples.T)

#print a_sampler[:,1:3]
#print a_scaled_samples

# find the best fit bandwith since this allows us
# to play with bias vs variance
grid = grid_search.GridSearchCV(neighbors.KernelDensity(), {'bandwidth':np.linspace(0.01, 0.25, 20)}, cv=4, verbose=1, n_jobs=4)
print '\nDetermining best bandwidth...\n'
grid.fit(a_scaled_samples)
#print grid.best_estimator_

kde = neighbors.KernelDensity(**grid.best_params_)
kde.fit(a_scaled_samples)

a_random_samples = kde.sample(num_draws_from_kde)
a_random_samples = scaler.inverse_transform(a_random_samples)

fig_corner_kde = corner.corner(a_random_samples, labels=['Energy [keV]', 'X [cm]', 'Y [cm]', 'Z [cm]'])

plt.show()

d_mc_inputs = {}
d_mc_inputs['scaler'] = scaler
d_mc_inputs['kde'] = kde

pickle.dump(d_mc_inputs, open('%smc_inputs.p' % (s_path_to_pickle_save), 'w'))

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_corner.savefig('%smc_corner_plot.png' % (s_path_to_plots))
fig_corner_kde.savefig('%smc_kde_corner_plot.png' % (s_path_to_plots))
"""


# ---------------------------------------------
# ---------------------------------------------
#  Create fake histogram for practice in code
# ---------------------------------------------
# ---------------------------------------------

"""
num_events_for_hist = 10000
num_bins_r2 = 30
num_bins_z = 30

a_r2_pos = np.random.normal(loc=1000, scale=50, size=num_events_for_hist)
a_r2_bins = np.linspace(0, 1600, num_bins_r2+1)
r2_bin_width = a_r2_bins[1] - a_r2_bins[0]

a_z_pos = np.random.normal(loc=-50, scale=10, size=num_events_for_hist)
a_z_bins = np.linspace(-95, -10, num_bins_z+1)
z_bin_width = a_z_bins[1] - a_z_bins[0]

a_r2_z_map, _, _ = np.histogram2d(a_r2_pos, a_z_pos, bins=[a_r2_bins, a_z_bins])
"""





#plt.pcolormesh(a_r2_bins, a_z_bins, a_r2_z_map)
#plt.show()

"""
cdf = np.cumsum(a_r2_z_map.ravel())
cdf = cdf / cdf[-1]

values = np.random.rand(num_events_for_hist)
value_bins = np.searchsorted(cdf, values)

a_r2_idx, a_z_idx = np.unravel_index(value_bins, (num_bins_r2, num_bins_z))

for current_r2_idx, current_z_idx in zip(a_r2_idx, a_z_idx):
    print current_r2_idx, current_z_idx
"""








#pickle.dump(d_corrections, open('%ssignal_correction_maps.p' % (s_path_to_pickle_save), 'w'))

#plt.show()


