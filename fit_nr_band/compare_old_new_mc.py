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

import pandas as pd


s_path_to_pickle_save = './fit_inputs/'
s_path_to_plots = './plots/supporting/ambe_mc/'

fig_mc, ax_mc = plt.subplots(1)


s_path_to_input_old = './resources/ambe_mc_old.p'
df_ambe_mc_old = pickle.load(open(s_path_to_input_old, 'rb'))


df_ambe_mc_old['X'] = df_ambe_mc_old['X']/10.
df_ambe_mc_old['Y'] = df_ambe_mc_old['Y']/10.
df_ambe_mc_old['Z'] = df_ambe_mc_old['Z']/10.
df_ambe_mc_old['distance_to_source'] = ((df_ambe_mc_old['X']-55.96)**2. + (df_ambe_mc_old['Y']-43.72)**2. + (df_ambe_mc_old['Z']+50.)**2.)**0.5


df_ambe_mc_old = df_ambe_mc_old[(df_ambe_mc_old['Ed'] > 0) & (df_ambe_mc_old['Ed'] < 100)]

# AmBe optimized
#df_ambe_mc_old = df_ambe_mc_old[((df_ambe_mc_old['X']**2. + df_ambe_mc_old['Y']**2.) < config_xe1t.max_r**2.) & (df_ambe_mc_old['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_mc_old['Z']) & (df_ambe_mc_old['distance_to_source'] < 80.)]
#df_ambe_mc_old = df_ambe_mc_old[((df_ambe_mc_old['X']**2. + df_ambe_mc_old['Y']**2.) < config_xe1t.max_r**2.) & (df_ambe_mc_old['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_mc_old['Z']) & (df_ambe_mc_old['distance_to_source'] < 80.)]
#df_ambe_mc_old = df_ambe_mc_old[df_ambe_mc_old['FiducialVolumeAmBe']]

# cylinder
#df_ambe_mc_old = df_ambe_mc_old[((df_ambe_mc_old['X']**2. + df_ambe_mc_old['Y']**2.) < config_xe1t.max_r_cylinder**2.) & (df_ambe_mc_old['Z'] < config_xe1t.max_z_cylinder) & (config_xe1t.min_z_cylinder < df_ambe_mc_old['Z'])]




s_path_to_input_new = './resources/ambe_mc_new.p'
df_ambe_mc_new = pickle.load(open(s_path_to_input_new, 'rb'))


df_ambe_mc_new['X'] = df_ambe_mc_new['X']/10.
df_ambe_mc_new['Y'] = df_ambe_mc_new['Y']/10.
df_ambe_mc_new['Z'] = df_ambe_mc_new['Z']/10.
df_ambe_mc_new['distance_to_source'] = ((df_ambe_mc_new['X']-55.96)**2. + (df_ambe_mc_new['Y']-43.72)**2. + (df_ambe_mc_new['Z']+50.)**2.)**0.5


df_ambe_mc_new = df_ambe_mc_new[(df_ambe_mc_new['Ed'] > 0) & (df_ambe_mc_new['Ed'] < 100)]

# AmBe optimized
#df_ambe_mc_new = df_ambe_mc_new[((df_ambe_mc_new['X']**2. + df_ambe_mc_new['Y']**2.) < config_xe1t.max_r**2.) & (df_ambe_mc_new['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_mc_new['Z']) & (df_ambe_mc_new['distance_to_source'] < 80.)]
#df_ambe_mc_new = df_ambe_mc_new[((df_ambe_mc_new['X']**2. + df_ambe_mc_new['Y']**2.) < config_xe1t.max_r**2.) & (df_ambe_mc_new['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_mc_new['Z']) & (df_ambe_mc_new['distance_to_source'] < 80.)]
#df_ambe_mc_new = df_ambe_mc_new[df_ambe_mc_new['FiducialVolumeAmBe']]

# cylinder
#df_ambe_mc_new = df_ambe_mc_new[((df_ambe_mc_new['X']**2. + df_ambe_mc_new['Y']**2.) < config_xe1t.max_r_cylinder**2.) & (df_ambe_mc_new['Z'] < config_xe1t.max_z_cylinder) & (config_xe1t.min_z_cylinder < df_ambe_mc_new['Z'])]






d_ambe_mc = {}
d_ambe_mc['energy_old'] = np.asarray(df_ambe_mc_old['Ed'])
d_ambe_mc['energy_new'] = np.asarray(df_ambe_mc_new['Ed'])

nb_energy = 200
lb_energy = 0
ub_energy = 20

a_energy_hist_old, a_energy_bins = np.histogram(d_ambe_mc['energy_old'], bins=nb_energy, range=[lb_energy, ub_energy])
a_energy_hist_new, a_energy_bins = np.histogram(d_ambe_mc['energy_new'], bins=nb_energy, range=[lb_energy, ub_energy])

diff_hist = np.asarray((a_energy_hist_old - a_energy_hist_new), dtype=np.float32) / np.asarray(a_energy_hist_old, dtype=np.float32)

ax_mc.plot(a_energy_bins[:-1], diff_hist, 'bo')
ax_mc.set_title('AmBe Energy Spectrum - MC')
ax_mc.set_xlabel('$Energy [keV]$')
ax_mc.set_ylabel('(Old - New) / Old')
#ax_mc.set_yscale('log', nonposy='clip')

plt.show()

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_mc.savefig('%scompare_ambe_mc.png' % (s_path_to_plots))






