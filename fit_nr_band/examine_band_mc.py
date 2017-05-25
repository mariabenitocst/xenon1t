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
from matplotlib.colors import LogNorm


s_path_to_pickle_save = './fit_inputs/'
s_path_to_plots = './plots/supporting/ambe_mc/'

fig_mc, ax_mc = plt.subplots(1)

"""
s_path_to_input = './resources/Xe1T_AmBe_NR_Spec.txt'
df_ambe_mc = pd.read_table(s_path_to_input, header=None, index_col=False, names=['energy', 'counts'])

d_ambe_mc = {}
d_ambe_mc['energy'] = np.asarray(df_ambe_mc['energy'])
d_ambe_mc['counts'] = np.asarray(df_ambe_mc['counts'])


ax_mc.plot(d_ambe_mc['energy'], d_ambe_mc['counts'], marker='.', linestyle='')
#ax_mc.hist(a_random_energies, bins=len(d_ambe_mc['energy']))

"""




#s_path_to_input = './resources/ambe_mc_170301.p'
#df_ambe_mc = pickle.load(open(s_path_to_input, 'rb'))

s_path_to_input = './resources/ambe_mc.p'
#s_path_to_input = './resources/ambe_mc_3mm.p'
#s_path_to_input = './resources/ambe_mc_matt.p'
df_ambe_mc = pd.DataFrame(pickle.load(open(s_path_to_input, 'rb')))


df_ambe_mc['X'] = df_ambe_mc['X']/10.
df_ambe_mc['Y'] = df_ambe_mc['Y']/10.
df_ambe_mc['Z'] = df_ambe_mc['Z']/10.
#df_ambe_mc['Z'] = df_ambe_mc['zpri']/10.
#print list(df_ambe_mc)
df_ambe_mc['distance_to_source'] = ((df_ambe_mc['X']-55.96)**2. + (df_ambe_mc['Y']-43.72)**2. + (df_ambe_mc['Z']+50.)**2.)**0.5

#print df_ambe_mc['Z']
#print df_ambe_mc['zpri']
#print df_ambe_mc['FiducialVolumeAmBe']

df_ambe_mc = df_ambe_mc[(df_ambe_mc['Ed'] > 0) & (df_ambe_mc['Ed'] < 100)]
#df_low_e = df_ambe_mc[(df_ambe_mc['Ed'] > 0.01) & (df_ambe_mc['Ed'] < 0.1)]
#print len(df_low_e)
#print df_low_e['Ed']

# AmBe optimized
df_ambe_mc = df_ambe_mc[((df_ambe_mc['X']**2. + df_ambe_mc['Y']**2.) < config_xe1t.detector_r**2.) & (df_ambe_mc['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_mc['Z']) & (df_ambe_mc['distance_to_source'] < 80.)]
#df_ambe_mc = df_ambe_mc[((df_ambe_mc['X']**2. + df_ambe_mc['Y']**2.) < config_xe1t.max_r**2.) & (df_ambe_mc['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_mc['Z']) & (df_ambe_mc['distance_to_source'] < 80.)]

#print len(df_ambe_mc['Ed'])

# cylinder
#df_ambe_mc = df_ambe_mc[((df_ambe_mc['X']**2. + df_ambe_mc['Y']**2.) < config_xe1t.max_r_cylinder**2.) & (df_ambe_mc['Z'] < config_xe1t.max_z_cylinder) & (config_xe1t.min_z_cylinder < df_ambe_mc['Z'])]
#df_ambe_mc = df_ambe_mc[((df_ambe_mc['X']**2. + df_ambe_mc['Y']**2.) < config_xe1t.max_r**2.) & (df_ambe_mc['Z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_mc['Z'])]

#print np.asarray(df_ambe_mc['X'], dtype=np.float32)
a_hist, bin_edges_energy, bin_edges_r = np.histogram2d(np.asarray(df_ambe_mc['Ed'], dtype=np.float32), np.asarray(np.asarray(df_ambe_mc['X'], dtype=np.float32)**2. + np.asarray(df_ambe_mc['Y'], dtype=np.float32)**2.), bins=40)
for i in xrange(a_hist.shape[0]):
    a_hist[:, i] = a_hist[:, i] / np.sum(a_hist[:, i])

#ax = plt.pcolormesh(bin_edges_energy, bin_edges_r, a_hist.T, norm=LogNorm())
#plt.show()


d_ambe_mc = {}
d_ambe_mc['energy'] = np.asarray(df_ambe_mc['Ed'])

nb_energy = 3000
lb_energy = 0
ub_energy = 100


# get linear histogram
a_energy_hist, a_energy_bins= np.histogram(d_ambe_mc['energy'], bins=nb_energy, range=[lb_energy, ub_energy])


# make plot for Patrick...
energy_bin_edges = np.logspace(-1, np.log10(ub_energy), 200)
ax_mc.hist(d_ambe_mc['energy'], bins=energy_bin_edges)

d_ambe_mc['a_energy_hist'] = a_energy_hist
d_ambe_mc['a_energy_bins'] = a_energy_bins


ax_mc.set_title('AmBe Energy Spectrum - MC')
ax_mc.set_xlabel('$Energy [keV]$')
ax_mc.set_ylabel('$Counts$')
ax_mc.set_yscale('log', nonposy='clip')

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_mc.savefig('%sambe_mc.png' % (s_path_to_plots))


df_el = pd.read_table('./resources/AmBe_elife_histo.txt', names=['el', 'counts'])

nb_el = len(df_el['el'])
bin_width_el = df_el['el'][1] - df_el['el'][0]
bin_edges_el = np.linspace(df_el['el'][0] - bin_width_el/2., df_el['el'][nb_el-1] + bin_width_el/2., nb_el+1)

fig_el, ax_el = plt.subplots(1)

ax_el.plot(df_el['el'], df_el['counts'], 'bo')

ax_el.set_title('Electron Lifetime - AmBe')
ax_el.set_xlabel(r'$\tau_{e^-} [\mu s]$')
ax_el.set_ylabel('$Counts$')

d_ambe_mc['a_el_hist'] = np.asarray(df_el['el'], dtype=np.float32)
d_ambe_mc['a_el_bins'] = np.asarray(bin_edges_el, dtype=np.float32)

d_ambe_mc['a_energy'] = np.asarray(df_ambe_mc['Ed'], dtype=np.float32)
d_ambe_mc['a_x'] = np.asarray(df_ambe_mc['X'], dtype=np.float32)
d_ambe_mc['a_y'] = np.asarray(df_ambe_mc['Y'], dtype=np.float32)
d_ambe_mc['a_z'] = np.asarray(df_ambe_mc['Z'], dtype=np.float32)

fig_el.savefig('%selectron_lifetime.png' % (s_path_to_plots))

pickle.dump(d_ambe_mc, open('%sambe_mc.p' % (s_path_to_pickle_save), 'w'))

#plt.show()






