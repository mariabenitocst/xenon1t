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
import root_pandas

s_path_to_pickle_save = './fit_inputs/'
s_path_to_plots = './plots/supporting/ambe_data/'

l_range_s1 = [0, 100]
l_range_log = [1., 2.5]

# pre-170226
"""
s_path_to_input = './resources/XENON1T_ambe_band_nov.txt'
df_ambe_data = pd.read_table(s_path_to_input, sep='    ', header=None, index_col=False, names=['s1', 's2'])

df_ambe_data['s1'] = np.asarray(df_ambe_data['s1'], dtype=np.float32)
df_ambe_data['s2'] = np.asarray(df_ambe_data['s2'], dtype=np.float32)
"""

# 170226

#s_path_to_input = './resources/nr_band_160226.pkl'
#df_ambe_data = pickle.load(open(s_path_to_input, 'rb'))
#df_ambe_data['distance_to_source'] = ((df_ambe_data['x']-55.96)**2. + (df_ambe_data['y']-43.72)**2. + (df_ambe_data['z']+50.)**2.)**0.5
#print df_ambe_data


# no single electron cut (170418)
#s_path_to_input = './resources/ambe_no_s1_pulse_shape.txt'
#df_ambe_data = pd.read_table(s_path_to_input, sep='\t', )


#s_path_to_input = './resources/list_lax_NR_0.6.2.csv'
#s_path_to_input = './resources/list_lax_NR_0.8.4.csv'
#s_path_to_input = './resources/list_lax_NR_0.9.1.csv'
#s_path_to_input = './resources/list_lax_NR_0.9.2.csv'
#s_path_to_input = './resources/data_AmBe_lowenergy.csv'
#df_ambe_data = pd.read_table(s_path_to_input, sep=',')
#print df_ambe_data['x'], df_ambe_data['y'], df_ambe_data['z'], df_ambe_data['distance_to_source']

s_path_to_input = './resources/data_AmBe_cs1_lt_200.root'
df_ambe_data = root_pandas.read_root(s_path_to_input)

# AmBe optimized
df_ambe_data = df_ambe_data[((df_ambe_data['x']**2. + df_ambe_data['y']**2.) < config_xe1t.max_r**2.) & (df_ambe_data['z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_data['z']) & (df_ambe_data['distance_to_source'] < 80.)]


# apply cuts
#df_ambe_data = df_ambe_data[df_ambe_data['CutLowEnergyAmBe']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutAmBeFiducial']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS1LowEnergyRange']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS2Threshold']]
df_ambe_data = df_ambe_data[df_ambe_data['CutInteractionPeaksBiggest']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS2AreaFractionTop']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS2SingleScatterSimple']]
df_ambe_data = df_ambe_data[df_ambe_data['CutDAQVeto']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutEndOfRunCheck']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutBusyTypeCheck']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutBusyCheck']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutHEVCheck']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS1SingleScatter']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS1AreaFractionTop']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS2PatternLikelihood']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS2Tails']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS1PatternLikelihood']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS2Width']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutS2WidthHigh']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutS2WidthLow']]
df_ambe_data = df_ambe_data[df_ambe_data['CutS1MaxPMT']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutSingleElectronS2s']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutLowEnergyBackground']]
#df_ambe_data = df_ambe_data[df_ambe_data['CutPreS2Junk']]
#df_ambe_data = df_ambe_data[df_ambe_data['']]


# cylinder
#df_ambe_data = df_ambe_data[((df_ambe_data['x']**2. + df_ambe_data['y']**2.) < config_xe1t.max_r_cylinder**2.) & (df_ambe_data['z'] < config_xe1t.max_z_cylinder) & (config_xe1t.min_z_cylinder < df_ambe_data['z'])]
#df_ambe_data = df_ambe_data[((df_ambe_data['x']**2. + df_ambe_data['y']**2.) < config_xe1t.max_r**2.) & (df_ambe_data['z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_data['z'])]

# other cuts
df_ambe_data = df_ambe_data[(df_ambe_data['cs2_new'] > 200.) & (df_ambe_data['cs1'] > config_xe1t.l_s1_settings[1]) & (df_ambe_data['cs1'] < config_xe1t.l_s1_settings[2]) & (np.log10(df_ambe_data['cs2_bottom_new']/df_ambe_data['cs1']) > config_xe1t.l_log_settings[1]) & (np.log10(df_ambe_data['cs2_bottom_new']/df_ambe_data['cs1']) < config_xe1t.l_log_settings[2])]

df_ambe_data['s1'] = np.asarray(df_ambe_data['cs1'], dtype=np.float32)
df_ambe_data['s2'] = np.asarray(df_ambe_data['cs2_bottom_new'], dtype=np.float32)

print len(df_ambe_data['s1'])

fig_data, ax_data = plt.subplots(1)

ax_data.scatter(df_ambe_data['s1'], np.log10(df_ambe_data['s2']/df_ambe_data['s1']), marker='.')

ax_data.set_xlim(l_range_s1)
ax_data.set_ylim(l_range_log)
ax_data.set_title('AmBe Data Nov 2016')
ax_data.set_xlabel('$S1 [PE]$')
ax_data.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')

fig_data_xy, ax_data_xy = plt.subplots(1)

ax_data_xy.scatter(df_ambe_data['x'], df_ambe_data['y'], marker='.')
ax_data_xy.set_xlim(-config_xe1t.max_r, config_xe1t.max_r)
ax_data_xy.set_ylim(-config_xe1t.max_r, config_xe1t.max_r)
ax_data_xy.set_title('AmBe Data Nov 2016 - Position of Events')
ax_data_xy.set_xlabel('$X [cm]$')
ax_data_xy.set_ylabel(r'$Y [cm]$')

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_data.savefig('%sambe_data.png' % (s_path_to_plots))
fig_data_xy.savefig('%sambe_data_position.png' % (s_path_to_plots))

d_ambe_data = {}
d_ambe_data['s1'] = df_ambe_data['s1']
d_ambe_data['s2'] = df_ambe_data['s2']

pickle.dump(d_ambe_data, open('%sambe_data.p' % (s_path_to_pickle_save), 'w'))

#plt.show()
