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

s_path_to_input = './resources/nr_band_160226.pkl'
df_ambe_data = pickle.load(open(s_path_to_input, 'rb'))
#print df_ambe_data

df_ambe_data = df_ambe_data[((df_ambe_data['x']**2. + df_ambe_data['y']**2.) < config_xe1t.max_r**2.) & (df_ambe_data['z'] < config_xe1t.max_z) & (config_xe1t.min_z < df_ambe_data['z'])]
df_ambe_data = df_ambe_data[(df_ambe_data['cs2'] > 150.) & (df_ambe_data['cs1'] > config_xe1t.l_s1_settings[1]) & (df_ambe_data['cs1'] < config_xe1t.l_s1_settings[2]) & (np.log10(df_ambe_data['cs2']/df_ambe_data['cs1']) > config_xe1t.l_log_settings[1]) & (np.log10(df_ambe_data['cs2']/df_ambe_data['cs1']) < config_xe1t.l_log_settings[2])]

df_ambe_data['s1'] = np.asarray(df_ambe_data['cs1'], dtype=np.float32)
df_ambe_data['s2'] = np.asarray(df_ambe_data['cs2']*(1-df_ambe_data['s2_area_fraction_top']), dtype=np.float32)

fig_data, ax_data = plt.subplots(1)

ax_data.scatter(df_ambe_data['s1'], np.log10(df_ambe_data['s2']/df_ambe_data['s1']), marker='.')

ax_data.set_xlim(l_range_s1)
ax_data.set_ylim(l_range_log)
ax_data.set_title('AmBe Data Nov 2016')
ax_data.set_xlabel('$S1 [PE]$')
ax_data.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_data.savefig('%sambe_data.png' % (s_path_to_plots))

d_ambe_data = {}
d_ambe_data['s1'] = df_ambe_data['s1']
d_ambe_data['s2'] = df_ambe_data['s2']

pickle.dump(d_ambe_data, open('%sambe_data.p' % (s_path_to_pickle_save), 'w'))

#plt.show()
