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


df_el = pd.read_table('./resources/ElectronLifetimeDistribution_ScienceRun_wimps.txt', sep=' ', names=['el', 'counts'])

nb_el = len(df_el['el'])
bin_width_el = df_el['el'][1] - df_el['el'][0]
bin_edges_el = np.linspace(df_el['el'][0] - bin_width_el/2., df_el['el'][nb_el-1] + bin_width_el/2., nb_el+1)

fig_el, ax_el = plt.subplots(1)

ax_el.plot(df_el['el'], df_el['counts'], 'bo')

ax_el.set_title('Electron Lifetime - AmBe')
ax_el.set_xlabel(r'$\tau_{e^-} [\mu s]$')
ax_el.set_ylabel('$Counts$')

d_ambe_mc = {}
d_ambe_mc['a_el_hist'] = np.asarray(df_el['el'], dtype=np.float32)
d_ambe_mc['a_el_bins'] = np.asarray(bin_edges_el, dtype=np.float32)

fig_el.savefig('%selectron_lifetime_wimps.png' % (s_path_to_plots))

pickle.dump(d_ambe_mc, open('%swimp_mc.p' % (s_path_to_pickle_save), 'w'))

#plt.show()






