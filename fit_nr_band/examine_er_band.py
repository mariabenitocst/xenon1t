#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

import ROOT as root
from rootpy.plotting import Hist, Hist2D, Canvas, Legend, Graph
from rootpy.io import root_open
import config_xe1t
import numpy as np
import time, tqdm
import cPickle as pickle

import neriX_analysis


s_path_to_input = './resources/PointEstimate_lax0.9.5_V10_170324_LogSpace.root'
#s_path_to_input = './resources/MCERBandForNRFitting_prel.root'
s_path_to_plots = './plots/supporting/er_band/'
s_path_to_pickle_save = './fit_inputs/'

f_er = root_open(s_path_to_input, 'read')

# MCERBandForNRFitting_prel.root
#h_er = f_er.hmcband

# PointEstimate_lax0.9.5_V10_170324_LogSpace.root
h_er = f_er.hband

#a_er = neriX_analysis.convert_2D_hist_to_matrix(h_er)
#num_random_events = int(np.sum(a_er))
num_random_events = int(1e7)

a_s1 = np.zeros(num_random_events, dtype=np.float32)
a_log = np.zeros(num_random_events, dtype=np.float32)

x = np.asarray(0, dtype=np.float64)
y = np.asarray(0, dtype=np.float64)

for i in tqdm.tqdm(xrange(num_random_events)):
    h_er.GetRandom2(x, y)
    s1 = float(x)
    log_s2_s1 = float(y)
    #print 'Correcting for S2 total'
    #s2 = 10**log_s2_s1 * s1
    #s2 /= 0.35
    #log_s2_s1 = np.log10(s2/s1)
    a_s1[i], a_log[i] = s1, log_s2_s1


s1_bin_edges = np.linspace(config_xe1t.l_s1_settings[1], config_xe1t.l_s1_settings[2], config_xe1t.l_s1_settings[0]+1)
log_bin_edges = np.linspace(config_xe1t.l_log_settings[1], config_xe1t.l_log_settings[2], config_xe1t.l_log_settings[0]+1)


fig_er, ax_er = plt.subplots(1)

a_er_band_hist, a_er_band_s1_edges, a_er_band_log_edges, color_data = ax_er.hist2d(a_s1, a_log, bins=[s1_bin_edges, log_bin_edges])

ax_er.set_xlabel('$S1 [PE]$')
ax_er.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')

fig_er.colorbar(color_data, ax=ax_er)

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_er.savefig('%ser_band.png' % (s_path_to_plots))

d_er_band = {}
d_er_band['er_band_hist'] = a_er_band_hist
d_er_band['er_band_s1_edges'] = a_er_band_s1_edges
d_er_band['er_band_log_edges'] = a_er_band_log_edges


pickle.dump(d_er_band, open('%ser_band.p' % (s_path_to_pickle_save), 'w'))


#plt.show()
