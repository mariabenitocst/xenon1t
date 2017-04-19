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


s_path_to_plots = './plots/supporting/cut_acceptances/'
s_path_to_pickle_save = './fit_inputs/'


l_s1 = [0, 20, 20.01, 40, 40.01, 60, 60.01, 80, 80.01, 100, 100.01, 120]
l_s1_acceptance = [0.96596223, 0.96596223, 0.86819284, 0.86819284, 0.93647704, 0.93647704, 0.8764984, 0.8764984, 0.81591983, 0.81591983, 0.86688329, 0.86688329]

l_s2 = [150, 950, 950.01, 1550, 1550.01, 2350, 2350.01, 3150, 3150.01, 3950, 3950.01, 4750]
l_s2_acceptance = [0.7345679, 0.7345679, 0.87194771, 0.87194771, 0.83202406, 0.83202406, 0.87874901, 0.87874901, 0.83279339, 0.83279339, 0.88883064, 0.88883064]

assert len(l_s1) == len(l_s1_acceptance)
assert len(l_s2) == len(l_s2_acceptance)



f_s1_acceptance, ax_s1_acceptance = plt.subplots(1)

ax_s1_acceptance.plot(l_s1, l_s1_acceptance, 'b-')

ax_s1_acceptance.set_title('S1 Cut Acceptance')
ax_s1_acceptance.set_xlabel('S1 [PE]')
ax_s1_acceptance.set_ylabel('Acceptance')
ax_s1_acceptance.set_ylim(0, 1.03)






# --------------------------------------------
# --------------------------------------------
#  S2 Correction
# --------------------------------------------
# --------------------------------------------



f_s2_acceptance, ax_s2_acceptance = plt.subplots(1)

ax_s2_acceptance.plot(l_s2, l_s2_acceptance, 'r-')

ax_s2_acceptance.set_title('S2 Cut Acceptance')
ax_s2_acceptance.set_xlabel('S2 [PE]')
ax_s2_acceptance.set_ylabel('Acceptance')
ax_s2_acceptance.set_ylim(0, 1.03)


if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

f_s1_acceptance.savefig('%ss1_cut_acceptance.png' % (s_path_to_plots))
f_s2_acceptance.savefig('%ss2_cut_acceptance.png' % (s_path_to_plots))


d_cut_acceptances = {}
d_cut_acceptances['s1'] = {}
d_cut_acceptances['s1']['spline_pts'] = np.asarray(l_s1, dtype=np.float32)
d_cut_acceptances['s1']['acceptance'] = np.asarray(l_s1_acceptance, dtype=np.float32)

d_cut_acceptances['s2'] = {}
d_cut_acceptances['s2']['spline_pts'] = np.asarray(l_s2, dtype=np.float32)
d_cut_acceptances['s2']['acceptance'] = np.asarray(l_s2_acceptance, dtype=np.float32)

#plt.show()

pickle.dump(d_cut_acceptances, open('%scut_acceptances.p' % (s_path_to_pickle_save), 'w'))



