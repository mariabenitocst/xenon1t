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

s_path_to_input = './resources/bias_smearing_wimps.p'
s_path_to_plots = './plots/supporting/bias_and_smearing_wimps/'
s_path_to_pickle_save = './fit_inputs/'

d_smearing_input = pickle.load(open(s_path_to_input, 'rb'))
# ['s2smearings', 's1smearings', 's1s', 's1bias', 's2s', 's2bias', 's2smearing', 's1smearing']




df_smearing_input = pd.DataFrame(d_smearing_input)
d_smearing_input = df_smearing_input[(df_smearing_input['s1s'] <= config_xe1t.l_s1_settings[2])]

num_pts_to_keep_spline = 50
step_size_s1 = int(len(d_smearing_input['s1s']) / num_pts_to_keep_spline)



d_smearing = {}
d_smearing['s1'] = {}
d_smearing['s1']['points'] = np.asarray(d_smearing_input['s1s'][::step_size_s1], dtype=np.float32)
d_smearing['s1']['lb_smearing'] = np.asarray(d_smearing_input['s1smearing_mins'][::step_size_s1], dtype=np.float32)
d_smearing['s1']['ub_smearing'] = np.asarray(d_smearing_input['s1smearing_maxs'][::step_size_s1], dtype=np.float32)
d_smearing['s1']['lb_bias'] = np.asarray(d_smearing_input['s1bias_mins'][::step_size_s1], dtype=np.float32)
d_smearing['s1']['ub_bias'] = np.asarray(d_smearing_input['s1bias_maxs'][::step_size_s1], dtype=np.float32)

# remake only looking at S2
d_smearing_input = df_smearing_input[(df_smearing_input['s2s'] <= config_xe1t.l_s2_settings[2])]

step_size_s2 = int(len(d_smearing_input['s2s']) / num_pts_to_keep_spline)

d_smearing['s2'] = {}
d_smearing['s2']['points'] = np.asarray(d_smearing_input['s2s'][::step_size_s2], dtype=np.float32)
d_smearing['s2']['lb_smearing'] = np.asarray(d_smearing_input['s2smearing_mins'][::step_size_s2], dtype=np.float32)
d_smearing['s2']['ub_smearing'] = np.asarray(d_smearing_input['s2smearing_maxs'][::step_size_s2], dtype=np.float32)
d_smearing['s2']['lb_bias'] = np.asarray(d_smearing_input['s2bias_mins'][::step_size_s2], dtype=np.float32)
d_smearing['s2']['ub_bias'] = np.asarray(d_smearing_input['s2bias_maxs'][::step_size_s2], dtype=np.float32)

print len(d_smearing['s1']['points']), len(d_smearing['s2']['points'])

fig_s1, (ax_s1_bias, ax_s1_smearing) = plt.subplots(1, 2, figsize=(12, 6))

ax_s1_bias.plot(d_smearing['s1']['points'], d_smearing['s1']['lb_bias'])
ax_s1_bias.plot(d_smearing['s1']['points'], d_smearing['s1']['ub_bias'])
ax_s1_smearing.plot(d_smearing['s1']['points'], d_smearing['s1']['lb_smearing'])
ax_s1_smearing.plot(d_smearing['s1']['points'], d_smearing['s1']['ub_smearing'])

ax_s1_bias.set_xlabel('$S1 [PE]$')
ax_s1_bias.set_ylabel(r'$Bias [\frac{processed - truth}{truth}]$')

ax_s1_smearing.set_xlabel('$S1 [PE]$')
ax_s1_smearing.set_ylabel(r'$Smearing [\frac{processed - truth}{truth}]$')



fig_s2, (ax_s2_bias, ax_s2_smearing) = plt.subplots(1, 2, figsize=(12, 6))

ax_s2_bias.plot(d_smearing['s2']['points'], d_smearing['s2']['lb_bias'])
ax_s2_bias.plot(d_smearing['s2']['points'], d_smearing['s2']['ub_bias'])
ax_s2_smearing.plot(d_smearing['s2']['points'], d_smearing['s2']['lb_smearing'])
ax_s2_smearing.plot(d_smearing['s2']['points'], d_smearing['s2']['ub_smearing'])

ax_s2_bias.set_xlabel('$S2 [PE]$')
ax_s2_bias.set_ylabel(r'$Bias [\frac{processed - truth}{truth}]$')

ax_s2_smearing.set_xlabel('$S2 [PE]$')
ax_s2_smearing.set_ylabel(r'$Smearing [\frac{processed - truth}{truth}]$')

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

#print d_smearing['s1']['points']

fig_s1.savefig('%ss1_bias_and_smearing_wimps.png' % (s_path_to_plots))
fig_s2.savefig('%ss2_bias_and_smearing_wimps.png' % (s_path_to_plots))

pickle.dump(d_smearing, open('%ss1_s2_bias_and_smearing_wimps.p' % (s_path_to_pickle_save), 'w'))

#plt.show()





