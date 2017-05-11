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

import pandas as pd
import scipy.interpolate

b_wimp_input = False

if not b_wimp_input:
    #s_path_to_input = './resources/sr0_efficiency_ambe_conditions.csv'
    #s_path_to_input = './resources/sr0_efficiency_ambe_threshold_normal_noise.csv'
    s_path_to_input = './resources/sr0_efficiency_ambe_conditions_goodnoise.csv'
else:
    s_path_to_input = './resources/sr0_efficiency_weightedaverage_conditions.csv'

s_path_to_plots = './plots/supporting/acceptances/'
s_path_to_pickle_save = './fit_inputs/'



df_acceptance = pd.read_csv(s_path_to_input)


d_acceptances = {}
d_acceptances['pf_s1'] = {}
d_acceptances['pf_s1']['x_values'] = np.asarray(df_acceptance['photons_detected'], dtype=np.float32)
d_acceptances['pf_s1']['y_values_lower'] = np.asarray(df_acceptance['minimum'], dtype=np.float32)
d_acceptances['pf_s1']['y_values_mean'] = np.asarray(df_acceptance['median'], dtype=np.float32)
d_acceptances['pf_s1']['y_values_upper'] = np.asarray(df_acceptance['maximum'], dtype=np.float32)




fig_pf_s1, ax_pf_s1 = plt.subplots(1)
ax_pf_s1.fill_between(d_acceptances['pf_s1']['x_values'], d_acceptances['pf_s1']['y_values_lower'], d_acceptances['pf_s1']['y_values_upper'], alpha=0.2)
ax_pf_s1.plot(d_acceptances['pf_s1']['x_values'], d_acceptances['pf_s1']['y_values_mean'], 'b--')

#print 'adding line at 4.5 sigma'
#temp_line = d_acceptances['pf_s1']['y_values_mean'] + 4.5*(d_acceptances['pf_s1']['y_values_upper']-d_acceptances['pf_s1']['y_values_mean'])
#ax_pf_s1.plot(d_acceptances['pf_s1']['x_values'], temp_line, 'r--')

#ax_pf_s1.set_xlim(min(d_acceptances['pf_s1']['x_values']), max(d_acceptances['pf_s1']['x_values']))
ax_pf_s1.set_ylim(0, 1.03)

ax_pf_s1.set_xlim(0, 25)
ax_pf_s1.set_xlabel('$S1 [PE]$')
ax_pf_s1.set_ylabel('$Acceptance$')

ax_pf_s1.legend(loc='best', fontsize=10)

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

if not b_wimp_input:
    fig_pf_s1.savefig('%spf_acceptance_s1_ambe.png' % (s_path_to_plots))
else:
    fig_pf_s1.savefig('%spf_acceptance_s1_wimps.png' % (s_path_to_plots))

print d_acceptances['pf_s1']['x_values']
print d_acceptances['pf_s1']['y_values_lower']

if not b_wimp_input:
    pickle.dump(d_acceptances, open('%sacceptances_ambe.p' % (s_path_to_pickle_save), 'w'))
else:
    pickle.dump(d_acceptances, open('%sacceptances_wimps.p' % (s_path_to_pickle_save), 'w'))


#plt.show()
