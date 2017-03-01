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


s_path_to_input = './resources/AccForXe1T.root'
s_path_to_plots = './plots/supporting/acceptances/'
s_path_to_pickle_save = './fit_inputs/'

f_acceptance = root_open(s_path_to_input, 'read')
"""
KEY: TGraphAsymmErrors	gus1acc;1	
KEY: TGraphAsymmErrors	gs1acc;1
KEY: TGraphAsymmErrors	gs2acc;1	
KEY: TF1	fus1acc;1	[0]
KEY: TGraphAsymmErrors	gpeakfindingacc_lower;1	
KEY: TGraphAsymmErrors	gpeakfindingacc_upper;1
"""


# will record points for spline in dictionary
# will record upper and lower bounds
# will use single RV to draw certain distance between them
# ex: acc = lower + f * (upper-lower)
num_pts = f_acceptance.gpeakfindingacc_lower.GetN()
print 'Number of points: %d\n' % (num_pts)

d_acceptances = {}
d_acceptances['pf_s1'] = {}
d_acceptances['pf_s1']['x_values'] = np.zeros(num_pts)
d_acceptances['pf_s1']['y_values_lower'] = np.zeros(num_pts)
d_acceptances['pf_s1']['y_values_mean'] = np.zeros(num_pts)
d_acceptances['pf_s1']['y_values_upper'] = np.zeros(num_pts)

for i in xrange(num_pts):
    d_acceptances['pf_s1']['x_values'][i] = f_acceptance.gpeakfindingacc_lower.GetX()[i]
    d_acceptances['pf_s1']['y_values_lower'][i] = f_acceptance.gpeakfindingacc_lower.GetY()[i]
    d_acceptances['pf_s1']['y_values_mean'][i] = f_acceptance.gpeakfindingacc_mean.GetY()[i]
    d_acceptances['pf_s1']['y_values_upper'][i] = f_acceptance.gpeakfindingacc_upper.GetY()[i]

fig_pf_s1, ax_pf_s1 = plt.subplots(1)
ax_pf_s1.fill_between(d_acceptances['pf_s1']['x_values'], d_acceptances['pf_s1']['y_values_lower'], d_acceptances['pf_s1']['y_values_upper'], alpha=0.2)
ax_pf_s1.plot(d_acceptances['pf_s1']['x_values'], d_acceptances['pf_s1']['y_values_mean'], 'b--')

ax_pf_s1.set_xlim(min(d_acceptances['pf_s1']['x_values']), max(d_acceptances['pf_s1']['x_values']))
ax_pf_s1.set_ylim(0, 1.03)

ax_pf_s1.set_xlabel('$S1 [PE]$')
ax_pf_s1.set_ylabel('$Acceptance$')

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_pf_s1.savefig('%spf_acceptance_s1.png' % (s_path_to_plots))

#print d_acceptances['pf_s1']['x_values']
#print d_acceptances['pf_s1']['y_values_lower']

pickle.dump(d_acceptances, open('%sacceptances.p' % (s_path_to_pickle_save), 'w'))

#plt.show()
