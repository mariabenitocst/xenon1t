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

import scipy.interpolate

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
d_acceptances['pf_s1']['y_values_mean_old'] = np.zeros(num_pts)
d_acceptances['pf_s1']['y_values_upper'] = np.zeros(num_pts)

# new acceptances 3/2/17
x = [ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15, 16, 17, 18, 19.]
y = [ 0., 0., 0.002, 0.2765, 0.55, 0.739, 0.819, 0.8605, 0.891, 0.9315, 0.938, 0.95, 0.958, 0.9705, 0.975, 0.977, 0.983, 0.983, 0.9775, 0.984 ]
new_acceptances = scipy.interpolate.interp1d(x, y, kind='cubic')

for i in xrange(num_pts):
    d_acceptances['pf_s1']['x_values'][i] = f_acceptance.gpeakfindingacc_lower.GetX()[i]
    
    
    if d_acceptances['pf_s1']['x_values'][i] < 20:
        d_acceptances['pf_s1']['y_values_mean_old'][i] = f_acceptance.gpeakfindingacc_mean.GetY()[i]
        
        d_acceptances['pf_s1']['y_values_mean'][i] = new_acceptances(d_acceptances['pf_s1']['x_values'][i])
        d_acceptances['pf_s1']['y_values_lower'][i] = f_acceptance.gpeakfindingacc_lower.GetY()[i] - d_acceptances['pf_s1']['y_values_mean_old'][i] + d_acceptances['pf_s1']['y_values_mean'][i]
        d_acceptances['pf_s1']['y_values_upper'][i] = f_acceptance.gpeakfindingacc_upper.GetY()[i] - d_acceptances['pf_s1']['y_values_mean_old'][i] + d_acceptances['pf_s1']['y_values_mean'][i]

    else:
        d_acceptances['pf_s1']['y_values_lower'][i] = 1.
        d_acceptances['pf_s1']['y_values_upper'][i] = 1.
        d_acceptances['pf_s1']['y_values_mean'][i] = (d_acceptances['pf_s1']['y_values_upper'][i] + d_acceptances['pf_s1']['y_values_lower'][i]) /2.
        d_acceptances['pf_s1']['y_values_mean_old'][i] = 1.





fig_pf_s1, ax_pf_s1 = plt.subplots(1)
ax_pf_s1.fill_between(d_acceptances['pf_s1']['x_values'], d_acceptances['pf_s1']['y_values_lower'], d_acceptances['pf_s1']['y_values_upper'], alpha=0.2)
ax_pf_s1.plot(d_acceptances['pf_s1']['x_values'], d_acceptances['pf_s1']['y_values_mean'], 'b--', label='New Acceptance - 160302')
ax_pf_s1.plot(d_acceptances['pf_s1']['x_values'], d_acceptances['pf_s1']['y_values_mean_old'], 'g-', label='Old Acceptance')

#ax_pf_s1.set_xlim(min(d_acceptances['pf_s1']['x_values']), max(d_acceptances['pf_s1']['x_values']))
ax_pf_s1.set_ylim(0, 1.03)

ax_pf_s1.set_xlim(0, 25)
ax_pf_s1.set_xlabel('$S1 [PE]$')
ax_pf_s1.set_ylabel('$Acceptance$')

ax_pf_s1.legend(loc='best', fontsize=10)

if not os.path.isdir(s_path_to_plots):
    os.mkdir(s_path_to_plots)

fig_pf_s1.savefig('%spf_acceptance_s1.png' % (s_path_to_plots))

#print d_acceptances['pf_s1']['x_values']
#print d_acceptances['pf_s1']['y_values_lower']

pickle.dump(d_acceptances, open('%sacceptances.p' % (s_path_to_pickle_save), 'w'))

#plt.show()
