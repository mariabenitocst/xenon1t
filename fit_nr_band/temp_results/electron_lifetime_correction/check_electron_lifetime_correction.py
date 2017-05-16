#!/usr/bin/python

import sys
import neriX_analysis
import os, re
import random

sys.path.insert(0, '../..')
import config_xe1t, nr_analysis_xe1t

import pandas as pd

import numpy as np

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

import tqdm

from rootpy import stl
from rootpy.io import File
from rootpy.tree import Tree, TreeModel, TreeChain
from rootpy.plotting import Hist, Hist2D, Canvas, Graph, func


from math import floor

from sklearn import neighbors
from sklearn import grid_search
from sklearn import preprocessing


import cPickle as pickle

a_s1_bin_edges = config_xe1t.a_s1_bin_edges
a_s1_bin_centers = (a_s1_bin_edges[1:]+a_s1_bin_edges[:-1])/2.
a_log_bin_edges = config_xe1t.a_log_bin_edges
a_log_bin_centers = (a_log_bin_edges[1:]+a_log_bin_edges[:-1])/2.

l_log_pts_without_correction = list(np.zeros(len(a_log_bin_edges)-1))
l_log_pts_with_correction = list(np.zeros(len(a_log_bin_edges)-1))

a_hist_with_correction = pickle.load(open('./hist_with_correction.p', 'r'))
a_hist_without_correction = pickle.load(open('./hist_without_correction.p', 'r'))

l_pts_to_pop = []

for i in xrange(len(a_s1_bin_centers)):
    if np.sum(a_hist_with_correction[:,i]) == 0 or np.sum(a_hist_without_correction[:,i]) == 0:
        l_pts_to_pop.append(i)
        continue

    l_log_pts_with_correction[i] = np.dot(a_hist_with_correction[:,i], a_log_bin_centers) / np.sum(a_hist_with_correction[:,i])
    l_log_pts_without_correction[i] = np.dot(a_hist_without_correction[:,i], a_log_bin_centers) / np.sum(a_hist_without_correction[:,i])


l_pts_to_pop = sorted(l_pts_to_pop, reverse=True)
for point in l_pts_to_pop:
    l_log_pts_with_correction.pop(point)
    l_log_pts_without_correction.pop(point)

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(8, 16))

ax1.pcolormesh(a_s1_bin_edges, a_log_bin_edges, a_hist_with_correction, cmap='Blues')
ax1.plot(a_s1_bin_centers, l_log_pts_with_correction, 'm--', label='Mean with Correction')
ax1.plot(a_s1_bin_centers, l_log_pts_without_correction, 'g--', label='Mean without Correction')
ax1.legend(loc='best')

ax1.set_title('MC with Correction')
ax1.set_xlabel('S1 [PE]')
ax1.set_ylabel(r'$log_{10}(\frac{S2}{S1})$')

ax2.pcolormesh(a_s1_bin_edges, a_log_bin_edges, a_hist_without_correction, cmap='Blues')
ax2.plot(a_s1_bin_centers, l_log_pts_with_correction, 'm--')
ax2.plot(a_s1_bin_centers, l_log_pts_without_correction, 'g--')

ax2.set_title('MC without Correction')
ax2.set_xlabel('S1 [PE]')
ax2.set_ylabel(r'$log_{10}(\frac{S2}{S1})$')

ax3.plot(a_s1_bin_centers, (np.asarray(l_log_pts_with_correction) - np.asarray(l_log_pts_without_correction)) / np.asarray(l_log_pts_without_correction), 'b--')
ax3.set_ylim(-0.03, 0.03)

ax3.set_title('(Corrected Mean - Uncorrected Mean) / Uncorrected Mean')
ax3.set_xlabel('S1 [PE]')
ax3.set_ylabel(r'$\Delta log_{10}(\frac{S2}{S1})$')

plt.show()
