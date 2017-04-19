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
import root_numpy
from rootpy.io import root_open

import corner
from sklearn import neighbors
from sklearn import grid_search
from sklearn import preprocessing


s_path_to_input = './resources/combined_ambe_mc.root'
s_path_to_pickle_save = './resources/ambe_mc_matt.p'

f_ambe_mc = root_open(s_path_to_input, 'read')

l_arrays = root_numpy.tree2array(f_ambe_mc.tSort, branches=['X', 'Y', 'Z', 'Ed', 'NR', 'ns'])

d_single_scatters = {}
d_single_scatters['X'] = []
d_single_scatters['Y'] = []
d_single_scatters['Z'] = []
d_single_scatters['Ed'] = []

#NR=1.0, ns=1

for i in tqdm.tqdm(xrange(len(l_arrays['Ed']))):
    if (l_arrays['ns'][i] == 1) and (l_arrays['NR'][i] == 1.):
        d_single_scatters['X'].append(l_arrays['X'][i][0])
        d_single_scatters['Y'].append(l_arrays['Y'][i][0])
        d_single_scatters['Z'].append(l_arrays['Z'][i][0])
        d_single_scatters['Ed'].append(l_arrays['Ed'][i][0])


#print l_arrays['X']

d_ambe_mc = {}
d_ambe_mc['X'] = np.asarray(d_single_scatters['X'], dtype=np.float32)
d_ambe_mc['Y'] = np.asarray(d_single_scatters['Y'], dtype=np.float32)
d_ambe_mc['Z'] = np.asarray(d_single_scatters['Z'], dtype=np.float32)
d_ambe_mc['Ed'] = np.asarray(d_single_scatters['Ed'], dtype=np.float32)


pickle.dump(d_ambe_mc, open(s_path_to_pickle_save, 'w'))
