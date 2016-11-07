#!/usr/bin/python
import pickle
#print pickle.Pickler.dispatch
import dill
#print pickle.Pickler.dispatch

import ROOT as root
import sys, os

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

import pandas
import root_pandas

from rootpy.plotting import root2matplotlib as rplt

import emcee, corner, click
import neriX_analysis, neriX_datasets
from rootpy.plotting import Hist2D, Hist, Legend, Canvas
from rootpy.tree import Tree
import numpy as np
import tqdm, time, copy_reg, types, pickle
from root_numpy import tree2array, array2tree
import scipy.optimize as op
import scipy.stats
import scipy.special
import scipy.misc

import gc
gc.disable()

def neg_ln_likelihood_2d_gaussian(a_parameters, x):
    mean_x, mean_y, var_x, var_y, cov_xy = a_parameters
    ln_likelihood = -np.sum(scipy.stats.multivariate_normal.logpdf(x.T, mean=[mean_x, mean_y], cov=[[var_x, cov_xy], [cov_xy, var_y]]))
    if np.isfinite(ln_likelihood):
        return ln_likelihood
    else:
        return -np.inf

def neg_ln_likelihood_1d_gaussian(a_parameters, x):
    mean_x, stdev_x = a_parameters
    #print a_parameters
    ln_likelihood = -np.sum(scipy.stats.norm.logpdf(x, loc=mean_x, scale=stdev_x))
    if np.isfinite(ln_likelihood):
        return ln_likelihood
    else:
        return -np.inf

def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))














