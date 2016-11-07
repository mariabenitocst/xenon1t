#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import ROOT as root
from rootpy.plotting import Hist, Hist2D, Canvas, Legend
import numpy as np
import corner, time
import cPickle as pickle
import fit_nr_example

if len(sys.argv) != 4:
    print 'Use is python make_corner_plot.py <name> <num walkers> <num_steps_to_include_in_plots>'
    sys.exit()

print '\n\nBy default look for all energies - change source if anything else is needed.\n'



name = sys.argv[1]
num_walkers = int(sys.argv[2])
num_steps_to_include = int(sys.argv[3])


fitter = fit_nr_example.fit_nr(name, num_mc_events=int(1e4))

fitter.create_corner_plot(num_walkers=num_walkers, num_steps_to_include=num_steps_to_include)
fitter.close_workers()


