#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import ROOT as root
from rootpy.plotting import Hist, Hist2D, Canvas, Legend
import neriX_analysis
import numpy as np
import corner
import cPickle as pickle
import time

if len(sys.argv) != 3:
    print 'Use is python create_corner_plot.py <filename> <num walkers>'
    sys.exit()

print '\n\nBy default look for all energies - change source if anything else is needed.\n'



filename = sys.argv[1]
num_walkers = int(sys.argv[2])

l_plots = ['plots', filename]

dir_specifier_name = filename

sPathToFile = './results/%s/acceptance_fraction.p' % (dir_specifier_name)

if os.path.exists(sPathToFile):
	a_acceptance_fraction = pickle.load(open(sPathToFile, 'r'))
	print 'Successfully loaded acceptance fraction array!'
	#a_acor = pickle.load(open(sPathToFile_autocorrelation, 'r'))
else:
	print sPathToFile
	print 'Could not find file!'
	sys.exit()

#print a_acor

c_acceptance = Canvas()
h_acceptance = Hist(100, 0, 1, name='h_acceptance', title='Acceptance Fraction of Most Recent Sampler')
h_acceptance.fill_array(a_acceptance_fraction)
h_acceptance.Scale(1./h_acceptance.Integral())
#h_acceptance.SetStats(0)
h_acceptance.Draw()
c_acceptance.Update()

raw_input('Press enter to continue...')

plot_name = '%s_acceptance_fraction' % (dir_specifier_name)
plot_name = 'yields_fit_%s' % (plot_name)

neriX_analysis.save_plot(l_plots, c_acceptance, plot_name)





