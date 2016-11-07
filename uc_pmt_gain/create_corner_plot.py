#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import ROOT as root
from rootpy.plotting import Hist, Hist2D, Canvas, Legend
import numpy as np
import corner, time
import cPickle as pickle

if len(sys.argv) != 3:
    print 'Use is python create_corner_plot.py <filename> <num walkers>'
    sys.exit()

print '\n\nBy default look for all energies - change source if anything else is needed.\n'



filename = sys.argv[1]
num_walkers = int(sys.argv[2])

l_plots = ['plots', filename]

dir_specifier_name = filename

sPathToFile = './results/%s/sampler_dictionary.p' % (dir_specifier_name)

if os.path.exists(sPathToFile):
    dSampler = pickle.load(open(sPathToFile, 'r'))
    l_chains = []
    for sampler in dSampler[num_walkers]:
        l_chains.append(sampler['_chain'])

    aSampler = np.concatenate(l_chains, axis=1)

    print 'Successfully loaded sampler!'
else:
    print sPathToFile
    print 'Could not find file!'
    sys.exit()



lLabelsForCorner = ['p_hit_first_dynode', 'electrons_per_dynode', 'p_e_freed', 'bkg_mean', 'bkg_std', 'mean_num_pe_mpe1', 'mean_num_pe_mpe2']
num_dim = len(lLabelsForCorner)


num_steps = 1500

samples = aSampler[:, -num_steps:, :].reshape((-1, num_dim))
start_time = time.time()
print 'Starting corner plot...\n'
fig = corner.corner(samples, labels=lLabelsForCorner, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3e', title_kwargs={"fontsize": 12})
print 'Corner plot took %.3f minutes.\n\n' % ((time.time()-start_time)/60.)

# path for save
sPathForSave = './'
for directory in l_plots:
    sPathForSave += directory + '/'

if not os.path.exists(sPathForSave):
    os.makedirs(sPathForSave)

fig.savefig('%s%s_corner_plot.png' % (sPathForSave, dir_specifier_name))



#raw_input('Enter to continue...')
