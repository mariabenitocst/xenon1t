import ROOT as root
from rootpy.plotting import Canvas, Hist
from rootpy.io import File, root_open
import neriX_analysis
import neriX_datasets, neriX_config
import sys, os
import numpy as np
from scipy import stats

import dill, pickle, root_numpy

if(len(sys.argv) != 2):
	print 'Usage is python convert_nerix_data_to_pickle.py <filename>'
	sys.exit(1)

filename = sys.argv[1]

s_data_path = '%srun_16/%s.root' % (neriX_config.pathToData, filename)
root_file = File(s_data_path)

channel = 16
parameter_to_draw = 'SingleIntegral[%d]' % (channel)

print

a_integral = root_numpy.tree2array(root_file.T0, branches=parameter_to_draw)
a_integral = np.asarray(a_integral)

num_to_choose = int(len(a_integral)*0.2)

data_dir = './data/'

pickle.dump(np.random.choice(a_integral, num_to_choose), open('%s%s.p' % (data_dir, filename), 'w'))

