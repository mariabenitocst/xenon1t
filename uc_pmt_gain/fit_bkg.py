import ROOT as root
from rootpy.plotting import Canvas, Hist
from rootpy.io import File, root_open
import neriX_analysis
import neriX_datasets
import sys, os
import numpy as np
from scipy import stats

import cPickle as pickle

#print '\n\n\n\nBatch mode on.\n\n\n\n\n'
#gROOT.SetBatch(True)

root.TVirtualFitter.SetMaxIterations(5000)

def bkg_fitting(filename):

    if filename[-5:] == '.root':
        filename = filename[:-5]

    s_root_path = './data/%s.root' % filename


    f_integral = root_open(s_root_path, 'read')

    conversion_to_num_electrons = 1./(250e6)/50./10./1.6e-19*2./2**12


    # find fit limits
    l_percentiles = [10, 90]
    lb_fit, ub_fit = -30, 30

    # grab histogram
    h_bkg = f_integral.bkgd_data_integral_hist

    c_fit = Canvas()
    h_bkg.Draw()


    fit_func = root.TF1('fit_func', 'gaus', lb_fit, ub_fit)
    h_bkg.Fit('fit_func', 'MELSR')

    fit_func.Draw('same')
    c_fit.Update()

    d_fit = {}
    d_fit['bkg_mean'] = fit_func.GetParameter(1)*conversion_to_num_electrons
    d_fit['bkg_std'] = fit_func.GetParameter(2)*conversion_to_num_electrons
    d_fit['bkg_mean_unc'] = fit_func.GetParError(1)*conversion_to_num_electrons
    d_fit['bkg_std_unc'] = fit_func.GetParError(2)*conversion_to_num_electrons

    print d_fit

    #raw_input('Press enter to continue...')


    pickle.dump(d_fit, open('./bkg_results/bkg_%s.p' % (file), 'w'))








file = 'darkbox_spectra_0062_0061' # 3 photons (863.48)


#file = 'darkbox_spectra_0066_0065' # 3 photons (426.52)


#file = 'darkbox_spectra_0067_0068' # 4 photons (175.12, 354)


#file = 'darkbox_spectra_0071_0072' # 4 photons (750.37)


#file = 'darkbox_spectra_0073_0074' # 5 photons (349)




# ideal vs ideal makes almost no difference (since bkg width is tiny)
bkg_fitting(file)


