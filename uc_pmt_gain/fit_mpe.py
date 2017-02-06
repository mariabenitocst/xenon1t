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

def mpe_fitting(filename, run, num_photons, use_ideal=True):

    run_number = int(run)

    if filename[-5:] == '.root':
        filename = filename[:-5]

    s_data_path = './data/%s.p' % filename

    if filename[:5] == 'nerix':
        file_identifier = filename
    else:
        file_identifier = filename[-9:]

    a_integral = pickle.load((open(s_data_path, 'r')))

    # max_num_events used to limit amplitudes
    max_num_events = len(a_integral)

    s_path_to_save = './results/%s/' % (file_identifier)
    l_colors = [4, 2, 8, 7, 5, 9] + [4, 2, 8, 7, 5, 9]

    d_mpe_fit = {}

    if file_identifier == '0062_0061':
        d_mpe_fit['settings'] = [250, -1e6, 2e7]

        d_mpe_fit['bkg_mean_low'] = -1e6
        d_mpe_fit['bkg_mean_high'] = 2e6
        d_mpe_fit['bkg_width_low'] = 1e4
        d_mpe_fit['bkg_width_high'] = 1e6
        d_mpe_fit['spe_mean_low'] = 3.5e6
        d_mpe_fit['spe_mean_high'] = 1e7
        d_mpe_fit['spe_width_low'] = 8e5
        d_mpe_fit['spe_width_high'] = 3e6
        d_mpe_fit['ua_mean_low'] = 1e5
        d_mpe_fit['ua_mean_high'] = 5e6
        d_mpe_fit['ua_width_low'] = 1e4
        d_mpe_fit['ua_width_high'] = 1e6

        d_mpe_fit['spe_mean_guess'] = 6e6
        d_mpe_fit['spe_width_guess'] = 1.9e6
    
    elif file_identifier == '0066_0065':
        d_mpe_fit['settings'] = [250, -1e6, 1.2e7]

        d_mpe_fit['bkg_mean_low'] = -1e6
        d_mpe_fit['bkg_mean_high'] = 2e6
        d_mpe_fit['bkg_width_low'] = 1e4
        d_mpe_fit['bkg_width_high'] = 1e6
        d_mpe_fit['spe_mean_low'] = 2e6
        d_mpe_fit['spe_mean_high'] = 6e6
        d_mpe_fit['spe_width_low'] = 8e5
        d_mpe_fit['spe_width_high'] = 3e6
        d_mpe_fit['ua_mean_low'] = 1e5
        d_mpe_fit['ua_mean_high'] = 5e6
        d_mpe_fit['ua_width_low'] = 1e4
        d_mpe_fit['ua_width_high'] = 1e6

        d_mpe_fit['spe_mean_guess'] = 3.6e6
        d_mpe_fit['spe_width_guess'] = 1.1e6

    elif file_identifier == '0067_0068':
        d_mpe_fit['settings'] = [250, -1e6, 7.5e6]

        d_mpe_fit['bkg_mean_low'] = -5e5
        d_mpe_fit['bkg_mean_high'] = 5e5
        d_mpe_fit['bkg_width_low'] = 1e4
        d_mpe_fit['bkg_width_high'] = 1e6
        d_mpe_fit['spe_mean_low'] = 1.2e6
        d_mpe_fit['spe_mean_high'] = 3.5e6
        d_mpe_fit['spe_width_low'] = 3e5
        d_mpe_fit['spe_width_high'] = 1e6
        d_mpe_fit['ua_mean_low'] = 1e5
        d_mpe_fit['ua_mean_high'] = 1.5e6
        d_mpe_fit['ua_width_low'] = 1e4
        d_mpe_fit['ua_width_high'] = 1e6

        d_mpe_fit['spe_mean_guess'] = 2.1e6
        d_mpe_fit['spe_width_guess'] = 6.6e5

    elif file_identifier == '0071_0072':
        d_mpe_fit['settings'] = [250, -1e6, 3.4e7]

        d_mpe_fit['bkg_mean_low'] = -1e6
        d_mpe_fit['bkg_mean_high'] = 2e6
        d_mpe_fit['bkg_width_low'] = 1e4
        d_mpe_fit['bkg_width_high'] = 1e6
        d_mpe_fit['spe_mean_low'] = 7.5e6
        d_mpe_fit['spe_mean_high'] = 1.4e7
        d_mpe_fit['spe_width_low'] = 1e6
        d_mpe_fit['spe_width_high'] = 3.5e6
        d_mpe_fit['ua_mean_low'] = 1e5
        d_mpe_fit['ua_mean_high'] = 5e6
        d_mpe_fit['ua_width_low'] = 1e4
        d_mpe_fit['ua_width_high'] = 1e6

        d_mpe_fit['spe_mean_guess'] = 9.5e6
        d_mpe_fit['spe_width_guess'] = 2.9e6

    elif file_identifier == '0073_0074':
        d_mpe_fit['settings'] = [50, -1e6, 4.2e7]

        d_mpe_fit['bkg_mean_low'] = -1e6
        d_mpe_fit['bkg_mean_high'] = 2e6
        d_mpe_fit['bkg_width_low'] = 1e5
        d_mpe_fit['bkg_width_high'] = 2e6
        d_mpe_fit['spe_mean_low'] = 8.5e6
        d_mpe_fit['spe_mean_high'] = 1.4e7
        d_mpe_fit['spe_width_low'] = 1.5e6
        d_mpe_fit['spe_width_high'] = 4.5e6
        d_mpe_fit['ua_mean_low'] = 5e5
        d_mpe_fit['ua_mean_high'] = 4e6
        d_mpe_fit['ua_width_low'] = 0.5e6
        d_mpe_fit['ua_width_high'] = 1e6

        d_mpe_fit['spe_mean_guess'] = 9.5e6
        d_mpe_fit['spe_width_guess'] = 2.9e6

    elif file_identifier == 'nerix_160418_1523':
        d_mpe_fit['settings'] = [50, -5e5, 3.e6]
        
        d_mpe_fit['bkg_mean_low'] = -5e5
        d_mpe_fit['bkg_mean_high'] = 5e5
        d_mpe_fit['bkg_width_low'] = 1e4
        d_mpe_fit['bkg_width_high'] = 5e5
        d_mpe_fit['spe_mean_low'] = 6e5
        d_mpe_fit['spe_mean_high'] = 11e5
        d_mpe_fit['spe_width_low'] = 3e5
        d_mpe_fit['spe_width_high'] = 9e5
        d_mpe_fit['ua_mean_low'] = 1e3
        d_mpe_fit['ua_mean_high'] = 5e5
        d_mpe_fit['ua_width_low'] = 1e4
        d_mpe_fit['ua_width_high'] = 5e5

    elif file_identifier == 'nerix_160418_1531':
        d_mpe_fit['settings'] = [50, -5e5, 4.e6]
        
        d_mpe_fit['bkg_mean_low'] = -5e5
        d_mpe_fit['bkg_mean_high'] = 5e5
        d_mpe_fit['bkg_width_low'] = 1e4
        d_mpe_fit['bkg_width_high'] = 5e5
        d_mpe_fit['spe_mean_low'] = 6e5
        d_mpe_fit['spe_mean_high'] = 11e5
        d_mpe_fit['spe_width_low'] = 3e5
        d_mpe_fit['spe_width_high'] = 9e5
        d_mpe_fit['ua_mean_low'] = 1e3
        d_mpe_fit['ua_mean_high'] = 5e5
        d_mpe_fit['ua_width_low'] = 1e4
        d_mpe_fit['ua_width_high'] = 5e5

    else:
        print '\n\nSettings do not exist for given setup: %s\n\n' % (file_identifier)
        sys.exit()

    l_plots = ['plots', file_identifier]


    par_names = ['p0_ampl', 'mean_bkg', 'width_bkg', 'mean_spe', 'width_spe'] + ['p%d_ampl' % (i + 1) for i in xrange(num_photons)]




    a_integral = pickle.load((open(s_data_path, 'r')))

    if use_ideal:
        l_mpe_fit_func = ['[5]/(2*3.14*%d*[4]**2.)**0.5*TMath::Poisson(%d, [0])*exp(-0.5/%.1f*((x - %.1f*[3])/[4])**2)' % (iElectron, iElectron, iElectron, iElectron) for iElectron in xrange(1, num_photons + 1)]
    else:
        l_mpe_fit_func = ['[5]/(2*3.14*([2]**2. + %d*[4]**2.))**0.5*TMath::Poisson(%d, [0])*exp(-0.5*((x - %.1f*[3] - [1])/(%.1f*[4]**2 + [2]**2)**0.5)**2)' % (iElectron, iElectron, iElectron, iElectron) for iElectron in xrange(1, num_photons + 1)]

    h_mpe_spec = Hist(*d_mpe_fit['settings'], name='h_mpe_spec', title='MPE Spectrum with Gaussian Fit - %s' % filename)
    h_mpe_spec.SetMarkerSize(0)
    h_mpe_spec.fill_array(a_integral)

    c1 = Canvas()

    h_mpe_spec.Draw()

    s_bkg = '[5]/(2*3.14*[2]**2.)**0.5*TMath::Poisson(0, [0])*exp(-0.5*((x - [1])/[2])**2)'
    s_under_amplified = '[6]/(2*3.14*[8]**2.)**0.5*exp(-0.5*((x - [7])/[8])**2)'
    s_fit_mpe = '(%s) + (%s) + (%s)' % (s_bkg, s_under_amplified, ' + '.join(l_mpe_fit_func))
    s_fit_mpe = '(%s)*([1] < [3] ? 1. : 0.)*([1] < [7] ? 1. : 0.)*([7] < [3] ? 1. : 0.)' % (s_fit_mpe)
    fit_mpe = root.TF1('fit_mpe', s_fit_mpe, *d_mpe_fit['settings'][1:])
    fit_mpe.SetLineColor(46)
    fit_mpe.SetLineStyle(2)
    fit_mpe.SetLineWidth(3)

    h_mpe_spec.GetXaxis().SetTitle('Integrated Charge [e-]')
    h_mpe_spec.GetYaxis().SetTitle('Counts')
    h_mpe_spec.GetYaxis().SetTitleOffset(1.4)
    h_mpe_spec.SetStats(0)

    c1.SetLogy()


    fit_mpe.SetParLimits(0, 0.9, 2.5)
    fit_mpe.SetParameter(0, 1.3)
    fit_mpe.SetParLimits(1, d_mpe_fit['bkg_mean_low'], d_mpe_fit['bkg_mean_high'])
    fit_mpe.SetParameter(1, (d_mpe_fit['bkg_mean_low']+d_mpe_fit['bkg_mean_high'])/2.)
    fit_mpe.SetParLimits(2, d_mpe_fit['bkg_width_low'], d_mpe_fit['bkg_width_high'])
    fit_mpe.SetParameter(2, (d_mpe_fit['bkg_width_low']+d_mpe_fit['bkg_width_high'])/2.)
    fit_mpe.SetParLimits(3, d_mpe_fit['spe_mean_low'], d_mpe_fit['spe_mean_high'])
    fit_mpe.SetParameter(3, d_mpe_fit['spe_mean_guess'])
    fit_mpe.SetParLimits(4, d_mpe_fit['spe_width_low'], d_mpe_fit['spe_width_high'])
    fit_mpe.SetParameter(4, d_mpe_fit['spe_width_guess'])
    fit_mpe.SetParLimits(5, 10, max_num_events*1e6)
    fit_mpe.SetParameter(5, (10+max_num_events*1e6)/2.)
    fit_mpe.SetParLimits(6, 10, max_num_events*1e6)
    fit_mpe.SetParameter(6, (10+max_num_events*1e6)/2.)
    fit_mpe.SetParLimits(7, d_mpe_fit['ua_mean_low'], d_mpe_fit['ua_mean_high'])
    fit_mpe.SetParameter(7, (d_mpe_fit['ua_mean_low']+d_mpe_fit['ua_mean_high'])/2.)
    fit_mpe.SetParLimits(8, d_mpe_fit['ua_width_low'], d_mpe_fit['ua_width_high'])
    fit_mpe.SetParameter(8, (d_mpe_fit['ua_width_low']+d_mpe_fit['ua_width_high'])/2.)


    """
    for i, guess in enumerate(mpe_par_guesses):
        fit_mpe.SetParameter(i, guess)

    for i in xrange(len(par_names)):
        fit_mpe.SetParName(i, par_names[i])


    for photon in xrange(num_photons):
        fit_mpe.SetParLimits(5 + photon, 0, max_num_events)
    """

    fitResult = h_mpe_spec.Fit('fit_mpe', 'MILES')



    # draw individual peaks
    s_gaussian = '[0]*exp(-0.5/%.1f*((x - %.1f*[1])/[2])**2)'
    l_functions = []
    l_individual_integrals = [0. for i in xrange(num_photons+2)]
    for i in xrange(num_photons + 2):
        l_functions.append(root.TF1('peak_%d' % i, '[0]*exp(-0.5*((x - [1])/[2])**2)', *d_mpe_fit['settings'][1:]))

        # set parameters
        if i == 0:
            ampl = fit_mpe.GetParameter(5)*root.TMath.Poisson(0, fit_mpe.GetParameter(0))
            mean = fit_mpe.GetParameter(1)
            width = fit_mpe.GetParameter(2)
            if width > 0:
                ampl /= (2*3.14*width**2.)**0.5
            l_functions[i].SetParameters(ampl, mean, width)
            
        # under amplified peak
        elif i == (num_photons + 1):
            ampl = fit_mpe.GetParameter(6)
            mean = fit_mpe.GetParameter(7)
            width = fit_mpe.GetParameter(8)
            if width > 0:
                ampl /= (2*3.14*width**2.)**0.5
            l_functions[i].SetParameters(ampl, mean, width)

        else:
            ampl = fit_mpe.GetParameter(5)*root.TMath.Poisson(i, fit_mpe.GetParameter(0))
            if use_ideal:
                mean = fit_mpe.GetParameter(3) * i
                width = fit_mpe.GetParameter(4)*i**0.5
            else:
                mean = fit_mpe.GetParameter(3)*i + fit_mpe.GetParameter(1)
                width = (fit_mpe.GetParameter(4)**2*i + fit_mpe.GetParameter(2)**2)**0.5

            if width > 0:
                ampl /= (2*3.14*width**2.)**0.5

            l_functions[i].SetParameters(ampl, mean, width)



        l_individual_integrals[i] = ampl*width*(2*3.1415)**0.5
        l_functions[i].SetLineColor(l_colors[i])
        l_functions[i].Draw('same')




    c1.Update()


    fitStatus = fitResult.CovMatrixStatus()
    if fitStatus != 3:
        neriX_analysis.failure_message('Fit failed, please adjust guesses and try again.')
        fit_successful = False
    else:
        neriX_analysis.success_message('Fit successful, please copy output to appropriate files.')
        fit_successful = True

    #if not os.path.exists(sPathToSaveOutput):
    #    os.makedirs(sPathToSaveOutput)

    fitter = root.TVirtualFitter.Fitter(fit_mpe)
    #fitter = root.TVirtualFitter.GetFitter()
    amin = np.asarray([0], dtype=np.float64)
    dum1 = np.asarray([0], dtype=np.float64)
    dum2 = np.asarray([0], dtype=np.float64)
    dum3 = np.asarray([0], dtype=np.int32)
    dum4 = np.asarray([0], dtype=np.int32)

    fitter.GetStats(amin, dum1, dum2, dum3, dum4)


    print '\n\namin for %d photons: %f' % (num_photons, amin)
    print 'fAmin for %d photons: %f\n\n' % (num_photons, root.gMinuit.fAmin)
    print fit_mpe.GetChisquare()


    # draw tpavetext
    tpt_mpe = root.TPaveText(.55,.75,.85,.85,'blNDC')
    tpt_mpe.AddText('#mu_{SPE} = %.2e #pm %.2e' % (fit_mpe.GetParameter(3), fit_mpe.GetParError(3)))
    tpt_mpe.AddText('#sigma_{SPE} = %.2e #pm %.2e' % (fit_mpe.GetParameter(4), fit_mpe.GetParError(4)))
    tpt_mpe.Draw('same')

    tpt_mpe.SetTextColor(root.kBlack)
    tpt_mpe.SetFillStyle(0)
    tpt_mpe.SetBorderSize(0)

    c1.Update()


    neriX_analysis.save_plot(l_plots, c1, 'mpe_poisson_gaussian_fit_%s' % (file_identifier))


    return (0,0,0)








#file = 'nerix_160418_1523' # 3 photons (27.86)
#num_photons = 3

#file = 'nerix_160418_1531' # 5 photons (43.44)
#num_photons = 5

#file = 'darkbox_spectra_0062_0061' # 3 photons (863.48)
#num_photons = 3

#file = 'darkbox_spectra_0066_0065' # 3 photons (426.52)
#num_photons = 3

#file = 'darkbox_spectra_0067_0068' # 4 photons (175.12, 354)
#num_photons = 4

#file = 'darkbox_spectra_0071_0072' # 4 photons (750.37)
#num_photons = 4

file = 'darkbox_spectra_0073_0074' # 5 photons (349)
num_photons = 5



# ideal vs ideal makes almost no difference (since bkg width is tiny)
mpe_fitting(file, 16, num_photons, use_ideal=False)



"""
# see above for ln(L)
#lFiles = ['nerix_160418_1523'] # 4 photons (24.69)
#lFiles = ['darkbox_spectra_0062_0061'] # 3 photons (863.48)
#lFiles = ['darkbox_spectra_0066_0065'] # 3 photons (274.45)
#lFiles = ['darkbox_spectra_0067_0068'] # 4 photons (76.08)
#lFiles = ['darkbox_spectra_0071_0072'] # 4 photons (750.37)
lFiles = ['darkbox_spectra_0073_0074'] # 5 photons (344)


lNumPhotons = [2, 3, 4, 5]





sForFile = ''
sForTextFile = ''

for file in lFiles:
    lLnlikelihoods = [0 for i in xrange(len(lNumPhotons))]
    lStrings = [0 for i in xrange(len(lNumPhotons))]
    for i, num_photons in enumerate(lNumPhotons):
        print '\n\n\nStarting %s with %d photons.\n\n\n' % (file, lNumPhotons[i])
        lStrings[i], lLnlikelihoods[i], forTextFile = mpe_fitting(file, 15, num_photons, use_ideal=False)

    # find first step with p-value > 0.05 and take previous

    # ex:
    # 2->3: p < 0.05
    # 3->4: p > 0.05 - STOP!
    # use 3 electrons

    for i in xrange(len(lNumPhotons) - 1):
        testStat = -2 * (lLnlikelihoods[i] - lLnlikelihoods[i + 1])
        pValue = 1 - stats.chi2.cdf(testStat, 1)
        indexToUse = i
        if pValue > 0.05:
            break

    # run one more time to save image of "correct"
    # number of electrons
    dummy1, dummy2, forTextFile = mpe_fitting(file, 15, num_photons, use_ideal=False)

    sForTextFile += forTextFile

    sSingleFile = lStrings[indexToUse]
    sForFile += '# %s\n%s\n\n\n' % (file, sSingleFile)

print sForFile

print '\n\n\nCopy and paste into text file\n\n\n'

print sForTextFile
"""