import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import neriX_analysis
import neriX_datasets
import sys, os
import numpy as np
from scipy import stats

import cPickle as pickle


s_identifier_uc = '0066_0065'
s_identifier_nerix = 'nerix_160418_1523'

s_path_to_files = './plot_dicts/'
s_path_to_save = './paper_plots/'


d_unc_band_uc = pickle.load(open('%s%s_model_fit_unc_band.p' % (s_path_to_files, s_identifier_uc), 'r'))
d_best_fit_uc = pickle.load(open('%s%s_model_fit_best_fit.p' % (s_path_to_files, s_identifier_uc), 'r'))
d_spe_response_uc = pickle.load(open('%s%s_model_fit_spe_response.p' % (s_path_to_files, s_identifier_uc), 'r'))

a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = d_unc_band_uc['data_x'], d_unc_band_uc['data_y'], d_unc_band_uc['data_x_err_low'], d_unc_band_uc['data_x_err_high'], d_unc_band_uc['data_y_err_low'], d_unc_band_uc['data_y_err_high']


fig_best_uc, ax_best_uc = plt.subplots(1)
ax_best_uc.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='black', fmt='.')
ax_best_uc.fill_between(d_unc_band_uc['mc_x'], d_unc_band_uc['mc_one_sigma_below'], d_unc_band_uc['mc_one_sigma_above'], facecolor='red', alpha=0.2, interpolate=True)
ax_best_uc.plot(d_best_fit_uc['mc_x'], d_best_fit_uc['mc_y'], 'r--')

ax_best_uc.set_yscale('log')
ax_best_uc.set_xlim(-1e6, 1.2e7)
ax_best_uc.set_ylim(50, 3600)

#ax_best_uc.set_title('R11410 - 1500 V')
ax_best_uc.set_xlabel('Integrated Charge [$e^-$]')
ax_best_uc.set_ylabel('Counts')

s_mean_gain_uc = r'$\mu_{SPE} = %.2e \pm %.2e \ [e^-]$' % (3.170e+06, 2.450e+04)
s_stdev_gain_uc = r'$\sigma_{SPE} = %.2e \pm %.2e \ [e^-]$' % (1.560e+06, 1.330e+04)
s_occupancy_uc = r'$\lambda = 1.256^{+0.010}_{-0.010}$'

ax_best_uc.text(0.75, 0.9, '%s\n%s\n%s' % (s_mean_gain_uc, s_stdev_gain_uc, s_occupancy_uc), ha='center', va='center', transform=ax_best_uc.transAxes)



fig_spe_uc, ax_spe_uc = plt.subplots(1)
ax_spe_uc.fill_between(d_spe_response_uc['mc_x'], d_spe_response_uc['mc_one_sigma_below'], d_spe_response_uc['mc_one_sigma_above'], facecolor='blue', alpha=0.5, interpolate=True)



ax_spe_uc.set_xlabel(r'Integrated Charge [$e^{-}$]')
ax_spe_uc.set_ylabel('Normalized Counts')

ax_spe_uc.set_xlim(-1e6, 0.9e7)
ax_spe_uc.set_ylim(0, 0.010)



d_unc_band_nerix = pickle.load(open('%s%s_unc_band.p' % (s_path_to_files, s_identifier_nerix), 'r'))
d_best_fit_nerix = pickle.load(open('%s%s_best_fit.p' % (s_path_to_files, s_identifier_nerix), 'r'))
d_spe_response_nerix = pickle.load(open('%s%s_spe_response.p' % (s_path_to_files, s_identifier_nerix), 'r'))


a_x_values, a_y_values, a_x_err_low, a_x_err_high, a_y_err_low, a_y_err_high = d_unc_band_nerix['data_x'], d_unc_band_nerix['data_y'], d_unc_band_nerix['data_x_err_low'], d_unc_band_nerix['data_x_err_high'], d_unc_band_nerix['data_y_err_low'], d_unc_band_nerix['data_y_err_high']


fig_best_nerix, ax_best_nerix = plt.subplots(1)
ax_best_nerix.errorbar(a_x_values, a_y_values, xerr=[a_x_err_low, a_x_err_high], yerr=[a_y_err_low, a_y_err_high], color='black', fmt='.')
ax_best_nerix.fill_between(d_unc_band_nerix['mc_x'], d_unc_band_nerix['mc_one_sigma_below'], d_unc_band_nerix['mc_one_sigma_above'], facecolor='red', alpha=0.2, interpolate=True)
ax_best_nerix.plot(d_best_fit_nerix['mc_x'], d_best_fit_nerix['mc_y'], 'r--')

ax_best_nerix.set_yscale('log')
#ax_best_nerix.set_xlim(-1e6, 1.2e7)
ax_best_nerix.set_ylim(85, 1000)

#ax_best_nerix.set_title('R11410 - 1500 V')
ax_best_nerix.set_xlabel('Integrated Charge [$e^-$]')
ax_best_nerix.set_ylabel('Counts')

s_mean_gain_nerix = r'$\mu_{SPE} = %.2e \pm %.2e \ [e^-]$' % (7.50e5, 5.23e4)
s_stdev_gain_nerix = r'$\sigma_{SPE} = %.2e \pm %.2e \ [e^-]$' % (5.85e5, 1.18e4)
s_occupancy_nerix = r'$\lambda = 1.528^{+0.239}_{-0.132}$'

ax_best_nerix.text(0.75, 0.9, '%s\n%s\n%s' % (s_mean_gain_nerix, s_stdev_gain_nerix, s_occupancy_nerix), ha='center', va='center', transform=ax_best_nerix.transAxes)


fig_spe_nerix, ax_spe_nerix = plt.subplots(1)
ax_spe_nerix.fill_between(d_spe_response_nerix['mc_x'], d_spe_response_nerix['mc_one_sigma_below'], d_spe_response_nerix['mc_one_sigma_above'], facecolor='blue', alpha=0.5, interpolate=True)



ax_spe_nerix.set_xlabel(r'Integrated Charge [$e^{-}$]')
ax_spe_nerix.set_ylabel('Normalized Counts')

ax_spe_nerix.set_xlim(1e0, 2.1e6)
ax_spe_nerix.set_ylim(0.0001, 0.030)
#ax_spe_nerix.set_xscale('log')
#ax_spe_nerix.set_yscale('log')


fig_best_uc.savefig('%suc_best_fit.png' % (s_path_to_save))
fig_best_nerix.savefig('%snerix_best_fit.png' % (s_path_to_save))
fig_spe_uc.savefig('%suc_spe_response.png' % (s_path_to_save))


plt.show()

