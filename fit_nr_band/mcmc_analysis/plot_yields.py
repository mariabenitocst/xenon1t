#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import ROOT as root
import neriX_analysis
from rootpy.plotting import Hist, Hist2D, Canvas, Legend
import numpy as np
import cPickle as pickle
import tqdm

import config_xe1t


def get_ly_qy_from_nest_pars(energy, mean_field, w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb):

    dimensionless_energy = 11.5*energy*54**(-7./3.)
    g_value = 3*dimensionless_energy**0.15 + 0.7*dimensionless_energy**0.6 + dimensionless_energy
    lindhard_factor = kappa*g_value / (1. + kappa*g_value)
    penning_factor = 1. / (1. + eta*dimensionless_energy**lamb)
    sigma = gamma*mean_field**(-delta)
    
    exciton_to_ion_ratio = alpha*mean_field**(-zeta)*(1.-np.exp(-beta*dimensionless_energy))
    prob_exciton_success = 1. - 1./(1. + exciton_to_ion_ratio)

    num_quanta = energy*lindhard_factor/w_value
    num_excitons = num_quanta*prob_exciton_success
    num_ions = num_quanta - num_excitons

    prob_recombination = 1. - np.log(1.+num_ions*sigma)/(num_ions*sigma)
    num_recombined = num_ions*prob_recombination
    num_excitons += num_recombined
    num_ions -= num_recombined

    num_photons = num_excitons*penning_factor
    num_electrons = num_ions

    return num_photons/energy, num_electrons/energy



if len(sys.argv) != 2:
    print 'Use is python plot_yields.py <num walkers band>'
    sys.exit()

print '\n\nBy default look for all energies - change source if anything else is needed.\n'


d_yields = {}

num_walkers = int(sys.argv[1])

directory_descriptor = 'run_0_band'

l_cathode_settings_in_use = [12.]
l_degree_settings_in_use = [-4]

s_degree_settings = ''
for degree_setting in l_degree_settings_in_use:
    s_degree_settings += '%s,' % (degree_setting)
s_degree_settings = s_degree_settings[:-1]

s_cathode_settings = ''
for cathode_setting in l_cathode_settings_in_use:
    d_yields[cathode_setting] = {}
    s_cathode_settings += '%.3f,' % (cathode_setting)
s_cathode_settings = s_cathode_settings[:-1]

name_of_results_directory = config_xe1t.results_directory_name + '/%s' % (directory_descriptor)
l_plots = ['plots', directory_descriptor, '%s_kV_%s_deg' % (s_cathode_settings, s_degree_settings)]

s_path_to_file = './%s/%s_kV_%s_deg/sampler_dictionary.p' % (name_of_results_directory, s_cathode_settings, s_degree_settings)


if os.path.exists(s_path_to_file):
    dSampler = pickle.load(open(s_path_to_file, 'r'))
    l_chains = []
    for sampler in dSampler[num_walkers]:
        l_chains.append(sampler['_chain'])

    a_full_sampler = np.concatenate(l_chains, axis=1)

    print 'Successfully loaded sampler!'
else:
    print s_path_to_file
    print 'Could not find file!'
    sys.exit()


num_steps_nest = 5000
num_steps_band = 300
num_steps_to_pull_from = 1000


total_length_sampler = len(a_full_sampler)
num_dim = a_full_sampler.shape[2]
a_samples_band = a_full_sampler.reshape(-1, num_dim)

print a_samples_band.shape

# markov chain itself is random sample of posterior so use this
# to randomly sample parameter posterior (why do the work twice?)


num_energies = config_xe1t.l_energy_settings_yield_plots[0]
min_energy = config_xe1t.l_energy_settings_yield_plots[1]
max_energy = config_xe1t.l_energy_settings_yield_plots[2]
a_energies = np.linspace(min_energy, max_energy, num_energies)


a_temp_py = np.zeros(num_steps_nest)
a_temp_qy = np.zeros(num_steps_nest)

# get yields from NEST
for cathode_setting in l_cathode_settings_in_use:

    d_yields[cathode_setting]['a_py_median_nest'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_py_lb_nest'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_py_ub_nest'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_py_2lb_nest'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_py_2ub_nest'] = np.zeros(num_energies)
    
    d_yields[cathode_setting]['a_qy_median_nest'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_qy_lb_nest'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_qy_ub_nest'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_qy_2lb_nest'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_qy_2ub_nest'] = np.zeros(num_energies)

    for bin_number, energy in tqdm.tqdm(enumerate(a_energies)):
        for i in xrange(num_steps_nest):
            
            # get NEST values
            w_value_nest = np.random.normal(13.7, 0.4)/1000.
            
            current_rv = np.random.randn()
            if current_rv < 0:
                alpha_nest = 1.240 - current_rv*0.073
            else:
                alpha_nest = 1.240 + current_rv*0.079
            
            current_rv = np.random.randn()
            if current_rv < 0:
                zeta_nest = 0.0472 - current_rv*0.0073
            else:
                zeta_nest = 0.0472 + current_rv*0.0088

            current_rv = np.random.randn()
            if current_rv < 0:
                beta_nest = 239 - current_rv*8.8
            else:
                beta_nest = 239 + current_rv*28

            current_rv = np.random.randn()
            if current_rv < 0:
                gamma_nest = 0.01385 - current_rv*0.00073
            else:
                gamma_nest = 0.01385 + current_rv*0.00058

            current_rv = np.random.randn()
            if current_rv < 0:
                delta_nest = 0.0620 - current_rv*0.0064
            else:
                delta_nest = 0.0620 + current_rv*0.0056

            current_rv = np.random.randn()
            if current_rv < 0:
                kappa_nest = 0.1394 - current_rv*0.0026
            else:
                kappa_nest = 0.1394 + current_rv*0.0032

            current_rv = np.random.randn()
            if current_rv < 0:
                eta_nest = 3.3 - current_rv*0.7
            else:
                eta_nest = 3.3 + current_rv*5.3

            current_rv = np.random.randn()
            if current_rv < 0:
                lamb_nest = 1.14 - current_rv*0.09
            else:
                lamb_nest = 1.14 + current_rv*0.45


            a_temp_py[i], a_temp_qy[i] = get_ly_qy_from_nest_pars(energy, config_xe1t.d_cathode_voltages_to_field[cathode_setting], w_value_nest, alpha_nest, zeta_nest, beta_nest, gamma_nest, delta_nest, kappa_nest, eta_nest, lamb_nest)



        py_two_sigma_below, py_one_sigma_below, py_median, py_one_sigma_above, py_two_sigma_above = np.percentile(a_temp_py, [2.5, 16, 50, 84, 97.5])
        qy_two_sigma_below, qy_one_sigma_below, qy_median, qy_one_sigma_above, qy_two_sigma_above = np.percentile(a_temp_qy, [2.5, 16, 50, 84, 97.5])

        d_yields[cathode_setting]['a_py_median_nest'][bin_number] =py_median
        d_yields[cathode_setting]['a_py_lb_nest'][bin_number] = py_one_sigma_below
        d_yields[cathode_setting]['a_py_ub_nest'][bin_number] = py_one_sigma_above
        d_yields[cathode_setting]['a_py_2lb_nest'][bin_number] = py_two_sigma_below
        d_yields[cathode_setting]['a_py_2ub_nest'][bin_number] = py_two_sigma_above
        
        d_yields[cathode_setting]['a_qy_median_nest'][bin_number] = qy_median
        d_yields[cathode_setting]['a_qy_lb_nest'][bin_number] = qy_one_sigma_below
        d_yields[cathode_setting]['a_qy_ub_nest'][bin_number] = qy_one_sigma_above
        d_yields[cathode_setting]['a_qy_2lb_nest'][bin_number] = qy_two_sigma_below
        d_yields[cathode_setting]['a_qy_2ub_nest'][bin_number] = qy_two_sigma_above



# get yields from bands
a_temp_py = np.zeros(num_steps_band)
a_temp_qy = np.zeros(num_steps_band)
count_fields = 0
for cathode_setting in l_cathode_settings_in_use:
    d_yields[cathode_setting]['a_py_median_band'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_py_lb_band'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_py_ub_band'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_py_2lb_band'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_py_2ub_band'] = np.zeros(num_energies)
    
    d_yields[cathode_setting]['a_qy_median_band'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_qy_lb_band'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_qy_ub_band'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_qy_2lb_band'] = np.zeros(num_energies)
    d_yields[cathode_setting]['a_qy_2ub_band'] = np.zeros(num_energies)

    for bin_number, energy in tqdm.tqdm(enumerate(a_energies)):
        for i in xrange(num_steps_band):
            
            #a_current_pars = a_samples_band[-(np.random.randint(1, num_steps_to_pull_from*num_walkers) % total_length_sampler), :]
            a_current_pars = a_samples_band[-i, :]
            
            w_value = a_current_pars[0]/1000.
            alpha = a_current_pars[1]
            zeta = a_current_pars[2]
            beta = a_current_pars[3]
            gamma = a_current_pars[4]
            delta = a_current_pars[5]
            kappa = a_current_pars[6]
            eta = a_current_pars[7]
            lamb = a_current_pars[8]
            
            
            a_temp_py[i], a_temp_qy[i] = get_ly_qy_from_nest_pars(energy, config_xe1t.d_cathode_voltages_to_field[cathode_setting], w_value, alpha, zeta, beta, gamma, delta, kappa, eta, lamb)



        py_two_sigma_below, py_one_sigma_below, py_median, py_one_sigma_above, py_two_sigma_above = np.percentile(a_temp_py, [2.5, 16, 50, 84, 97.5])
        qy_two_sigma_below, qy_one_sigma_below, qy_median, qy_one_sigma_above, qy_two_sigma_above = np.percentile(a_temp_qy, [2.5, 16, 50, 84, 97.5])

        d_yields[cathode_setting]['a_py_median_band'][bin_number] =py_median
        d_yields[cathode_setting]['a_py_lb_band'][bin_number] = py_one_sigma_below
        d_yields[cathode_setting]['a_py_ub_band'][bin_number] = py_one_sigma_above
        d_yields[cathode_setting]['a_py_2lb_band'][bin_number] = py_two_sigma_below
        d_yields[cathode_setting]['a_py_2ub_band'][bin_number] = py_two_sigma_above
        
        d_yields[cathode_setting]['a_qy_median_band'][bin_number] = qy_median
        d_yields[cathode_setting]['a_qy_lb_band'][bin_number] = qy_one_sigma_below
        d_yields[cathode_setting]['a_qy_ub_band'][bin_number] = qy_one_sigma_above
        d_yields[cathode_setting]['a_qy_2lb_band'][bin_number] = qy_two_sigma_below
        d_yields[cathode_setting]['a_qy_2ub_band'][bin_number] = qy_two_sigma_above


    count_fields += 1



# --------------------------------------------
# --------------------------------------------
#  Points from Plante et al
# --------------------------------------------
# --------------------------------------------


def multiply_leff_array_by_co57_ly(l_leff):
    for i in xrange(len(l_leff)):
        l_leff[i] *= 63.
    return l_leff

l_plante_energies = [3, 3.9, 5.0, 6.5, 8.4, 10.7, 14.8, 55.2]
l_plante_energies_unc = [0.6, 0.7, 0.8, 1.0, 1.3, 1.6, 1.3, 8.8]
l_plante_ly = [0.088, 0.095, 0.098, 0.121, 0.139, 0.143, 0.144, 0.268]
l_plante_ly_unc = [[0.015, 0.016, 0.015, 0.010, 0.011, 0.010, 0.009, 0.013], [0.014, 0.015, 0.014, 0.010, 0.011, 0.010, 0.009, 0.013]]

l_plante_ly = multiply_leff_array_by_co57_ly(l_plante_ly)
l_plante_ly_unc[0] = multiply_leff_array_by_co57_ly(l_plante_ly_unc[0])
l_plante_ly_unc[1] = multiply_leff_array_by_co57_ly(l_plante_ly_unc[1])



# --------------------------------------------
# --------------------------------------------
#  Points from Lux
# --------------------------------------------
# --------------------------------------------

l_lux_energies_qy = [0.70, 1.10, 1.47, 2.00, 2.77, 3.86, 5.55, 8.02, 11.52, 16.56, 24.2]
l_lux_energies_qy_unc = [0.13, 0.18, 0.12, 0.10, 0.10, 0.08, 0.09, 0.10, 0.12, 0.16, 0.2]
l_lux_qy = [8.2, 7.4, 10.1, 8.0, 7.5, 7.3, 7.2, 6.8, 5.88, 5.28, 4.62]
l_lux_qy_unc = [[2.1, 1.7, 1.6, 0.6, 0.5, 0.3, 0.2, 0.17, 0.13, 0.13, 0.10], [2.4, 1.9, 1.5, 0.9, 0.5, 0.3, 0.2, 0.15, 0.12, 0.11, 0.13]]

l_lux_energies_py = [1.08, 1.92, 3.13, 4.45, 5.89, 7.44, 9.1, 10.9, 12.8]
l_lux_energies_py_unc = [0.13, 0.09, 0.11, 0.11, 0.13, 0.17, 0.2, 0.3, 0.3]
l_lux_py = [4.9, 5.2, 4.9, 6.4, 6.1, 7.4, 7.9, 8.1, 8.9]
l_lux_py_unc = [[1.0, 0.4, 0.4, 0.4, 0.3, 0.4, 0.4, 0.5, 0.4], [1.2, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6]]



first_shade = 0.2
second_shade = 0.1


f, (ax1, ax2) = plt.subplots(2)

ax1.set_xlim(min_energy, max_energy)
ax1.set_ylim(3.5, 16)
ax1.set_xscale('log', nonposx='clip')
ax1.set_yscale('log', nonposx='clip')
ax1.set_xlabel('Energy [keV]')
ax1.set_ylabel(r'$L_{y} [\frac{photons}{keV}]$')

for cathode_setting in l_cathode_settings_in_use:

    nest_handle, = ax1.plot(a_energies, d_yields[cathode_setting]['a_py_median_nest'], marker='', linestyle='--', color='r', label='NEST 120 V/cm')
    ax1.fill_between(a_energies, d_yields[cathode_setting]['a_py_lb_nest'], d_yields[cathode_setting]['a_py_ub_nest'], facecolor='r', alpha=first_shade, interpolate=True)
    ax1.fill_between(a_energies, d_yields[cathode_setting]['a_py_2lb_nest'], d_yields[cathode_setting]['a_py_2ub_nest'], facecolor='r', alpha=second_shade, interpolate=True)

    xe1t_handle, = ax1.plot(a_energies, d_yields[cathode_setting]['a_py_median_band'], marker='', linestyle='--', color='b', label='XENON1T 120 V/cm')
    ax1.fill_between(a_energies, d_yields[cathode_setting]['a_py_lb_band'], d_yields[cathode_setting]['a_py_ub_band'], facecolor='b', alpha=first_shade, interpolate=True)
    ax1.fill_between(a_energies, d_yields[cathode_setting]['a_py_2lb_band'], d_yields[cathode_setting]['a_py_2ub_band'], facecolor='b', alpha=second_shade, interpolate=True)

plante_handle = ax1.errorbar(l_plante_energies, l_plante_ly, xerr=l_plante_energies_unc, yerr=l_plante_ly_unc, marker='.', color='black', linestyle='', label='Plante 0 V/cm')
lux_handle = ax1.errorbar(l_lux_energies_py, l_lux_py, xerr=l_lux_energies_py_unc, yerr=l_lux_py_unc, fmt='c.', label='LUX 200 V/cm')




ax2.set_xlim(min_energy, max_energy)
ax2.set_ylim(3.5, 15)
ax2.set_xscale('log', nonposx='clip')
ax2.set_yscale('log', nonposx='clip')
ax2.set_xlabel('Energy [keV]')
ax2.set_ylabel(r'$Q_{y} [\frac{electrons}{keV}]$')

for cathode_setting in l_cathode_settings_in_use:

    # NEST
    ax2.plot(a_energies, d_yields[cathode_setting]['a_qy_median_nest'], marker='', linestyle='--', color='r')
    ax2.fill_between(a_energies, d_yields[cathode_setting]['a_qy_lb_nest'], d_yields[cathode_setting]['a_qy_ub_nest'], facecolor='r', alpha=first_shade, interpolate=True)
    ax2.fill_between(a_energies, d_yields[cathode_setting]['a_qy_2lb_nest'], d_yields[cathode_setting]['a_qy_2ub_nest'], facecolor='r', alpha=second_shade, interpolate=True)

    # band
    ax2.plot(a_energies, d_yields[cathode_setting]['a_qy_median_band'], marker='', linestyle='--', color='b')
    ax2.fill_between(a_energies, d_yields[cathode_setting]['a_qy_lb_band'], d_yields[cathode_setting]['a_qy_ub_band'], facecolor='b', alpha=first_shade, interpolate=True)
    ax2.fill_between(a_energies, d_yields[cathode_setting]['a_qy_2lb_band'], d_yields[cathode_setting]['a_qy_2ub_band'], facecolor='b', alpha=second_shade, interpolate=True)


ax2.errorbar(l_lux_energies_qy, l_lux_qy, xerr=l_lux_energies_qy_unc, yerr=l_lux_qy_unc, fmt='c.')

#ax1.legend(handles=[nest_handle, xe1t_handle, plante_handle, lux_handle], loc='best', fontsize=5)
ax1.legend(loc='best', fontsize='8')

s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)

plt.tight_layout()

d_best_fit_yields = {}
d_best_fit_yields['energy'] = a_energies

d_best_fit_yields['py_median'] = d_yields[cathode_setting]['a_py_median_band']
d_best_fit_yields['py_minus_one_sigma'] = d_yields[cathode_setting]['a_py_lb_band']
d_best_fit_yields['py_plus_one_sigma'] = d_yields[cathode_setting]['a_py_ub_band']
d_best_fit_yields['py_minus_two_sigma'] = d_yields[cathode_setting]['a_py_2lb_band']
d_best_fit_yields['py_plus_two_sigma'] = d_yields[cathode_setting]['a_py_2ub_band']

d_best_fit_yields['qy_median'] = d_yields[cathode_setting]['a_qy_median_band']
d_best_fit_yields['qy_minus_one_sigma'] = d_yields[cathode_setting]['a_qy_lb_band']
d_best_fit_yields['qy_plus_one_sigma'] = d_yields[cathode_setting]['a_qy_ub_band']
d_best_fit_yields['qy_minus_two_sigma'] = d_yields[cathode_setting]['a_qy_2lb_band']
d_best_fit_yields['qy_plus_two_sigma'] = d_yields[cathode_setting]['a_qy_2ub_band']


f.savefig('%sly_qy_comparison.png' % (s_path_for_save))
pickle.dump(d_best_fit_yields, open('%sbest_fit_yields.p' % (s_path_for_save), 'w'))


#plt.show()






