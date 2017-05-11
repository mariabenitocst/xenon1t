import os.path
import numpy as np

path_to_this_module = os.path.dirname(__file__)
results_directory_name = path_to_this_module + '/mcmc_analysis/mcmc_results_local'

path_to_fit_inputs = '%s/fit_inputs/' % (path_to_this_module)

path_to_mc_outputs = '%s/mcmc_analysis/mc_output/' % (path_to_this_module)

l_energy_settings = [300, 0, 70]
l_energy_settings_yield_plots = [100, 2, 20]
l_energy_settings_uniform_nr = [140, 0, 70]

num_bins = 30
l_s1_settings = [num_bins, 1, 100]
l_s2_settings = [num_bins, 57, 5000]
l_log_settings = [num_bins, 0.5, 2.5]
#l_log_settings = [num_bins, 1.4, 3.0]

a_s1_bin_edges = np.logspace(np.log10(l_s1_settings[1]), np.log10(l_s1_settings[2]), l_s1_settings[0]+1, dtype=np.float32)
#a_s1_bin_edges = np.linspace(l_s1_settings[1], l_s1_settings[2], l_s1_settings[0]+1, dtype=np.float32)
a_log_bin_edges = np.linspace(l_log_settings[1], l_log_settings[2], num=l_log_settings[0]+1, dtype=np.float32)

l_s1_settings_pl = [67, 3, 70]
l_s2_settings_pl = [70, 50, 8000] # log spaced
#l_s2_settings_pl = [70, 200, 25000]#[70, 50, 8000] # log spaced

l_quantiles = [20, 80]

l_allowed_degree_settings = [-4]
l_allowed_cathode_settings = [12.]

d_cathode_voltages_to_field = {12:120} # in kV:V/cm

detector_r = 45 # double check

# detector paramaters
max_r = 36.94 #42. # cm
#max_r_fv = 39.85 # 47.9 # cm
min_z = -83.45
max_z = -13.4

# cylinder parameters
max_r_cylinder = 36.94 # cm
min_z_cylinder = -92.9 # cm
max_z_cylinder = -9 # cm

z_gate = 0
e_drift_velocity = 0.144 # cm/us



# ------------------------------------------------
# ------------------------------------------------
# Set default parameter values for use in MC
# ------------------------------------------------
# ------------------------------------------------


w_value = 13.7
w_value_uncertainty = 0.2

g1_value = 0.1442 #0.1471#0.1511 #0.109
g1_uncertainty = 0.0068 #0.0042 #0.0035

spe_res_value = 0.363
spe_res_uncertainty = 0.0001

dpe_lb = 0.17
dpe_ub = 0.24
dpe_median = (dpe_ub-dpe_lb)
dpe_std = (1./12.*(dpe_ub-dpe_lb)**2)**0.5

bias_median = (1. - 0.)/0.5
bias_std = (1./12.*(1. - 0.)**2)**0.5

extraction_efficiency_value = 0.961 #0.936
extraction_efficiency_uncertainty = 0.046 #0.034

gas_gain_value = 11.69 #32.90 #11.69 #15.53
gas_gain_uncertainty = 0.26 #0.26*32.9/11.69 #0.26 #2.25

gas_gain_width =  8.22  #3.73
gas_gain_width_uncertainty = 0.0001

l_means_pax_bias = [0.]
l_cov_matrix_pax_bias = [0.01]

l_means_pax_smearing = [0.05]
l_cov_matrix_pax_smearing = [0.000001]

prob_er_value = 0.5
prob_er_uncertainty = 0.1

cut_acceptance_s1_intercept = 0.96517
cut_acceptance_s1_intercept_uncertainty = 0.0421*cut_acceptance_s1_intercept

cut_acceptance_s1_slope = -0.000466585
cut_acceptance_s1_slope_uncertainty = 0.0421*cut_acceptance_s1_slope

cut_acceptance_s2_intercept = 0.851629
cut_acceptance_s2_intercept_uncertainty = 0.0176*cut_acceptance_s2_intercept

cut_acceptance_s2_slope = 7.20775e-06
cut_acceptance_s2_slope_uncertainty = 0.0176*cut_acceptance_s2_slope

ms_par_0 = 1.2
ms_par_0_unc = 0.3

ms_par_1 = 3.422
ms_par_1_unc = 0.682

# NEST model values and uncertainties

nest_lindhard_model = {}
nest_lindhard_model['values'] = {'alpha':1.240,
                                 'zeta':0.0472,
                                 'beta':239,
                                 'gamma':0.01385,
                                 'delta':0.0620,
                                 'kappa':0.1394,
                                 'eta':3.3,
                                 'lambda':1.14}
nest_lindhard_model['uncertainty'] = {'alpha':[-0.073, 0.079],
                                      'zeta':[-0.0073, 0.0088],
                                      'beta':[-8.8, 28],
                                      'gamma':[-0.00073, 0.00058],
                                      'delta':[-0.0064, 0.0056],
                                      'kappa':[-0.0026, 0.0032],
                                      'eta':[-0.7, 5.3],
                                      'lambda':[-0.09, 0.45]}







