#!/usr/bin/python
import sys
import os, re
import random

from math import floor
import numpy as np
import pandas as pd
import tqdm

import config_xe1t

import cPickle as pickle

import ROOT as root


class nr_analysis_xe1t(object):
    def __init__(self, identifier, lax_version, num_mc_events, num_walkers, num_steps_to_include, device_number=0, b_mc_paper_mode=False, wimp_mass=False, b_conservative_acceptance_posterior=False):
    
        self.identifier = identifier
        #l_possible_identifiers = ['ambe', 'wimp', 'radiogenic_neutron', 'uniform_nr', 'band']
        l_possible_identifiers = ['cnns', 'radiogenic_neutron', 'uniform_nr', 'ambe', 'ambe_f', 'wimp']
    
        if self.identifier == 'wimp':
            assert wimp_mass != False
            self.wimp_mass = wimp_mass
    
    
        if not (self.identifier in l_possible_identifiers):
            print '\nCannot run for that configuration, please change class accordingly.\n'
            sys.exit()
    
        # num_steps_to_include is how large of sampler you keep
        self.num_steps_to_include = num_steps_to_include
        self.num_mc_events = int(num_mc_events)
        self.device_number = device_number
        self.b_conservative_acceptance_posterior = b_conservative_acceptance_posterior
        
        
        # expected rate will be calculated in each
        # otherwise will be set to 1 such that PDF is given
        self.expected_rate = 1


        self.b_mc_paper_comparison = b_mc_paper_mode
        if self.b_mc_paper_comparison:
            print '\n\nUsing MC Paper Comparison Mode'
            print 'This means acceptances=1, g1=0.12, G=8.12\n\n'


        self.num_walkers = int(num_walkers)

        self.fiducial_volume_mass = 1000. # kg
        print '\n\nGiving rate assuming %0.f kg FV\n\n\n' % (self.fiducial_volume_mass)

        self.lax_version = lax_version
        print '\n\nCurrently using %s for fitting\n\n\n' % (self.lax_version)

        self.dir_specifier_name = 'run_0_band'


        l_degree_settings_in_use = [-4]
        s_degree_settings = ''
        for degree_setting in l_degree_settings_in_use:
            s_degree_settings += '%s,' % (degree_setting)
        s_degree_settings = s_degree_settings[:-1]


        l_cathode_settings_in_use = [12.]
        s_cathode_settings = ''
        for cathode_setting in l_cathode_settings_in_use:
            s_cathode_settings += '%.3f,' % (cathode_setting)
        s_cathode_settings = s_cathode_settings[:-1]

        name_of_results_directory = config_xe1t.results_directory_name
        l_plots = ['plots', self.dir_specifier_name, '%s_kV_%s_deg' % (s_cathode_settings, s_degree_settings)]

        #s_path_to_file = './%s/%s/%s_kV_%s_deg/sampler_dictionary_170424_high_pax_acceptance.p' % (name_of_results_directory, self.dir_specifier_name, s_cathode_settings, s_degree_settings)
        s_path_to_file = './%s/%s/%s_kV_%s_deg/sampler_dictionary.p' % (name_of_results_directory, self.dir_specifier_name, s_cathode_settings, s_degree_settings)

        self.s_path_to_plots = './plots/%s/%s_kV_%s_deg/' % (self.dir_specifier_name, s_cathode_settings, s_degree_settings)

        if os.path.exists(s_path_to_file):
            dSampler = pickle.load(open(s_path_to_file, 'r'))
            l_chains = []
            l_ln_likelihoods = []
            for sampler in dSampler[num_walkers]:
                l_chains.append(sampler['_chain'])
                l_ln_likelihoods.append(sampler['_lnprob'])

            a_full_sampler = np.concatenate(l_chains, axis=1)
            a_full_ln_likelihood = np.concatenate(l_ln_likelihoods, axis=1)

            print 'Successfully loaded sampler!'
        else:
            print s_path_to_file
            print 'Could not find file!'
            sys.exit()


        # get block and grid size
        self.d_gpu_scale = {}
        self.block_dim = 1024
        self.d_gpu_scale['block'] = (self.block_dim,1,1)
        self.num_blocks = floor(num_mc_events / float(self.block_dim))
        self.d_gpu_scale['grid'] = (int(self.num_blocks), 1)
        num_mc_events = int(self.num_blocks*self.block_dim)



        # get significant value sets to loop over later
        self.a_samples = a_full_sampler[:, -num_steps_to_include:, :].reshape((-1, a_full_sampler.shape[2]))
        self.a_best_fit = np.percentile(self.a_samples, 50., axis=0)
    
        """
        print '\nUsing best ln likelihood from chain\n'
        a_best_fit_indices = np.unravel_index(np.argmax(a_full_ln_likelihood), a_full_ln_likelihood.shape)
        self.a_best_fit = a_full_sampler[a_best_fit_indices]
        """
        
        
        
        
    # get shortened samples array
    def get_mcmc_samples(self):
        return self.a_samples
    
    
    
    def get_best_fit_mcmc(self):
        return self.a_best_fit



    # this function returns a dictionary with all of the info
    # needed to run the gpu code with the exception of binning/histograms
    # or arrays that will be filled
    def prepare_gpu(self):

        self.d_plotting_information = {}

        # need to prepare GPU for MC simulations
        import cuda_full_observables_production
        from pycuda.compiler import SourceModule
        import pycuda.driver as drv
        import pycuda.tools
        import pycuda.gpuarray

        drv.init()
        self.dev = drv.Device(self.device_number)
        self.ctx = self.dev.make_context()
        print 'Device Name: %s\n' % (self.dev.name())



        if not (self.identifier == 'ambe' or self.identifier == 'ambe_f'):
            self.d_mc_energy = pickle.load(open('%swimp_mc.p' % config_xe1t.path_to_fit_inputs, 'r'))
        else:
            self.d_mc_energy = pickle.load(open('%sambe_mc.p' % config_xe1t.path_to_fit_inputs, 'r'))

        
        self.d_er_band = pickle.load(open('%ser_band.p' % config_xe1t.path_to_fit_inputs, 'r'))



        # -----------------------------------------
        #  Energy
        # -----------------------------------------

        # need to use switch on energy
        # the only used output will be a_mc_energy (nothing else
        # should have "self"
        
        self.a_mc_energy = np.zeros(self.num_mc_events, dtype=np.float32)

        if self.identifier == 'cnns':

            df_cnns_hist = pd.read_table('./cnns_spectra/cnns-1keV.txt', sep=' ')
            d_cnns_hist = {}
            d_cnns_hist['energy_kev'] = np.asarray(df_cnns_hist['energy_kev'], dtype=np.float32)
            d_cnns_hist['counts'] = np.asarray(df_cnns_hist['counts'], dtype=np.float32)


            df_cnns_hist = pd.read_table('./cnns_spectra/cnns-1keV.txt', sep=' ')
            d_cnns_hist = {}
            d_cnns_hist['energy_kev'] = np.asarray(df_cnns_hist['energy_kev'], dtype=np.float32)
            d_cnns_hist['counts'] = np.asarray(df_cnns_hist['counts'], dtype=np.float32)

            # ['energy_kev', 'events/ton/yr/keV']
                


            bin_width = d_cnns_hist['energy_kev'][1] - d_cnns_hist['energy_kev'][0]

            cdf = np.cumsum(d_cnns_hist['counts'])
            self.expected_rate = cdf[-1]*bin_width*self.fiducial_volume_mass/1000. # events/yr
            cdf = cdf / cdf[-1]

            print '\n\nExpected rate from energy spectra should be 90*1 events/yr (170323)'
            print 'Found expected rate to be %.2e events/yr\n\n' % (self.expected_rate)

            values = np.random.rand(self.num_mc_events)
            value_bins = np.searchsorted(cdf, values)
            random_from_cdf = d_cnns_hist['energy_kev'][value_bins]


            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                current_random_num = np.random.random()*bin_width + random_from_cdf[i]
                
                # need to cover edge case of zero bin
                if current_random_num < 0:
                    current_random_num = -current_random_num
                
                self.a_mc_energy[i] = current_random_num


        elif self.identifier == 'radiogenic_neutron':
        
            f_rn = root.TFile('./radiogenic_neutrons_spectra/RadiogenicNeutrons_SR0.root')
            h_energy = f_rn.hEdTotal_1tFVcylSR0
            h_pos = f_rn.h2dTotal_1tFVcylSR0

            self.expected_rate = h_energy.Integral(1, 300)*365*self.fiducial_volume_mass # events/yr
            
            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                current_random_num = h_energy.GetRandom()
                
                self.a_mc_energy[i] = current_random_num


        elif self.identifier == 'uniform_nr':
            self.a_mc_energy = np.random.uniform(config_xe1t.l_energy_settings[1], config_xe1t.l_energy_settings[2], size=self.num_mc_events)
            self.a_mc_energy = np.asarray(self.a_mc_energy, dtype=np.float32)
            
            
        elif self.identifier == 'ambe' or self.identifier == 'ambe_f':
            pass
            """
            bin_width = self.d_mc_energy['a_energy_bins'][1] - self.d_mc_energy['a_energy_bins'][0]

            cdf = np.cumsum(self.d_mc_energy['a_energy_hist'])
            cdf = cdf / cdf[-1]
            values = np.random.rand(self.num_mc_events)
            value_bins = np.searchsorted(cdf, values)
            random_from_cdf = self.d_mc_energy['a_energy_bins'][value_bins]

            self.a_mc_energy = np.zeros(self.num_mc_events, dtype=np.float32)
            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                current_random_num = np.random.random()*bin_width + random_from_cdf[i]
                
                # need to cover edge case of zero bin
                if current_random_num < 0:
                    current_random_num = -current_random_num
                
                self.a_mc_energy[i] = current_random_num
            """
                
        elif self.identifier == 'wimp':
        
            df_wimp_hist = pd.read_csv('./wimp_spectra/wimp_%dgev_1e-45cm2.csv' % (self.wimp_mass))
            d_wimp_hist = {}
            d_wimp_hist['energy_kev'] = np.asarray(df_wimp_hist['kev'], dtype=np.float32)
            d_wimp_hist['rate_per_kev_kg_day'] = np.asarray(df_wimp_hist['events_perkg_perkev_perday'], dtype=np.float32)
        
            bin_width = d_wimp_hist['energy_kev'][1] - d_wimp_hist['energy_kev'][0]

            cdf = np.cumsum(d_wimp_hist['rate_per_kev_kg_day'])
            self.expected_rate = cdf[-1]*bin_width*self.fiducial_volume_mass # at 1E-45 cm^2
            cdf = cdf / cdf[-1]



            values = np.random.rand(self.num_mc_events)
            value_bins = np.searchsorted(cdf, values)
            random_from_cdf = d_wimp_hist['energy_kev'][value_bins]

            self.a_mc_energy = np.zeros(self.num_mc_events, dtype=np.float32)
            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                current_random_num = np.random.random()*bin_width + random_from_cdf[i]
                
                # need to cover edge case of zero bin
                if current_random_num < 0:
                    current_random_num = -current_random_num
                
                self.a_mc_energy[i] = current_random_num


        # -----------------------------------------
        #  Z
        # -----------------------------------------
        
        self.a_mc_z = np.zeros(self.num_mc_events, dtype=np.float32)
        
        if self.identifier == 'cnns' or self.identifier == 'uniform_nr' or self.identifier == 'wimp':

            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                self.a_mc_z[i] = np.random.uniform(config_xe1t.min_z_cylinder, config_xe1t.max_z_cylinder)

        elif self.identifier == 'radiogenic_neutron':
            # will fill in x,y
            pass

        elif self.identifier == 'ambe' or self.identifier == 'ambe_f':
            pass
            """
            d_mc_positions = pickle.load(open('%smc_maps.p' % config_xe1t.path_to_fit_inputs, 'r'))
        
            bin_width = d_mc_positions['z_bin_edges'][1] - d_mc_positions['z_bin_edges'][0]

            cdf = np.cumsum(d_mc_positions['z_map'])
            cdf = cdf / cdf[-1]
            values = np.random.rand(self.num_mc_events)
            value_bins = np.searchsorted(cdf, values)
            random_from_cdf = d_mc_positions['z_bin_edges'][value_bins]

            self.a_mc_z = np.zeros(self.num_mc_events, dtype=np.float32)
            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                current_random_num = np.random.random()*bin_width + random_from_cdf[i]
                
                self.a_mc_z[i] = current_random_num
            """


        # -----------------------------------------
        #  X, Y
        # -----------------------------------------

        self.a_mc_x = np.zeros(self.num_mc_events, dtype=np.float32)
        self.a_mc_y = np.zeros(self.num_mc_events, dtype=np.float32)

        if self.identifier == 'cnns' or self.identifier == 'uniform_nr' or self.identifier == 'wimp':


            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                current_r = np.random.uniform(0, config_xe1t.max_r_cylinder**2)**0.5
                current_phi = np.random.uniform(0, 2*np.pi)

                current_random_num_x = current_r * np.cos(current_phi)
                current_random_num_y = current_r * np.sin(current_phi)
                
                
                self.a_mc_x[i] = current_random_num_x
                self.a_mc_y[i] = current_random_num_y


        elif self.identifier == 'radiogenic_neutron':
            for i in tqdm.tqdm(xrange(self.num_mc_events)):
                current_r = np.asarray(1e50, dtype=np.float64)
                current_z = np.asarray(0, dtype=np.float64)
                current_phi = np.random.uniform(0, 2*np.pi)
                
                h_pos.GetRandom2(current_r, current_z)
                # given in r^2 so must correct
                current_r = current_r**0.5

                current_random_num_x = current_r * np.cos(current_phi)
                current_random_num_y = current_r * np.sin(current_phi)
                
                # must divide by 10 to go from mm -> cm
                self.a_mc_x[i] = current_random_num_x/10.
                self.a_mc_y[i] = current_random_num_y/10.
                self.a_mc_z[i] = current_z/10.


        elif self.identifier == 'ambe' or self.identifier == 'ambe_f':

            num_times_to_copy_array = (len(self.d_mc_energy['a_energy'])%self.num_mc_events)+1
            self.a_mc_energy = np.concatenate([self.d_mc_energy['a_energy']]*num_times_to_copy_array)[:self.num_mc_events]
        

            # filling directly from arrays for x, y, energy
            self.a_mc_x = np.concatenate([self.d_mc_energy['a_x']]*num_times_to_copy_array)[:self.num_mc_events]
            self.a_mc_y = np.concatenate([self.d_mc_energy['a_y']]*num_times_to_copy_array)[:self.num_mc_events]
            self.a_mc_z = np.concatenate([self.d_mc_energy['a_z']]*num_times_to_copy_array)[:self.num_mc_events]


        # -----------------------------------------
        #  Electron Lifetime
        # -----------------------------------------

        bin_width = self.d_mc_energy['a_el_bins'][1] - self.d_mc_energy['a_el_bins'][0]

        cdf = np.cumsum(self.d_mc_energy['a_el_hist'])
        cdf = cdf / cdf[-1]
        values = np.random.rand(self.num_mc_events)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = self.d_mc_energy['a_el_bins'][value_bins]

        self.a_e_survival_prob = np.zeros(self.num_mc_events, dtype=np.float32)
        for i in tqdm.tqdm(xrange(self.num_mc_events)):
            current_random_num = np.random.random()*bin_width + random_from_cdf[i]
            
            # current random number is lifetime whch we need to convert
            # draw from z array to make sure they are connected!
            self.a_e_survival_prob[i] = np.exp(-(config_xe1t.z_gate - self.a_mc_z[i]) / config_xe1t.e_drift_velocity / current_random_num)

        if self.b_mc_paper_comparison:
            print 'Using MC Paper infinite electron lifetime'
            self.a_e_survival_prob = np.full(self.num_mc_events, 0.9999999, dtype=np.float32)





        # ------------------------------------------------
        # ------------------------------------------------
        # Pull required arrays for correction maps
        # ------------------------------------------------
        # ------------------------------------------------


        d_corrections = pickle.load(open('%ssignal_correction_maps.p' % config_xe1t.path_to_fit_inputs, 'r'))
            
        self.bin_edges_r2 = np.asarray(d_corrections['s1']['r2_bin_edges'], dtype=np.float32)
        self.bin_edges_z = np.asarray(d_corrections['s1']['z_bin_edges'], dtype=np.float32)
        self.s1_correction_map = np.asarray(d_corrections['s1']['map'], dtype=np.float32).T
        self.s1_correction_map = self.s1_correction_map.flatten()

        self.bin_edges_x = np.asarray(d_corrections['s2']['x_bin_edges'], dtype=np.float32)
        self.bin_edges_y = np.asarray(d_corrections['s2']['y_bin_edges'], dtype=np.float32)
        self.s2_correction_map = np.asarray(d_corrections['s2']['map'], dtype=np.float32).T
        self.s2_correction_map = self.s2_correction_map.flatten()



        # -----------------------------------------
        #  Get array of ER bands S1 and S2
        # -----------------------------------------


        bin_width_s1 = self.d_er_band['er_band_s1_edges'][1] - self.d_er_band['er_band_s1_edges'][0]
        bin_width_log = self.d_er_band['er_band_log_edges'][1] - self.d_er_band['er_band_log_edges'][0]

        cdf = np.cumsum(self.d_er_band['er_band_hist'].ravel())
        cdf = cdf / cdf[-1]
        values = np.random.rand(self.num_mc_events)
        value_bins = np.searchsorted(cdf, values)
        s1_idx, log_idx = np.unravel_index(value_bins, (len(self.d_er_band['er_band_s1_edges'])-1, len(self.d_er_band['er_band_log_edges'])-1))

        self.a_er_s1 = np.zeros(self.num_mc_events, dtype=np.float32)
        self.a_er_log = np.zeros(self.num_mc_events, dtype=np.float32)
        for i in tqdm.tqdm(xrange(self.num_mc_events)):
            current_random_num_s1 = np.random.random()*bin_width_s1 + self.d_er_band['er_band_s1_edges'][s1_idx[i]]
            current_random_num_log = np.random.random()*bin_width_log + self.d_er_band['er_band_log_edges'][log_idx[i]]
            
            
            self.a_er_s1[i] = current_random_num_s1
            self.a_er_log[i] = current_random_num_log






        # ------------------------------------------------
        # ------------------------------------------------
        # Pull required arrays for splines
        # ------------------------------------------------
        # ------------------------------------------------

        d_bias_smearing = pickle.load(open('%ss1_s2_bias_and_smearing.p' % (config_xe1t.path_to_fit_inputs), 'r'))
        self.a_s1bs_s1s = np.asarray(d_bias_smearing['s1']['points'], dtype=np.float32)
        self.a_s1bs_lb_bias = np.asarray(d_bias_smearing['s1']['lb_bias'], dtype=np.float32)
        self.a_s1bs_ub_bias = np.asarray(d_bias_smearing['s1']['ub_bias'], dtype=np.float32)
        self.a_s1bs_lb_smearing = np.asarray(d_bias_smearing['s1']['lb_smearing'], dtype=np.float32)
        self.a_s1bs_ub_smearing = np.asarray(d_bias_smearing['s1']['ub_smearing'], dtype=np.float32)
        self.a_s2bs_s2s = np.asarray(d_bias_smearing['s2']['points'], dtype=np.float32)
        self.a_s2bs_lb_bias = np.asarray(d_bias_smearing['s2']['lb_bias'], dtype=np.float32)
        self.a_s2bs_ub_bias = np.asarray(d_bias_smearing['s2']['ub_bias'], dtype=np.float32)
        self.a_s2bs_lb_smearing = np.asarray(d_bias_smearing['s2']['lb_smearing'], dtype=np.float32)
        self.a_s2bs_ub_smearing = np.asarray(d_bias_smearing['s2']['ub_smearing'], dtype=np.float32)

        if not (self.identifier == 'ambe' or self.identifier == 'ambe_f'):
            d_acceptances = pickle.load(open('%sacceptances_wimps.p' % (config_xe1t.path_to_fit_inputs), 'r'))
        else:
            d_acceptances = pickle.load(open('%sacceptances_ambe.p' % (config_xe1t.path_to_fit_inputs), 'r'))
        s_acceptance = ''

        # PF acceptance
        if self.b_mc_paper_comparison:
            print '\n\nTesting with 100% acceptance\n\n'
            d_acceptances['pf_s1']['y_values_lower'] = [1 for i in xrange(len(d_acceptances['pf_s1']['x_values']))]
            d_acceptances['pf_s1']['y_values_mean'] = [1 for i in xrange(len(d_acceptances['pf_s1']['x_values']))]
            d_acceptances['pf_s1']['y_values_upper'] = [1 for i in xrange(len(d_acceptances['pf_s1']['x_values']))]
            s_acceptance += ' (no acceptances or S2 threshold and g1=0.12)'

        self.a_s1pf_s1s = np.asarray(d_acceptances['pf_s1']['x_values'], dtype=np.float32)
        self.a_s1pf_lb_acc = np.asarray(d_acceptances['pf_s1']['y_values_lower'], dtype=np.float32)
        self.a_s1pf_mean_acc = np.asarray(d_acceptances['pf_s1']['y_values_mean'], dtype=np.float32)
        self.a_s1pf_ub_acc = np.asarray(d_acceptances['pf_s1']['y_values_upper'], dtype=np.float32)


        # cut acceptances
        if self.b_mc_paper_comparison:
            print '\n\nTesting with 100% acceptance\n\n'
            self.d_plotting_information['cut_acceptance_s1_intercept'] = 1.
            self.d_plotting_information['cut_acceptance_s1_intercept_uncertainty'] = 0

            self.d_plotting_information['cut_acceptance_s1_slope'] = 0.
            self.d_plotting_information['cut_acceptance_s1_slope_uncertainty'] = 0.

            self.d_plotting_information['cut_acceptance_s2_intercept'] = 1.
            self.d_plotting_information['cut_acceptance_s2_intercept_uncertainty'] = 0.

            self.d_plotting_information['cut_acceptance_s2_slope'] = 0.
            self.d_plotting_information['cut_acceptance_s2_slope_uncertainty'] = 0.
        else:
            self.d_plotting_information['cut_acceptance_s1_intercept'] = config_xe1t.cut_acceptance_s1_intercept
            self.d_plotting_information['cut_acceptance_s1_intercept_uncertainty'] = config_xe1t.cut_acceptance_s1_intercept_uncertainty

            self.d_plotting_information['cut_acceptance_s1_slope'] = config_xe1t.cut_acceptance_s1_slope
            self.d_plotting_information['cut_acceptance_s1_slope_uncertainty'] = config_xe1t.cut_acceptance_s1_slope_uncertainty

            self.d_plotting_information['cut_acceptance_s2_intercept'] = config_xe1t.cut_acceptance_s2_intercept
            self.d_plotting_information['cut_acceptance_s2_intercept_uncertainty'] = config_xe1t.cut_acceptance_s2_intercept_uncertainty

            self.d_plotting_information['cut_acceptance_s2_slope'] = config_xe1t.cut_acceptance_s2_slope
            self.d_plotting_information['cut_acceptance_s2_slope_uncertainty'] = config_xe1t.cut_acceptance_s2_slope_uncertainty




        print 'Putting arrays on GPU...'
        self.d_plotting_information['gpu_energies'] = pycuda.gpuarray.to_gpu(self.a_mc_energy)
        self.d_plotting_information['mc_energy'] = self.a_mc_energy

        self.d_plotting_information['gpu_x_positions'] = pycuda.gpuarray.to_gpu(self.a_mc_x)
        self.d_plotting_information['mc_x'] = self.a_mc_x

        self.d_plotting_information['gpu_y_positions'] = pycuda.gpuarray.to_gpu(self.a_mc_y)
        self.d_plotting_information['mc_y'] = self.a_mc_y

        self.d_plotting_information['gpu_z_positions'] = pycuda.gpuarray.to_gpu(self.a_mc_z)
        self.d_plotting_information['mc_z'] = self.a_mc_z

        self.d_plotting_information['gpu_e_survival_prob'] = pycuda.gpuarray.to_gpu(self.a_e_survival_prob)
                
        self.d_plotting_information['gpu_er_band_s1'] = pycuda.gpuarray.to_gpu(self.a_er_s1)
        self.d_plotting_information['gpu_er_band_log'] = pycuda.gpuarray.to_gpu(self.a_er_log)

        self.d_plotting_information['gpu_bin_edges_r2'] = pycuda.gpuarray.to_gpu(self.bin_edges_r2)

        self.d_plotting_information['gpu_bin_edges_z'] = pycuda.gpuarray.to_gpu(self.bin_edges_z)

        self.d_plotting_information['gpu_s1_correction_map'] = pycuda.gpuarray.to_gpu(self.s1_correction_map)

        self.d_plotting_information['gpu_bin_edges_x'] = pycuda.gpuarray.to_gpu(self.bin_edges_x)

        self.d_plotting_information['gpu_bin_edges_y'] = pycuda.gpuarray.to_gpu(self.bin_edges_y)

        self.d_plotting_information['gpu_s2_correction_map'] = pycuda.gpuarray.to_gpu(self.s2_correction_map)

        self.d_plotting_information['gpu_s1bs_s1s'] = pycuda.gpuarray.to_gpu(self.a_s1bs_s1s)

        self.d_plotting_information['gpu_s1bs_lb_bias'] = pycuda.gpuarray.to_gpu(self.a_s1bs_lb_bias)

        self.d_plotting_information['gpu_s1bs_ub_bias'] = pycuda.gpuarray.to_gpu(self.a_s1bs_ub_bias)

        self.d_plotting_information['gpu_s1bs_lb_smearing'] = pycuda.gpuarray.to_gpu(self.a_s1bs_lb_smearing)

        self.d_plotting_information['gpu_s1bs_ub_smearing'] = pycuda.gpuarray.to_gpu(self.a_s1bs_ub_smearing)

        self.d_plotting_information['gpu_s2bs_s2s'] = pycuda.gpuarray.to_gpu(self.a_s2bs_s2s)

        self.d_plotting_information['gpu_s2bs_lb_bias'] = pycuda.gpuarray.to_gpu(self.a_s2bs_lb_bias)

        self.d_plotting_information['gpu_s2bs_ub_bias'] = pycuda.gpuarray.to_gpu(self.a_s2bs_ub_bias)

        self.d_plotting_information['gpu_s2bs_lb_smearing'] = pycuda.gpuarray.to_gpu(self.a_s2bs_lb_smearing)

        self.d_plotting_information['gpu_s2bs_ub_smearing'] = pycuda.gpuarray.to_gpu(self.a_s2bs_ub_smearing)

        self.d_plotting_information['gpu_s1pf_s1s'] = pycuda.gpuarray.to_gpu(self.a_s1pf_s1s)

        self.d_plotting_information['gpu_s1pf_lb_acc'] = pycuda.gpuarray.to_gpu(self.a_s1pf_lb_acc)

        self.d_plotting_information['gpu_s1pf_mean_acc'] = pycuda.gpuarray.to_gpu(self.a_s1pf_mean_acc)

        self.d_plotting_information['gpu_s1pf_ub_acc'] = pycuda.gpuarray.to_gpu(self.a_s1pf_ub_acc)






        # get random seeds setup
        local_gpu_setup_kernel = pycuda.compiler.SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('setup_kernel')
        self.local_rng_states = drv.mem_alloc(np.int32(self.num_blocks*self.block_dim)*pycuda.characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
        local_gpu_setup_kernel(np.int32(int(self.num_blocks*self.block_dim)), self.local_rng_states, np.uint64(0), np.uint64(0), grid=(int(self.num_blocks), 1), block=(int(self.block_dim), 1, 1))


        # get observables function

        if self.identifier == 'cnns' or self.identifier == 'radiogenic_neutron' or self.identifier == 'wimp':
            gpu_function_name = 'gpu_full_observables_production_with_arrays_no_fv'
        elif self.identifier == 'ambe' or self.identifier == 'ambe_f':
            gpu_function_name = 'gpu_full_observables_production_with_arrays'
        elif self.identifier == 'uniform_nr':
            gpu_function_name = 'gpu_full_observables_production_with_arrays_no_fv_corrected_and_uncorrected_and_acceptances'


        self.gpu_observables_func = SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function(gpu_function_name)


        return self.d_plotting_information
        
        
        
    def get_rng_states(self):
        return self.local_rng_states



    def call_gpu_func(self, args):
        self.gpu_observables_func(*args, **self.d_gpu_scale)




    def yield_unfixed_parameters(self):
    
        # make dictionary of unfixed values (g1, acceptance_par, ...)
        d_sampler_values = {}
        for i in xrange(self.num_steps_to_include*self.num_walkers):
        
            a_fit_parameters = self.a_samples[-i, :]

            # yield the resulting dictionary
            yield self.organize_parameters_from_array(a_fit_parameters)


    
    
    def get_best_fit_parameters(self):
    
        a_fit_parameters = self.a_best_fit
        return self.organize_parameters_from_array(a_fit_parameters)
    
    
    
    
    # given array of parameters put everything into dictionary
    def organize_parameters_from_array(self, a_fit_parameters):
    
        d_sampler_values = {}
        count_parameters = 0

        d_sampler_values['w_value'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['alpha'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['zeta'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['beta'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['gamma'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['delta'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['kappa'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['eta'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['lamb'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1

        if self.b_mc_paper_comparison:
            print 'Using MC Paper g1 (12%)'
            d_sampler_values['g1_value'] = np.asarray(0.12, dtype=np.float32)
        else:
            d_sampler_values['g1_value'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        
        
        d_sampler_values['extraction_efficiency'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1

        if self.b_mc_paper_comparison:
            print 'Using MC Paper G (57/150*20/0.936=8.12 PE/e)'
            d_sampler_values['gas_gain_value'] = np.asarray(8.12, dtype=np.float32)
        else:
            d_sampler_values['gas_gain_value'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1


        d_sampler_values['gas_gain_width'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['dpe_prob'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1

        d_sampler_values['s1_bias_par'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['s1_smearing_par'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['s2_bias_par'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        d_sampler_values['s2_smearing_par'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
        count_parameters += 1
        
        if not self.identifier == 'ambe_f:'
            if not self.b_conservative_acceptance_posterior:
                d_sampler_values['acceptance_par'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
            else:
                print '\n\nSetting PAX acceptance parameter equal to zero (conservative posterior mode)\n\n'
                d_sampler_values['acceptance_par'] = np.asarray(0, dtype=np.float32)
            
            count_parameters += 1
        
        else:
            d_sampler_values['acceptance_par_0'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
            count_parameters += 1
            d_sampler_values['acceptance_par_1'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
            count_parameters += 1
        

        d_sampler_values['current_cut_acceptance_s1_intercept'] = np.asarray(self.d_plotting_information['cut_acceptance_s1_intercept'] + a_fit_parameters[count_parameters]*self.d_plotting_information['cut_acceptance_s1_intercept_uncertainty'], dtype=np.float32)
        d_sampler_values['current_cut_acceptance_s1_slope'] = np.asarray(self.d_plotting_information['cut_acceptance_s1_slope'] + a_fit_parameters[count_parameters]*self.d_plotting_information['cut_acceptance_s1_slope_uncertainty'], dtype=np.float32)
        
        d_sampler_values['current_cut_acceptance_s2_intercept'] = np.asarray(self.d_plotting_information['cut_acceptance_s2_intercept'] + a_fit_parameters[count_parameters]*self.d_plotting_information['cut_acceptance_s2_intercept_uncertainty'], dtype=np.float32)
        d_sampler_values['current_cut_acceptance_s2_slope'] = np.asarray(self.d_plotting_information['cut_acceptance_s2_slope'] + a_fit_parameters[count_parameters]*self.d_plotting_information['cut_acceptance_s2_slope_uncertainty'], dtype=np.float32)
        count_parameters += 1

        if self.identifier == 'cnns' or self.identifier == 'radiogenic_neutron' or self.identifier == 'uniform_nr' or self.identifier == 'wimp':
            d_sampler_values['prob_bkg'] = np.asarray(0, dtype=np.float32)
            d_sampler_values['scale_par'] = np.asarray(1, dtype=np.float32)
            count_parameters += 2
        elif (self.identifier == 'ambe' or self.identifier == 'ambe_f'):
            d_sampler_values['prob_bkg'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
            count_parameters += 1
            d_sampler_values['scale_par'] = np.asarray(a_fit_parameters[count_parameters], dtype=np.float32)
            count_parameters += 1
        
        
        d_sampler_values['num_pts_s1bs'] = np.asarray(len(self.a_s1bs_s1s), dtype=np.int32)
        d_sampler_values['num_pts_s2bs'] = np.asarray(len(self.a_s2bs_s2s), dtype=np.int32)
        d_sampler_values['num_pts_s1pf'] = np.asarray(len(self.a_s1pf_s1s), dtype=np.int32)

        d_sampler_values['num_bins_r2'] = np.asarray(len(self.bin_edges_r2)-1, dtype=np.int32)
        d_sampler_values['num_bins_z'] = np.asarray(len(self.bin_edges_z)-1, dtype=np.int32)
        d_sampler_values['num_bins_x'] = np.asarray(len(self.bin_edges_x)-1, dtype=np.int32)
        d_sampler_values['num_bins_y'] = np.asarray(len(self.bin_edges_y)-1, dtype=np.int32)



        # yield the resulting dictionary
        return d_sampler_values
    
        
        
    def get_scale_factor(self):
        # give correct normalization used to make histograms
        return self.expected_rate / float(self.num_mc_events)
    


    def get_save_name_beginning(self):
        if self.identifier != 'wimp':
            return '%s_%s' % (self.identifier, self.lax_version)
        else:
            return '%s_%04ld_GeV_%s' % (self.identifier, self.wimp_mass, self.lax_version)
            
    
    
    
    def get_path_to_plots(self):
        return self.s_path_to_plots
    
    
    
    def get_mc_energies(self):
        return self.a_mc_energy




    def end_gpu_context(self):
        # end GPU context
        self.ctx.pop()
        print 'Successfully ended GPU context!'


