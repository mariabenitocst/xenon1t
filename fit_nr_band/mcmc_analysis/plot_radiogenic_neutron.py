#!/usr/bin/python
import sys, array, os
sys.path.insert(0, '..')

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.interpolate import spline

import numpy as np
import cPickle as pickle
import tqdm

import ROOT as root

import config_xe1t

dir_specifier_name = 'run_0_band'

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


l_plots = ['plots', dir_specifier_name, '%s_kV_%s_deg' % (s_cathode_settings, s_degree_settings)]


s_path_for_save = './'
for directory in l_plots:
    s_path_for_save += directory + '/'

if not os.path.exists(s_path_for_save):
    os.makedirs(s_path_for_save)


lax_version = '0.11.1'

l_s1_settings = [70, 0, 70]
l_log_settings = config_xe1t.l_log_settings
l_s2_settings = config_xe1t.l_s2_settings


s1_edges = np.linspace(l_s1_settings[1], l_s1_settings[2], l_s1_settings[0]+1)
log_edges = np.linspace(l_log_settings[1], l_log_settings[2], l_log_settings[0]+1)
s2_edges = np.linspace(l_s2_settings[1], l_s2_settings[2], l_s2_settings[0]+1)

bin_edges_s1_th2 = np.linspace(config_xe1t.l_s1_settings_pl[1], config_xe1t.l_s1_settings_pl[2], config_xe1t.l_s1_settings_pl[0]+1)
bin_edges_s1_th2 = np.asarray(bin_edges_s1_th2, dtype=np.float32)
bin_edges_s2_th2 = np.linspace(np.log10(config_xe1t.l_s2_settings_pl[1]), np.log10(config_xe1t.l_s2_settings_pl[2]), config_xe1t.l_s2_settings_pl[0]+1)
bin_edges_s2_th2 = np.asarray(bin_edges_s2_th2, dtype=np.float32)

roi_lb_cut_s1 = 3
for i, bin_edge in enumerate(s1_edges):
    # cut at 3 PE for dark matter range
    if bin_edge >= roi_lb_cut_s1:
        break

cut_bin = i
assert cut_bin >= 0



d_arrays = pickle.load(open('./mc_output/radiogenic_neutron_lax_0.11.1.p', 'r'))


fig_radiogenic_neutron, ax_radiogenic_neutron = plt.subplots(1)

a_hist_2d, _, _ = np.histogram2d(d_arrays['d_mc']['s1'], np.log10(d_arrays['d_mc']['s2']/d_arrays['d_mc']['s1']), bins=[s1_edges, log_edges], weights=d_arrays['d_mc']['weight'])
a_hist_2d *= d_arrays['scale_factor']

a_roi_hist2d = a_hist_2d[cut_bin:, :]

cax_radiogenic_neutron = ax_radiogenic_neutron.pcolor(s1_edges, log_edges, a_hist_2d.T, cmap='Blues')
ax_radiogenic_neutron.axvline(x=roi_lb_cut_s1, color='r', linestyle='--')
fig_radiogenic_neutron.colorbar(cax_radiogenic_neutron, label=r'$\frac{Events}{Year}$', format='%.2e')


ax_radiogenic_neutron.set_title(r'Radiogenic Neutron Bkg')
ax_radiogenic_neutron.set_xlabel(r'S1 [PE]')
ax_radiogenic_neutron.set_ylabel(r'$Log_{10}(\frac{S2}{S1})$')

ax_radiogenic_neutron.text(0.45, 0.85, r'$\mathrm{Integrated\ Rate} = %.2e \ \frac{\mathrm{Events}}{\mathrm{Year}}$' % (np.sum(a_roi_hist2d)), transform=ax_radiogenic_neutron.transAxes, fontsize=10)

#ax_radiogenic_neutron.set_xlim(d_arrays[label]['bin_centers'][0], d_arrays[label]['bin_centers'][-1])
#ax_radiogenic_neutron.set_ylim(0, 1.05)

ax_radiogenic_neutron.legend(loc='best', fontsize=10)

fig_radiogenic_neutron.savefig('%sradiogenic_neutron_%s.png' % (s_path_for_save, lax_version))



# make ROOT histogram for pl
f_constant = root.TF2('constant', '0.0000000001', bin_edges_s1_th2[0], bin_edges_s1_th2[-1], bin_edges_s2_th2[0], bin_edges_s2_th2[-1])
f_hist = root.TFile('./mc_output/radiogenic_neutron_bkg_%s.root' % (lax_version), 'RECREATE')

s_key = 'radiogenic_neutron_bkg'
h_current = root.TH2F(s_key, s_key, config_xe1t.l_s1_settings_pl[0], bin_edges_s1_th2, config_xe1t.l_s2_settings_pl[0], bin_edges_s2_th2)
h_current.Add(f_constant)

for i in tqdm.tqdm(xrange(len(d_arrays['d_mc']['s1'].values))):
    current_s1 = d_arrays['d_mc']['s1'].values[i]
    current_s2 = d_arrays['d_mc']['s2'].values[i]
    current_weight = d_arrays['d_mc']['weight'].values[i]
    if ((current_s1 > bin_edges_s1_th2[0]) & (current_s1 < bin_edges_s1_th2[-1]) & (np.log10(current_s2) > bin_edges_s2_th2[0]) & (np.log10(current_s2) < bin_edges_s2_th2[-1])):
        h_current.Fill(current_s1, np.log10(current_s2), current_weight)

h_current.Scale(d_arrays['scale_factor']/365.)
h_current.Write()

print h_current.Integral()

f_hist.Close()



#plt.show()
