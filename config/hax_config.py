import sys

hax_experiment = None
hax_pax_processed_data_path = None
hax_minitree_path = None

def hax_config_init(experiment):
	hax_experiment = experiment
	if experiment == 'XENON1T':
		hax_pax_processed_data_path = '/project/lgrandi/xenon1t/processed/pax_v5.0.0/'
	elif experiment == 'XENON100':
		hax_pax_processed_data_path = '/project/lgrandi/xenon100/archive/root/merged/xenon100/Pax4.4.0_Reprocessed_9/'
	else:
		print('\nNot currently setup to handle experiment "%s".  Options are "XENON1T" and "XENON100"')
		sys.exit()

	hax_minitree_path = '/home/mda2149/xenon1t/minitrees/'



