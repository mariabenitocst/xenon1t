import sys

class hax_config:
	def __init__(self, experiment):

		self.hax_experiment = experiment
		if experiment == 'XENON1T':
			self.hax_pax_processed_data_path = '/project/lgrandi/xenon1t/processed/pax_v5.0.0/'
		elif experiment == 'XENON100':
			self.hax_pax_processed_data_path = '/project/lgrandi/xenon100/archive/root/merged/xenon100/PaxReprocessed_v4.4.0_OSG/'
		else:
			print('\nNot currently setup to handle experiment "%s".  Options are "XENON1T" and "XENON100"')
			sys.exit()

		self.hax_minitree_path = '/home/mda2149/xenon1t/minitrees/'
		self.hax_file_lists_path = '/home/mda2149/xenon1t/minitrees/file_lists/'



