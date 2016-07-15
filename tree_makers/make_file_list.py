import click
import sys
import hax
from collections import defaultdict
import hax_config



@click.command
@click.argument('experiment', nargs=1)
def load_files(filename):
	# file name will be the output files

	file_path = './file_lists/'
	hax_config.init(experiment)
	
	# initialize hax
	hax.init(main_data_paths=[hax_config.hax_pax_processed_data_path], experiment=experiment, minitree_paths=[hax_config.hax_minitree_path])
	datasets = hax.runs.datasets
	print(datasets)



if __name__ == '__main__':
	load_files(filename)

     