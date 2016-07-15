import click
import sys
import hax
from collections import defaultdict
import hax_config

if len(sys.argv) != 2:
	print 'Use is "python make_file_list.py <experiment name>'
	sys.exit()


file_path = './file_lists/'
hax_config.init(experiment)

# initialize hax
hax.init(main_data_paths=[hax_config.hax_pax_processed_data_path], experiment=experiment, minitree_paths=[hax_config.hax_minitree_path])
datasets = hax.runs.datasets
print(datasets)



