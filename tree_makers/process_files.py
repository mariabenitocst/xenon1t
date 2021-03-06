import click
import sys
import hax
from collections import defaultdict
import hax_config

if len(sys.argv) != 3:
	print('Use is "python process_files.py <filename> <treemaker file>')
	print('Filename must have either a root extension (single file) or a txt extension (list of root files).')
	print('Treemaker file and class must share a name.')
	sys.exit()


# check extension first
# if txt then is a list of files
# if .root it is pax processed single file
filename = sys.argv[1]
treemaker_file = sys.argv[2]
if treemaker_file[-3:] == '.py':
	treemaker_file = treemaker_file[:-3]

path_to_file_lists = './file_lists/'
mod_treemaker = __import__(treemaker_file)


if filename[-4:] == 'root':
	print('\nSingle ROOT file given...\n')
	l_files = [filename[:-5]]
elif filename[-3:] == 'txt':
	print('\nText file given...\n')
	l_files = []
	with open(path_to_file_lists + filename, 'r') as f_file_list:
		for line in f_file_list:
			if line[0] == '#':
				continue
			if line[-1] == '\n' or line[-1] == '\t' or line[-1] == ' ':
				line = line[:-1]
			l_files.append(line)

else:
	print('\nFilename given is not root or txt, please try again with appropriate file.\n\n')
	sys.exit()

hc = hax_config.hax_config('XENON100')

# eventually need to add check on file to see if it
# is xenon100 or xenon1t
hax.init(main_data_paths=[hc.hax_pax_processed_data_path], experiment=hc.hax_experiment, minitree_paths=[hc.hax_minitree_path])

hax.minitrees.load(l_files, treemakers=[getattr(sys.modules[treemaker_file], treemaker_file)], force_reload=False)
#hax.minitrees.load(l_files, treemakers=[mod_treemaker[treemaker_file]], force_reload=True)




