import sys, os


# input a filename with the following format
# first line should be the treemaker used
# all following lines should be filenames
# without the treemaker extension
# returns the list of filenames and the
# treemaker
def read_file_list(filename):
	l_files = []
	with open(path_to_file_lists + filename, 'r') as f_file_list:
		for i, line in enumerate(f_file_list):
			if line[0] == '#':
				continue
			if line[-1] == '\n' or line[-1] == '\t' or line[-1] == ' ':
				line = line[:-1]
			if i == 0:
				treemaker_name = line
				continue
			l_files.append(line[:-1])

	return l_files, treemaker_name







