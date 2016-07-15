import click
import sys
import hax
from collections import defaultdict


@click.command
@click.argument('filename', nargs=1)
def load_files(filename):
	# check extension first
	# if txt then is a list of files
	# if .root it is pax processed single file




	

     

if len(sys.argv) == 2:
    dataset = sys.argv[1]
    print("======= To be reduced: "+dataset)
    #hax.init(main_data_paths=['/project/lgrandi/xenon1t/processed/pax_v5.0.0/'], experiment='XENON1T')
    hax.init(main_data_paths=['/project/lgrandi/xenon1t/processed/pax_v5.0.0/'], experiment='XENON1T', minitree_paths=['/home/mda2149/xenon1t/minitrees'])
    data = hax.minitrees.load(dataset, treemakers=[S1S2Properties], force_reload=True)

else:
    hax.init(main_data_paths=['/project/lgrandi/xenon1t/processed/pax_v5.0.0/'], experiment='XENON1T', minitree_paths=['/home/mda2149/xenon1t/minitrees'])
    datasets = hax.runs.datasets
    print(datasets['trigger__status'])
    d_cs137 = datasets[(datasets['source__type']=='Cs137') & (datasets['trigger__status']=='processed')]
    d_cs137.to_csv('lifetime_files.txt', columns=['name'], header=False, index=False)
    
    print(d_cs137)
    l_datasets = d_cs137['name']
    #print(l_datasets)
    data = hax.minitrees.load(l_datasets, treemakers=[S1S2Properties], force_reload=False)
    

print(data['time'])
"""

hax.init(experiment='XENON1T')
datasets = hax.runs.datasets
d_cs137 = datasets[datasets['source__type']=='Cs137']
d_cs137.to_csv('lifetime_files.txt', columns=['name'], header=False, index=False)

"""

