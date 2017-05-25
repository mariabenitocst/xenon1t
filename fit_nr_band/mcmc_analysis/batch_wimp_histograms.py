from subprocess import Popen

"""
l_wimp_masses = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50, 70, 100, 120, 150, 200, 400, 1000, 2000]
#l_wimp_masses = [5, 6]

for mass in l_wimp_masses:
    print 'Running for %d GeV WIMP' % (mass)

    l_args = ['python', 'make_pl_histograms_wimps.py', '256', str(mass)]
    call(l_args)
"""

l_wimp_masses = [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4), (10, 5)]
#l_wimp_masses = [(12, 0), (15, 1), (20, 2), (30, 3), (40, 4), (50, 5)]
#l_wimp_masses = [(70, 0), (100, 1), (120, 2), (150, 3), (200, 4), (400, 5)]
#l_wimp_masses = [(1000, 0), (2000, 1), (3000, 2), (5000, 3), (7000, 4), (10000, 5)]


for mass, device_number in l_wimp_masses:
    print 'Running for %d GeV WIMP on GPU %d' % (mass, device_number)

    l_args = ['python', 'make_pl_wimp_arrays.py', '256', str(mass), str(device_number), 'f']
    Popen(l_args)


