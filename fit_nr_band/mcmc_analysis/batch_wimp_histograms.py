from subprocess import call

l_wimp_masses = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50, 70, 100, 120, 150, 200, 400, 1000, 2000]
#l_wimp_masses = [5, 6]

for mass in l_wimp_masses:
    print 'Running for %d GeV WIMP' % (mass)

    l_args = ['python', 'make_pl_histograms_wimps.py', '256', str(mass)]
    call(l_args)


