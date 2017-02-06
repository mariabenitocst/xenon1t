#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def sample_discrete_gaussian(mean_under_test, width_under_test):

    # go 5 sigma from center
    lower_bound_for_integral = int(round(mean_under_test - 5*width_under_test))
    upper_bound_for_integral = int(round(mean_under_test + 5*width_under_test))

    integral_of_dist = 0.
    for i in xrange(upper_bound_for_integral-lower_bound_for_integral+1):
        k = i + lower_bound_for_integral
        integral_of_dist += np.exp(-(k-mean_under_test)**2. / width_under_test**2. / 2.)
    

    r_uniform = np.random.uniform()

    cumulative_dist = 0.

    # add lower bound
    cumulative_dist = np.exp(-(lower_bound_for_integral-mean_under_test)**2. / width_under_test**2. / 2.) / integral_of_dist

    if r_uniform < cumulative_dist:
        return lower_bound_for_integral
    else:
        for i in xrange(upper_bound_for_integral-lower_bound_for_integral-1):
            k = 1 + lower_bound_for_integral + i
            cumulative_dist += np.exp(-(k-mean_under_test)**2. / width_under_test**2. / 2.) / integral_of_dist

            if (r_uniform > cumulative_dist) and (r_uniform < cumulative_dist + np.exp(-((k+1)-mean_under_test)**2. / width_under_test**2. / 2.) / integral_of_dist):
                return k+1


    # at this point must return upper bound since all others failed
    return upper_bound_for_integral




num_trials = 1000
mean_under_test = 5.5
width_under_test = 1.5

l_samples = [0 for i in xrange(num_trials)]

for i in xrange(num_trials):
    l_samples[i] = sample_discrete_gaussian(mean_under_test, width_under_test)

#print l_samples
print np.mean(l_samples)
print np.std(l_samples, ddof=1)


