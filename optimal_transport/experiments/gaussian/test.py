import numpy as np
import matplotlib.pyplot as plt
from optimal_transport.greenkhorn.ot import OT as greenkhorn
from optimal_transport.apdamd.ot import OT as apdamd
from optimal_transport.tests.sample_problem import sample_gaussian_OT_exact, simple_problem
from optimal_transport.ot import cost
import timeit

# Create the problem
n = 1  # dim
N = 100  # number of points
Gaussian1 = (np.zeros(n), np.eye(n))
Gaussian2 = (np.ones(n), np.eye(n) + np.triu(np.ones(n)) * 0.5)
my_problem = sample_gaussian_OT_exact(N, n, (Gaussian1, Gaussian2))


my_simple_problem = simple_problem()
print(apdamd(*my_simple_problem, eps=0.01, iter_max=None))