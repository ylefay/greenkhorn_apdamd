import numpy as np
import matplotlib.pyplot as plt
from ot.greenkhorn.ot import OT as greenkhorn
from ot.apdamd.ot import OT as apdamd
from ot.tests.sample_problem import sample_gaussian_OT_exact
from ot.ot import cost
import timeit

# Create the problem
n = 1  # dim
N = 100  # number of points
Gaussian1 = (np.zeros(n), np.eye(n))
Gaussian2 = (np.ones(n), np.eye(n) + np.triu(np.ones(n)) * 0.5)
my_problem = sample_gaussian_OT_exact(N, n, (Gaussian1, Gaussian2))
plt.plot(my_problem[-2])
plt.show()