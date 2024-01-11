import jax.numpy as np
import matplotlib.pyplot as plt
from optimal_transport.greenkhorn.ot import OT as greenkhorn
from optimal_transport.tests.sample_problem import sample_gaussian_OT_exact, simple_problem
from optimal_transport.ot import cost
from optimal_transport.apdamd.ot import OT
import timeit
import jax
from optimal_transport.ot import penalised_cost

# Create the problem
n = 1  # dim
N = 30  # number of points
eps = 0.06
Gaussian1 = (np.zeros(n), np.eye(n))
Gaussian2 = (np.ones(n), np.eye(n) + np.triu(np.ones((n, n))) * 0.5)
my_problem = sample_gaussian_OT_exact(N, n, (Gaussian1, Gaussian2))
eta = eps / (4 * np.log(my_problem[1].shape[0]))

# my_simple_problem = simple_problem()
my_simple_problem = my_problem

X, _ = OT(None, my_simple_problem[1], my_simple_problem[2], my_simple_problem[3], eps=eps, iter_max=None)
print(penalised_cost(my_simple_problem[1], X, eta))
X, _ = greenkhorn(None, *my_simple_problem[1:], eps=eps)
print(penalised_cost(my_simple_problem[1], X, eta))
