import numpy as np
from ot.tests.sample_problem import sample_gaussian_OT
import jax

jax.config.update("jax_enable_x64", True)

ID = 0
dim = 1
n_sample = 5
my_problem = sample_gaussian_OT(dim, n_sample, None)
my_problem = (my_problem[0], my_problem[1], my_problem[2], my_problem[3])
np.savez(f'my_problem_{n_sample}_{dim}_{ID}.npz', X=my_problem[0], C=my_problem[1], r=my_problem[2], c=my_problem[3])
