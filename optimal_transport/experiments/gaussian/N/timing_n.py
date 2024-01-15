import numpy as np
import optimal_transport.apdamd.ot
from optimal_transport.greenkhorn.ot import OT as greenkhorn
from optimal_transport.sinkhorn.ot import OT as sinkhorn
from optimal_transport.apdamd.ot import OT as apdamd
from optimal_transport.tests.sample_problem import sample_gaussian_OT_exact, simple_problem
from optimal_transport.ot import cost
import timeit
import os

# Create the problem
n = 1  # dim
N = 100  # number of points
Gaussian1 = (np.zeros(n), np.eye(n))
cho_sigma = np.eye(n) + np.triu(np.ones((n, n))) * 0.224745
Gaussian2 = (np.ones(n, ), cho_sigma.T @ cho_sigma)
print(cho_sigma.T @ cho_sigma)
# my_problem = sample_gaussian_OT_exact(N, n, None, True)
my_problem = sample_gaussian_OT_exact(N, n, (Gaussian1, Gaussian2), True)
Ns = [25, 50, 75, 100, 150]
eps = 0.5

timings_sinkhorn = dict()
timings_greenkhorn = dict()
timings_apdamd = dict()
std_timings_sinkhorn = dict()
std_timings_greenkhorn = dict()
std_timings_apdamd = dict()
transport_plans_sinkhorn = dict()
transport_plans_greenkhorn = dict()
transport_plans_apdamd = dict()
costs_sinkhorn = dict()
costs_greenkhorn = dict()
costs_apdamd = dict()
realised_iters_sinkhorn = dict()
realised_iters_greenkhorn = dict()
realised_iters_apdamd = dict()
theoretical_iters_sinkhorn = dict()
theoretical_iters_greenkhorn = dict()
theoretical_iters_apdamd = dict()
converge_sinkhorn = dict()
converge_greenkhorn = dict()
converge_apdamd = dict()

NAME = "run_1"

ENABLE_GREENKHORN = True
ENABLE_SINKHORN = True
ENABLE_APDAMD = True

n_iter = None

CODE_SINKHORN = """
sinkhorn(None, *my_problem[1:], eps=eps, iter_max=n_iter)
"""

CODE_GREENKHORN = """
greenkhorn(None, *my_problem[1:], eps=eps, iter_max=n_iter)
"""

CODE_APDAMD = """
apdamd( * my_problem, eps, iter_max=n_iter)
"""

REPEAT = 10
NUMBER = 1

if not os.path.exists(f'{NAME}.npz'):

    # Theoretical bounds on hte number of iterations
    for i, N in enumerate(Ns):
        my_problem = sample_gaussian_OT_exact(N, n, (Gaussian1, Gaussian2), True)
        theoretical_iters_apdamd[N] = optimal_transport.apdamd.ot.theoretical_bound_on_iter(*my_problem[1:],
                                                                                            eps)
        theoretical_iters_greenkhorn[N] = optimal_transport.greenkhorn.ot.theoretical_bound_on_iter(
            *my_problem[1:], eps)
        print(
            f"Theoretical upper bound on the number of iterations for Greenkhorn: {theoretical_iters_greenkhorn[N]}")
        print(f"Theoretical upper bound on the number of iterations for APDAMD: {theoretical_iters_apdamd[N]}")

    for i, N in enumerate(Ns):
        print(f"N: {N}")
        my_problem = sample_gaussian_OT_exact(N, n, (Gaussian1, Gaussian2), True)
        theoretical_iters_apdamd[N] = optimal_transport.apdamd.ot.theoretical_bound_on_iter(*my_problem[1:],
                                                                                            eps)
        theoretical_iters_greenkhorn[N] = optimal_transport.greenkhorn.ot.theoretical_bound_on_iter(
            *my_problem[1:], N)
        if ENABLE_GREENKHORN:
            transport_plans_greenkhorn[N], _ = greenkhorn(None, *my_problem[1:], eps=eps, iter_max=n_iter)
            costs_greenkhorn[N] = (cost(my_problem[1], transport_plans_greenkhorn[N]))
            print(costs_greenkhorn[N])
            timing_greenkhorn = timeit.repeat(CODE_GREENKHORN, repeat=REPEAT, number=NUMBER, globals=globals())
            timings_greenkhorn[N] = np.mean(timing_greenkhorn)
            std_timings_greenkhorn[N] = np.std(timing_greenkhorn)
            realised_iters_greenkhorn[N] = _
            converge_greenkhorn[N] = _ < min(n_iter if n_iter is not None else np.inf,
                                             theoretical_iters_greenkhorn[N])
        if ENABLE_SINKHORN:
            transport_plans_sinkhorn[N], _ = sinkhorn(None, *my_problem[1:], eps=eps, iter_max=n_iter)
            costs_sinkhorn[N] = (cost(my_problem[1], transport_plans_sinkhorn[N]))
            print(costs_sinkhorn[N])
            timing_sinkhorn = timeit.repeat(CODE_SINKHORN, repeat=REPEAT, number=NUMBER, globals=globals())
            timings_sinkhorn[N] = np.mean(timing_sinkhorn)
            std_timings_sinkhorn[N] = np.std(timing_sinkhorn)
            realised_iters_sinkhorn[N] = _

        if ENABLE_APDAMD:
            transport_plans_apdamd[N], _ = apdamd(None, *my_problem[1:], eps=eps, iter_max=n_iter)
            costs_apdamd[N] = cost(my_problem[1], transport_plans_apdamd[N])
            print(costs_apdamd[N])
            timing_apdamd = timeit.repeat(CODE_APDAMD, repeat=REPEAT, number=NUMBER, globals=globals())
            timings_apdamd[N] = np.mean(timing_apdamd)
            std_timings_apdamd[N] = np.std(timing_apdamd)
            realised_iters_apdamd[N] = _
            converge_apdamd[N] = _ < min(n_iter if n_iter is not None else np.inf,
                                         theoretical_iters_apdamd[N])
    print("finish")

    np.savez(f'./{NAME}.npz', timings_sinkhorn=timings_sinkhorn, timings_greenkhorn=timings_greenkhorn,
             timings_apdamd=timings_apdamd, std_timings_sinkhorn=std_timings_sinkhorn,
             std_timings_greenkhorn=std_timings_greenkhorn, std_timings_apdamd=std_timings_apdamd,
             transport_plans_sinkhorn=transport_plans_sinkhorn, transport_plans_greenkhorn=transport_plans_greenkhorn,
             transport_plans_apdamd=transport_plans_apdamd, costs_sinkhorn=costs_sinkhorn,
             costs_greenkhorn=costs_greenkhorn, costs_apdamd=costs_apdamd,
             realised_iters_sinkhorn=realised_iters_sinkhorn,
             realised_iters_greenkhorn=realised_iters_greenkhorn, realised_iters_apdamd=realised_iters_apdamd,
             theoretical_iters_greenkhorn=theoretical_iters_greenkhorn,
             theoretical_iters_apdamd=theoretical_iters_apdamd,
             converge_greenkhorn=converge_greenkhorn, converge_apdamd=converge_apdamd)

res = np.load(f'./{NAME}.npz', allow_pickle=True)
import csv


def write(path, mydict):
    with open(path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])


write(f'./{NAME}/timings_sinkhorn.csv', res['timings_sinkhorn'][()])
write(f'./{NAME}/timings_greenkhorn.csv', res['timings_greenkhorn'][()])
write(f'./{NAME}/timings_apdamd.csv', res['timings_apdamd'][()])
write(f'./{NAME}/std_timings_sinkhorn.csv', res['std_timings_sinkhorn'][()])
write(f'./{NAME}/std_timings_greenkhorn.csv', res['std_timings_greenkhorn'][()])
write(f'./{NAME}/std_timings_apdamd.csv', res['std_timings_apdamd'][()])
write(f'./{NAME}/costs_sinkhorn.csv', res['costs_sinkhorn'][()])
write(f'./{NAME}/costs_greenkhorn.csv', res['costs_greenkhorn'][()])
write(f'./{NAME}/costs_apdamd.csv', res['costs_apdamd'][()])
write(f'./{NAME}/realised_iters_sinkhorn.csv', res['realised_iters_sinkhorn'][()])
write(f'./{NAME}/realised_iters_greenkhorn.csv', res['realised_iters_greenkhorn'][()])
write(f'./{NAME}/realised_iters_apdamd.csv', res['realised_iters_apdamd'][()])
write(f'./{NAME}/theoretical_iters_greenkhorn.csv', res['theoretical_iters_greenkhorn'][()])
write(f'./{NAME}/theoretical_iters_apdamd.csv', res['theoretical_iters_apdamd'][()])
