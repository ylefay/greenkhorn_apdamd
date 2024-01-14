import numpy as np
import optimal_transport.apdamd.ot
from optimal_transport.greenkhorn.ot import OT as greenkhorn
from optimal_transport.sinkhorn.ot import OT as sinkhorn
from optimal_transport.apdamd.ot import OT as apdamd
from optimal_transport.tests.sample_problem import sample_gaussian_OT_exact, simple_problem
from optimal_transport.ot import cost
from optimal_transport.gaussian_ot import optimal_cost_for_gaussian
import timeit
import os


# Create the problem
n = 1  # dim
N = 100  # number of points
Gaussian1 = (np.zeros(n), np.eye(n))
Gaussian2 = (np.ones(n), np.eye(n) + np.triu(np.ones((n, n))) * 0.5)
my_problem = sample_gaussian_OT_exact(N, n, None, True)

epss = np.linspace(0.2, 0.5, 10)
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

NAME = "run_4"

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

REPEAT = 5
NUMBER = 1

if not os.path.exists(f'{NAME}.npz'):

    # Theoretical bounds on hte number of iterations
    for i, eps in enumerate(epss):
        theoretical_iters_apdamd[eps] = optimal_transport.apdamd.ot.theoretical_bound_on_iter(*my_problem[1:],
                                                                                              eps)
        theoretical_iters_greenkhorn[eps] = optimal_transport.greenkhorn.ot.theoretical_bound_on_iter(
            *my_problem[1:], eps)
        print(
            f"Theoretical upper bound on the number of iterations for Greenkhorn: {theoretical_iters_greenkhorn[eps]}")
        print(f"Theoretical upper bound on the number of iterations for APDAMD: {theoretical_iters_apdamd[eps]}")

    for i, eps in enumerate(epss):
        print(f"eps: {eps}")
        if ENABLE_GREENKHORN:
            transport_plans_greenkhorn[eps], _ = greenkhorn(None, *my_problem[1:], eps=eps, iter_max=n_iter)
            costs_greenkhorn[eps] = (cost(my_problem[1], transport_plans_greenkhorn[eps]))
            print(costs_greenkhorn[eps])
            timing_greenkhorn = timeit.repeat(CODE_GREENKHORN, repeat=REPEAT, number=NUMBER, globals=globals())
            timings_greenkhorn[eps] = np.mean(timing_greenkhorn)
            std_timings_greenkhorn[eps] = np.std(timing_greenkhorn)
            realised_iters_greenkhorn[eps] = _
            converge_greenkhorn[eps] = _ < min(n_iter if n_iter is not None else np.inf,
                                               theoretical_iters_greenkhorn[eps])
        if ENABLE_SINKHORN:
            transport_plans_sinkhorn[eps], _ = sinkhorn(None, *my_problem[1:], eps=eps, iter_max=n_iter)
            costs_sinkhorn[eps] = (cost(my_problem[1], transport_plans_sinkhorn[eps]))
            print(costs_sinkhorn[eps])
            timing_sinkhorn = timeit.repeat(CODE_SINKHORN, repeat=REPEAT, number=NUMBER, globals=globals())
            timings_sinkhorn[eps] = np.mean(timing_sinkhorn)
            std_timings_sinkhorn[eps] = np.std(timing_sinkhorn)
            realised_iters_sinkhorn[eps] = _

        if ENABLE_APDAMD:
            transport_plans_apdamd[eps], _ = apdamd(None, *my_problem[1:], eps=eps, iter_max=n_iter)
            costs_apdamd[eps] = cost(my_problem[1], transport_plans_apdamd[eps])
            print(costs_apdamd[eps])
            timing_apdamd = timeit.repeat(CODE_APDAMD, repeat=REPEAT, number=NUMBER, globals=globals())
            timings_apdamd[eps] = np.mean(timing_apdamd)
            std_timings_apdamd[eps] = np.std(timing_apdamd)
            realised_iters_apdamd[eps] = _
            converge_apdamd[eps] = _ < min(n_iter if n_iter is not None else np.inf,
                                           theoretical_iters_apdamd[eps])
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
