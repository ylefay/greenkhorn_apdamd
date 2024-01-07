from ot.greenkhorn.ot import OT as greenkhorn
from ot.apdamd.ot import OT as apdamd
from ot.tests.sample_problem import *
from ot.ot import cost
from timeit import repeat
import matplotlib.pyplot as plt

TITLE = 'timings_apdamd_only'
my_problem = np.load('apdamd_only_low_espilon/my_problem_5_1_0.npz', allow_pickle=True)
my_problem = (my_problem['X'], my_problem['C'], my_problem['r'], my_problem['c'])
epss = np.linspace(0.001, 0.005, 10)
timings_greenkhorn = dict()
timings_apdamd = dict()
std_timings_greenkhorn = dict()
std_timings_apdamd = dict()
transport_plans_greenkhorn = dict()
transport_plans_apdamd = dict()
costs_greenkhorn = dict()
costs_apdamd = dict()
n_iter = None
repeatition = 5
apdamd_code = '''
apdamd( * my_problem, eps, iter_max=n_iter)
'''
greenkhorn_code = '''
greenkhorn( * my_problem, eps, iter_max=n_iter)
'''

for i, eps in enumerate(epss):
    # transport_plans_greenkhorn[(n_iter, eps)], _ = greenkhorn(None, *my_problem[1:], eps=eps, iter_max=n_iter)
    # print(_)
    transport_plans_apdamd[(n_iter, eps)], _ = apdamd(*my_problem, eps=eps, iter_max=n_iter)
    print(_)
    costs_apdamd[(n_iter, eps)] = cost(my_problem[1], transport_plans_apdamd[(n_iter, eps)])
    # costs_greenkhorn[(n_iter, eps)] = (cost(my_problem[1], transport_plans_greenkhorn[(n_iter, eps)]))
    # timing_greenkhorn = repeat(stmt=greenkhorn_code, repeat=repeatition, number=1, globals=locals())
    timing_apdamd = repeat(stmt=apdamd_code, repeat=repeatition, number=1, globals=locals())
    # timings_greenkhorn[(n_iter, eps)] = np.mean(timing_greenkhorn)
    # std_timings_greenkhorn[(n_iter, eps)] = np.std(timing_greenkhorn)
    timings_apdamd[(n_iter, eps)] = np.mean(timing_apdamd)
    std_timings_apdamd[(n_iter, eps)] = np.std(timing_apdamd)

np.savez(f'./{TITLE}.npz', timings_greenkhorn=timings_greenkhorn, timings_apdamd=timings_apdamd,
         std_timings_greenkhorn=std_timings_greenkhorn, std_timings_apdamd=std_timings_apdamd,
         transport_plans_greenkhorn=transport_plans_greenkhorn, transport_plans_apdamd=transport_plans_apdamd,
         costs_greenkhorn=costs_greenkhorn, costs_apdamd=costs_apdamd)

result = np.load(f'./{TITLE}.npz', allow_pickle=True)
# timings_greenkhorn = result['timings_greenkhorn'][()]
timings_apdamd = result['timings_apdamd'][()]
# std_timings_greenkhorn = result['std_timings_greenkhorn'][()]
std_timings_apdamd = result['std_timings_apdamd'][()]
# plt.errorbar(epss, np.log([timings_greenkhorn[(n_iter, eps)] for eps in epss]),
#             np.array([std_timings_greenkhorn[(n_iter, eps)] for eps in epss]) / np.array(
#                 [timings_greenkhorn[(n_iter, eps)] for eps in epss]), label='greenkhorn')
plt.errorbar(epss, np.log([timings_apdamd[(n_iter, eps)] for eps in epss]),
             np.array([std_timings_apdamd[(n_iter, eps)] for eps in epss]) / np.array(
                 [timings_apdamd[(n_iter, eps)] for eps in epss])
             , label='apdamd')
plt.plot(epss, -1 * np.log(epss) + np.log(epss[0]) + np.log(timings_apdamd[(n_iter, epss[0])]), label='O(1/eps)')
# plt.plot(epss, -2 * np.log(epss) + 2 * np.log(epss[0]) + np.log(timings_greenkhorn[(n_iter, epss[0])]),
#         label='O(1/eps^2)')

plt.legend()
plt.savefig(f'./complexity_wrt_reg_{TITLE}.png')
plt.close()

# plt.plot(epss, [costs_greenkhorn[(n_iter, eps)] for eps in epss], label='greenkhorn')
plt.plot(epss, [costs_apdamd[(n_iter, eps)] for eps in epss], label='apdamd')
plt.legend()
plt.savefig(f'./cost_wrt_reg_{TITLE}.png')
