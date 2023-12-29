import numpy as np
from ot.tests.sample_problem import sample_problem, simple_problem
from ot.greenkhorn.ot import OT as greenkhorn
from ot.apdamd.ot import OT as apdamd

OT = [greenkhorn, apdamd]


def test(OT, eps, problem=None, n=None):
    if problem is None:
        if n is None:
            n = np.random.randint(2, 10)
        X, C, r, c = sample_problem(n)
    else:
        X, C, r, c = problem
    tp, n_iter = OT(X, C, r, c, eps)
    return tp


if __name__ == '__main__':
    np.random.seed(1)
    for ot_method in OT:
        test(ot_method, eps=0.5)
        test(ot_method, eps=0.08, problem=simple_problem())
        simple_sol = test(ot_method, eps=0.5, problem=simple_problem())
        simple_sol = test(ot_method, eps=4 * np.log(2), problem=simple_problem())
