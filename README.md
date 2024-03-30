# JAX implementation of the Greenkhorn Algorithm for Discrete Regularised Optimal Transport

The regularised optimal transport problem consists in finding an optimal transport map $X\in\mathbb{R}^{n\times n}$, such that it minimizes the cost function given by

$$
\langle C, X\rangle - \eta H(X),
$$

under the marginal constraints, $X1_n = r$ and $X^{\top}1_n = c$, where $\eta > 0$ is a regularization parameter and $H(X) = -\sum_{i,j}X_{i,j}(\log(X_{i,j})-1)$ is the entropic regularization.
The dual problem reduces to computing two dual potentials which satisfy first-order conditions, this constitutes a fixed-point system. These two equations can be iterated one after the other, and this gives rise to the so-called Sinkhorn Algorithm.
The Greenkhorn Algorithm is a greedy version of Sinkhorn algorithm that reach the optimal complexity bounds derived for the Sinkhorn Algorithm. At each iteration of the Sinkhorn Algorithm, the procedure performs $n$ lines or rows modifications on the variable of interest, whereas for the Greenkhorn algorithm, only one row or column is modified at each iteration. The adapative primal dual accelerated mirror descent algorithm is basically a mirror descent algorithm with a backtracking line search procedure.

# This repository includes
- A JAX implementation of the Greenkhorn Algorithm.
- A JAX implementation of the APDAMD Algorithm.
- Scripts that numerically validate the derived complexity bounds by [Lin, Tianyi and Ho, Nhat and Jordan, Michael, 2019].

# Reference
See [Lin, Tianyi and Ho, Nhat and Jordan, Michael, 2019]{https://proceedings.mlr.press/v97/lin19a.html}
