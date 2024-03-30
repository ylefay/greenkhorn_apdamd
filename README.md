# JAX implementation of the Greenkhorn Algorithm for Discrete Regularised Optimal Transport

The regularised optimal transport problem consists in finding an optimal transport map $X\in\mathbb{R}^{n\times n}$, such that it minimizes the cost function given by

$$
\langle C, X\rangle - \eta H(X),
$$

under the marginal constraints, $X1_n = r$ and $X^{\top}1_n = c$, where $\eta > 0$ is a regularization parameter and $H(X) = -\sum_{i,j}X_{i,j}(\log(X_{i,j})-1)$ is the entropic regularization. The matrix $C$ has entries $C_{i,j}$ which are the costs to go from $i$ to $j$.
The dual problem reduces to computing two dual potentials which satisfy first-order conditions, which constitute a fixed-point system that the two potentials must satisfy. These two equations can be iterated one after the other, and this gives rise to the so-called Sinkhorn Algorithm.
The Greenkhorn Algorithm is a greedy version of Sinkhorn algorithm that reach the optimal complexity bounds derived for the Sinkhorn Algorithm.

# This repository includes
- A JAX implementation of the Greenkhorn Algorithm.
- A JAX implementation of the APDAMD Algorithm.
- Scripts that numerically validate the derived complexity bounds by [Lin, Tianyi and Ho, Nhat and Jordan, Michael, 2019].

# Reference
See [Lin, Tianyi and Ho, Nhat and Jordan, Michael, 2019]{https://proceedings.mlr.press/v97/lin19a.html}
