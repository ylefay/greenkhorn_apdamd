import jax.lax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from optimal_transport.ot import r_fun, c_fun

jax.config.update("jax_enable_x64", True)

"""
https://jmlr.org/papers/volume23/20-277/20-277.pdf
"""


class F:
    """
    Primal objective function.
    """

    def __init__(self, c, eta):
        """
        Args:
            c: vectorized version of cost matrix C, shape (n ** 2,)
            eta: regularization parameter
        """
        self.c = c
        self.eta = eta

    def __call__(self, x):
        return jnp.dot(self.c, x) + self.eta * (xlogy(x, x) - x).sum()  # there was a mistake here  +x

    def grad(self, x):
        return self.c + self.eta * jnp.log(x)  # there was a mistake here  + self.eta


class Phi:
    """
    Dual objective function.
    """

    def __init__(self, C, eta, a, b):
        """
        args:
            C: Cost matrix, of shape (n, n)
            gamma: regularization parameter, positive scalar
            a: source distribution, of shape (n,)
            b: destination distribution, of shape (n,)
        """
        self.eta = eta
        # Gibbs kernel
        self.gibbs = jnp.exp(-C / eta)
        self.a = a
        self.b = b
        self.n = a.shape[0]
        # This is b in the "Ax = b" part of APDAMD
        self.stack = jnp.hstack((a, b))
        self.f = F(C.flatten(), eta)

    def x_lamb(self, y, z):
        """
        x(lambda) in dual formulation. This is the solution to maximizing the
        Lagrangian. Here the dual variables lambda = (y, z).
        Args:
            y: column dual variables, of shape (n,)
            z: row dual variables, of shape (n,)
        """
        X = jnp.exp(-z / self.eta) * (jnp.exp(-y / self.eta) * self.gibbs.T).T  # 1/jnp.exp(1) *
        return X.flatten()

    def _A_transpose_lambda(self, y, z):
        """
        Implicitly compute A^T lambda.
        args:
            y: dual variables corresponding to column constraints, of shape (n,)
            z: dual variables corresponding to row constraints, of shape (n,)
        """
        return jnp.add(y.reshape(self.n, 1), z).flatten()

    def _A_x(self, x):
        """
        Implicitly compute A x, equal to [X 1, X^T 1]
        args:
            x: flattened variables, of shape (n^2,)
        """
        X = x.reshape(self.n, self.n)
        return jnp.hstack((r_fun(X), c_fun(X)))

    def __call__(self, lamb):
        # Get dual variables
        y, z = lamb[:self.n], lamb[self.n:]
        val = jnp.dot(self.a, y) + jnp.dot(self.b, z)
        # Maximizing Lagrangian to get x(lambda)
        x_lamb = self.x_lamb(y, z).flatten()
        val -= self.f(x_lamb)
        val -= jnp.dot(self._A_transpose_lambda(y, z), x_lamb)
        return val

    def grad(self, lamb):
        """
        Gradient of the dual objective function with respect to lambda.
        args:
            lamb: dual variables, of shape (2n,)
        """
        y, z = lamb[:self.n], lamb[self.n:]
        # Maximizing Lagrangian to get x(lambda)
        x_lamb = self.x_lamb(y, z)
        return self.stack - self._A_x(x_lamb)


def apdamd(C, r, c, eta, tol, iter_max):
    """
    Adaptive Primal-Dual Accelerated Mirror Descent algorithm.
    JAX adaptation of https://github.com/joshnguyen99/partialot/blob/d7387fa3cee7fcecc4c3fa3b020bcec834798a6a/ot_solvers/apdamd.py#L75

    ///
    This implementation is based on Lin, Ho and Jordan NO:2019.
    ///

    This implementation is based on https://jmlr.org/papers/volume23/20-277/20-277.pdf
    It uses delta = n and B_phi(lambda1, lambda2) = (1 / 2n) ||lambda1 - lambda2||^2.


    returns:
        X: Optimal transport matrix. Non-neg array of shape (n, n).
    """

    n = r.shape[0]

    # Create primal and dual objectives
    phi = Phi(C, eta, r, c)

    ##### STEP 3: Run APDAGD until (eps_p / 2) accuracy #####
    delta = n
    L = 1
    alpha_bar = 0
    z = jnp.zeros(2 * n)
    Lambda = jnp.zeros(2 * n)
    x = jnp.zeros(n ** 2)

    def linesearch_criterion(inps):
        _, lhs, rhs, *_ = inps
        return lhs > rhs

    def iter_linesearch(inps):
        n_iter, _, _, M, z_t, _, lamb_t, _, _, alpha_bar_t, _ = inps
        # Line search
        # Initial guess for L
        M = 2 * M

        # Compute the step size
        alpha_t_1 = (1 + jnp.sqrt(1 + 4 * delta * M * alpha_bar_t)) / (2 * delta * M)

        # Compute the average coefficient
        alpha_bar_t_1 = alpha_bar_t + alpha_t_1

        # Compute the first average step
        mu_t_1 = (alpha_t_1 * z_t + alpha_bar_t * lamb_t) / alpha_bar_t_1

        # Compute the mirror descent
        grad_phi_mu_t_1 = phi.grad(mu_t_1)
        z_t_1 = z_t - 2 * n * alpha_t_1 * grad_phi_mu_t_1

        # Compute the second average step
        lamb_t_1 = (alpha_t_1 * z_t_1 + alpha_bar_t * lamb_t) / alpha_bar_t_1
        # Evaluate smoothness
        lhs = phi(lamb_t_1) - phi(mu_t_1) - jnp.dot(grad_phi_mu_t_1, lamb_t_1 - mu_t_1)
        rhs = 0.5 * M * jnp.linalg.norm(lamb_t_1 - mu_t_1, ord=jnp.inf) ** 2
        return n_iter + 1, lhs, rhs, M, z_t, z_t_1, lamb_t, lamb_t_1, alpha_bar_t_1, alpha_bar_t, alpha_t_1

    def criterion(inps):
        n_iter, error, *_ = inps
        return (n_iter < iter_max) & (error > tol)

    def iter(inps):
        n_iter, error, x, alpha_bar, L, z, Lambda = inps
        M = L / 2
        alpha_bar_t = alpha_bar
        z_t = z
        lamb_t = Lambda
        x_t = x

        # Line search
        inps = (0, jnp.inf, 0, M, z_t, z_t, lamb_t, lamb_t, alpha_bar_t, alpha_bar_t, alpha_bar_t)
        _, _, _, M, _, z, _, lamb_t_1, alpha_bar_t_1, alpha_bar_t, alpha_t_1 = jax.lax.while_loop(
            linesearch_criterion, iter_linesearch, inps)
        Lambda = lamb_t_1

        L = M / 2
        alpha_bar = alpha_bar_t_1

        # Recover primal solution
        y, _z = lamb_t_1.at[:n].get(), lamb_t_1.at[n:].get()
        x_t_1 = (alpha_t_1 * phi.x_lamb(y, _z) + alpha_bar_t * x_t) / alpha_bar_t_1
        x = x_t_1

        # Evaluate the stopping condition
        X_t_1 = x_t_1.reshape((n, n))
        error = jnp.linalg.norm(r_fun(X_t_1) - r, ord=1) + jnp.linalg.norm(c_fun(X_t_1) - c, ord=1)

        return n_iter + 1, error, x, alpha_bar, L, z, Lambda

    inps = (0, jnp.inf, x, alpha_bar, L, z, Lambda)
    n_iter, _, x, *_ = jax.lax.while_loop(criterion, iter, inps)

    return x, n_iter
