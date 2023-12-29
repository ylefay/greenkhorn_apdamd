import jax.numpy as jnp
from ot.ot import Round, penality
from ot.apdamd.apdamd import apdamd


def OT(X, C, r, c, eps, phi=None, bregman=None, delta=None, x_fun=None, z_fun=None, iter_max=100000):
    """
    Algorithm 4. in Tianyi Lin, Nhat Ho, Michael I. Jordan (2019)
    """
    n = C.shape[0]
    if X is None:
        X = jnp.ones((n, n)) / n ** 2
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    b = jnp.hstack([r_tilde, c_tilde])
    vecX = jnp.reshape(X, (n ** 2,), order='F')
    vecC = jnp.reshape(C, -1, order='F')
    A = jnp.hstack([X @ jnp.ones(n, ), X.T @ jnp.ones(n, )]).reshape(2 * n, 1) @ vecX.T.reshape(1, n ** 2) * 1 / (
            vecX @ vecX.T)

    def f(x):  # x = vecX
        return vecC.T @ x - penality(x, eta)

    if phi is None:
        # Fallback to phi(z) = 1/(2n) ||z||_2^2
        # Thus, B_phi(z, z') = 1/ n ||z - z'||_2^2
        # And, first-order condition for z is z = n alpha grad(phi)(mu) + z' = - alpha * mu + z'

        def phi(x):
            return 1 / (2 * n) * jnp.linalg.norm(x, ord=2) ** 2

        def z_fun(z, mu, alpha):
            return - alpha * mu + z

        def bregman(z, z_p):
            return 1 / n * jnp.linalg.norm(z - z_p, ord=2) ** 2

        def x_fun(Lambda, x):  # argmin_x f(x) + A.T @ Lambda @ x
            return jnp.exp(-(vecC.T + A.T @ Lambda) / eta)

        if delta is None:
            delta = n

    X_tilde, n_iter = apdamd(f, bregman, phi, vecX, A, b, eps_p / 2, delta=delta, x_fun=x_fun, z_fun=z_fun,
                             iter_max=iter_max)
    X_hat = Round(X_tilde, r, c)
    return X_hat, n_iter
