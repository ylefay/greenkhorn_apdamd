import jax.numpy as jnp
from ot.ot import Round
from ot.greenkhorn.greenkhorn import greenkhorn


def OT(X, C, r, c, eps, iter_max=100000):
    """
    Algorithm 2. in Tianyi Lin, Nhat Ho, Michael I. Jordan (2019)
    """
    n = C.shape[0]
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    X_tilde, n_iter = greenkhorn(X, C, eta, r_tilde, c_tilde, eps_p / 2, iter_max)
    X_hat = Round(X_tilde, r, c)
    return X_hat, n_iter
