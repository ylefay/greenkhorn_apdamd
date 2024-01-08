import jax.numpy as jnp
from ot.ot import Round
from ot.greenkhorn.greenkhorn import greenkhorn
import jax


def theoretical_bound_on_iter(C, r, c, eps):
    n = C.shape[0]
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    R = 1 / eta * jnp.linalg.norm(C, ord=jnp.inf) + jnp.log(n) - 2 * jnp.log(
        jnp.min(jnp.array([jnp.min(r), jnp.min(c)])))
    iter_max = 2 + 112 * n * R / eps_p
    return iter_max


def OT(X, C, r, c, eps, iter_max=100000000):
    """
    Algorithm 2. in Tianyi Lin, Nhat Ho, Michael I. Jordan (2019)
    """
    n = C.shape[0]
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    if iter_max is None:
        iter_max = theoretical_bound_on_iter(C, r, c, eps)
        iter_max = jnp.min(jnp.array([100000000, jnp.int64(iter_max)]))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    X_tilde, n_iter = greenkhorn(X, C, eta, r_tilde, c_tilde, eps_p / 2, iter_max)
    X_hat = Round(X_tilde, r, c)
    return X_hat, n_iter
