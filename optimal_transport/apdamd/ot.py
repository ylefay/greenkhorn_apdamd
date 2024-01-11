import jax.numpy as jnp
import jax
from optimal_transport.ot import Round
from optimal_transport.apdamd.apdamd import apdamd

jax.config.update("jax_enable_x64", True)


def theoretical_bound_on_iter(C, r, c, eps, delta=None):
    n = C.shape[0]
    if delta is None:
        delta = n
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    R = 1 / eta * jnp.linalg.norm(C, ord=jnp.inf) + jnp.log(n) - 2 * jnp.log(
        jnp.min(jnp.array([jnp.min(r), jnp.min(c)])))
    iter_max = 1 + jnp.sqrt(128 * delta * R / eps_p)
    return iter_max


def OT(_, C, r, c, eps, iter_max=100000000):
    """
    Algorithm 4. in Tianyi Lin, Nhat Ho, Michael I. Jordan (2019)
    Using phi = 1 / (2 n) ||x||^2_2.
    """
    n = C.shape[0]
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))  # apdamd tolerance
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    delta = n
    if iter_max is None:
        iter_max = theoretical_bound_on_iter(C, r, c, eps, delta)
        iter_max = jnp.min(jnp.array([1e10, jnp.int64(iter_max)]))

    x_tilde, n_iter = apdamd(C, r_tilde, c_tilde, eta, eps_p / 2, iter_max)

    X_hat = Round(x_tilde.reshape(n, n), r, c)
    return X_hat, n_iter
