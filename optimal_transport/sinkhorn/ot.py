import jax.numpy as jnp
from optimal_transport.ot import Round
from optimal_transport.sinkhorn.sinkhorn import sinkhorn
from optimal_transport.greenkhorn.ot import theoretical_bound_on_iter
import jax

jax.config.update("jax_enable_x64", True)


def OT(_, C, r, c, eps, iter_max=10e10):
    """
    Algorithm 2. in Tianyi Lin, Nhat Ho, Michael I. Jordan (2019)
    """
    n = C.shape[0]
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    if iter_max is None:
        iter_max = theoretical_bound_on_iter(C, r_tilde, c_tilde, eps_p / 2)
        iter_max = jnp.min(jnp.array([100000000, jnp.int64(iter_max)]))
    X_tilde, n_iter = sinkhorn(C, eta, r_tilde, c_tilde, eps_p / 2, iter_max)
    X_hat = Round(X_tilde, r, c)
    return X_hat, n_iter
