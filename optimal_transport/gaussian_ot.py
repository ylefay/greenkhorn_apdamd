import jax.numpy as jnp
import jax
from functools import partial


def Gaussian_OT(Gaussian_1, Gaussian_2):
    """
    Equation 2.40 from Computational Optimal Transport (Peyré, Cuturi, 2019)
    """
    m1, cov1 = Gaussian_1
    m2, cov2 = Gaussian_2
    sqrt_cov1 = jnp.linalg.cholesky(cov2)
    invsqrt_cov1 = jnp.linalg.inv(sqrt_cov1)
    cov = invsqrt_cov1 @ jnp.linalg.cholesky(sqrt_cov1 @ cov2 @ sqrt_cov1.T) @ invsqrt_cov1

    def T(x):
        return m2 + cov @ (x - m1)

    return T


def euclidean_wasserstein(xs, ys, order=2):
    return jnp.mean(jnp.linalg.norm(xs - ys, axis=-1, ord=order) ** order) ** (1 / order)


@partial(jax.vmap, in_axes=(0, None, None))
def pdf(x, m, cov):
    return jnp.exp(-(x - m).T @ jnp.linalg.inv(cov) @ (x - m) / 2)


def bures_metric(sigma1, sigma2):
    """
    Equation 2.42 from Computational Optimal Transport (Peyré, Cuturi, 2019)
    """
    sqrt_sigma1 = jnp.real(jax.scipy.linalg.sqrtm(sigma1))
    return jnp.sqrt(jnp.trace(sigma1 + sigma2 - 2 * jnp.real(jax.scipy.linalg.sqrtm(sqrt_sigma1 @ sigma2 @ sqrt_sigma1))))


def optimal_cost_for_gaussian(Gaussian_1, Gaussian_2):
    """
    Equation 2.41 from Computational Optimal Transport (Peyré, Cuturi, 2019)
    """
    m1, cov1 = Gaussian_1
    m2, cov2 = Gaussian_2
    return (jnp.linalg.norm(m1 - m2, ord=2) ** 2 + bures_metric(cov1, cov2) ** 2)
