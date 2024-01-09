import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def bregman_divergence(phi):
    def bregman(x, y):
        return phi(x) - phi(y) - (x - y).T @ jax.grad(phi)(y)

    return bregman


def apdamd(varphi, bregman_varphi, x, A, At, b, eps_p, f, bregman_phi, phi, delta, x_fun, z_fun,
           iter_max=None):
    n = x.shape[0]
    m = b.shape[0]
    alpha_bar = 0
    alpha = 0
    L = 1
    common_value = jnp.zeros(m, )
    z, mu, Lambd = common_value, common_value, common_value

    if z_fun is None:  # Fallback to scipy optimizer for computing z_fun given phi.
        def objective(z, z_p, mu, alpha):
            return jnp.dot(jax.grad(varphi)(mu), z - mu) + bregman_phi(z, z_p) / alpha

        def z_fun(z, mu, alpha):
            return minimize(lambda _z: objective(_z, z, mu, alpha), z, method='BFGS').x

    def criterion(inps):
        n_iter, x, *_ = inps
        return ((n_iter < iter_max) & (jnp.linalg.norm(A(x) - b, ord=1) > eps_p)) | (n_iter == 0)

    def criterion_line_search(inps):
        n_iter, M, _, _, mu, _, Lambd = inps
        return ((n_iter < jnp.inf) & (
                bregman_varphi(Lambd, mu) > M / 2 * jnp.linalg.norm(Lambd - mu, ord=jnp.inf) ** 2)) | (n_iter == 0)

    if x_fun is None:  # Fallback to scipy optimizer for computing x_fun given f.
        def x_fun(Lambd, x_init):
            return minimize(lambda _x: f(_x) + At(Lambd) @ _x.T, x_init, method='BFGS').x

    def line_search_iter(inps):
        n_iter, M, alpha_bar, alpha, mu, z, Lambd = inps
        old_alpha_bar = alpha_bar
        # M = jax.lax.cond(2 * M <= 2e260, lambda M: 2 * M, lambda M: M, M)
        M = 2 * M
        alpha = (1 + jnp.sqrt(1 + 4 * delta * M * alpha_bar)) / (2 * delta * M)
        alpha_bar = alpha_bar + alpha
        mu = (alpha * z + old_alpha_bar * Lambd) / alpha_bar
        z = z_fun(z, mu, alpha)
        Lambd = (alpha * z + old_alpha_bar * Lambd) / alpha_bar
        return n_iter + 1, M, alpha_bar, alpha, mu, z, Lambd

    def iter_fun(inps):
        n_iter, x, alpha_bar, alpha, L, z, mu, Lambd = inps
        M = L / 2
        old_alpha_bar = alpha_bar
        inps = (0, M, alpha_bar, alpha, mu, z, Lambd)
        # while criterion_line_search(inps):
        #    inps = line_search_iter(inps)
        # _, M, alpha_bar, alpha, mu, z, Lambd = inps
        _, M, alpha_bar, alpha, mu, z, Lambd = jax.lax.while_loop(criterion_line_search, line_search_iter, inps)
        x = (x_fun(mu, x) * alpha + old_alpha_bar * x) / alpha_bar
        # L = jax.lax.cond(M >= 2e-298, lambda M: M / 2, lambda M: M, M.astype(jnp.float64))
        L = M / 2
        return n_iter + 1, x, alpha_bar, alpha, L, z, mu, Lambd

    inps = (0, x, alpha_bar, alpha, L, z, mu, Lambd)
    n_iter, x, *_ = jax.lax.while_loop(criterion, iter_fun, inps)
    # while criterion(inps):
    #    inps = iter_fun(inps)
    # n_iter, x, *_ = inps
    return x, n_iter
