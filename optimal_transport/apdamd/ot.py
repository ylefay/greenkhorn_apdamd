import jax.numpy as jnp
import jax
from optimal_transport.ot import Round, penality
from optimal_transport.apdamd.apdamd import apdamd

jax.config.update("jax_enable_x64", True)


def bregman_divergence(phi):
    def bregman(x, y):
        return phi(x) - phi(y) - (x - y).T @ jax.grad(phi)(y)

    return bregman


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


def OT(X, C, r, c, eps, iter_max=100000000):
    """
    Algorithm 4. in Tianyi Lin, Nhat Ho, Michael I. Jordan (2019)
    Using phi = 1 / (2 n) ||x||^2_2.
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
        return vecC.T @ x + penality(x, eta)

    def A(x):
        X = x.reshape(n, n)
        return jnp.hstack((jnp.sum(X, axis=1), jnp.sum(X, axis=0)))

    def At(x):
        alpha = x.at[:n].get()
        beta = x.at[n:].get()
        return jnp.add(alpha.reshape(n, 1), beta).flatten()

    def phi(x):
        return 1 / (2 * n) * jnp.linalg.norm(x, ord=2) ** 2

    def bregman_phi(x, x_p):
        return 1 / (2 * n) * jnp.linalg.norm(x - x_p, ord=2) ** 2

    def x_fun(Lamb, x):  # argmin_x f(x) + <A.T @ Lambda, x>
        return jnp.exp((-vecC.T + At(Lamb)) / eta - 1)

    def varphi_tilde(Lambd):
        alpha = Lambd.at[:n].get()
        beta = Lambd.at[n:].get()
        return Lambd @ b - eta * jnp.log(
            jnp.einsum("i,j,ij->", jnp.exp(alpha / eta), jnp.exp(beta / eta), jnp.exp(-C / eta)))

    bregman_varphi_tilde = bregman_divergence(varphi_tilde)

    def z_fun(z, mu, alpha):
        # return - alpha * n * jax.grad(varphi_tilde)(mu) + z
        return - alpha * n * (b - A(x_fun(mu, None))) + z

    delta = n

    if iter_max is None:
        iter_max = theoretical_bound_on_iter(C, r, c, eps, delta)
        iter_max = jnp.min(jnp.array([1e10, jnp.int64(iter_max)]))

    x_tilde, n_iter = apdamd(varphi=varphi_tilde, bregman_varphi=bregman_varphi_tilde, x=vecX, A=A, At=At, b=b,
                             eps_p=eps_p / 2, f=f, bregman_phi=bregman_phi, phi=phi, delta=delta,
                             x_fun=x_fun, z_fun=z_fun,
                             iter_max=iter_max)
    X_tilde = jnp.reshape(x_tilde, (n, n), order='F')
    X_hat = Round(X_tilde, r, c)
    return X_hat, n_iter
