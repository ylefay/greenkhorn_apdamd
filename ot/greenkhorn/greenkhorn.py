import jax.numpy as jnp


def rho(a, b):
    return b - a + a * jnp.log(a / b)


def B(u, v, C, eta):
    return jnp.diag(jnp.exp(u)) @ jnp.exp(-C / eta) @ jnp.diag(jnp.exp(v))


def r_fun(A):
    """
    Row sum
    """
    return jnp.sum(A, axis=0)


def c_fun(A):
    """
    Column sum
    """
    return jnp.sum(A, axis=1)


def error(u, v, C, eta, r, c):
    """
    Error function defined in Section 3.
    """
    Buv = B(u, v, C, eta)
    return jnp.linalg.norm(r_fun(Buv) - r, order=1) + jnp.linalg.norm(c_fun(Buv) - c, order=1)

def greenkhorn(C, eta, r, c, tol):
    """
    Algorithm 1.
    """

    def iter(n_iter, u, v):

