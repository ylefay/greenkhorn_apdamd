import jax.numpy as jnp
import jax


def r_fun(A):
    """
    Row sum
    """
    return jnp.sum(A, axis=1)


def c_fun(A):
    """
    Column sum
    """
    return jnp.sum(A, axis=0)


def cost(C, X):
    return jnp.sum(C * X)


@jax.vmap
def xlogx(x):
    return jax.lax.cond(x > 0, lambda x: x * jnp.log(x), lambda x: 0., x)


def penality(X, eta):
    vecX = X.reshape(-1, order='F')
    return eta * jnp.sum(xlogx(vecX))


def penalised_cost(C, X, eta):
    return cost(C, X) + penality(X, eta)


def Round(F, r, c):
    """
    Algorithm 2. in Altschuler, Weed, Rigollet (2017)
    """
    rF = r_fun(F)
    X = jnp.diag(jax.vmap(lambda x: jnp.min(jnp.array([x, 1])))(r / rF))
    F_p = X @ F
    cF_p = c_fun(F_p)
    Y = jnp.diag(jax.vmap(lambda x: jnp.min(jnp.array([x, 1])))(c / cF_p))
    F_pp = F_p @ Y
    err_r = r - r_fun(F_pp)
    err_c = c - c_fun(F_pp)
    if jnp.linalg.norm(err_r) > 0:
        return F_pp + err_r.T @ err_c / jnp.linalg.norm(err_r, ord=1)
    return F_pp
