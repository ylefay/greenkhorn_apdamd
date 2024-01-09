import jax.lax
import jax.numpy as jnp
from optimal_transport.ot import r_fun, c_fun

jax.config.update("jax_enable_x64", True)


def rho(a, b):
    return b - a + a * jnp.log(a / b)


def greenkhorn(X, C, eta, r, c, tol, iter_max):
    """
    Algorithm 1. in Lin, Tianyi and Ho, Nhat and Jordan, Michael I. (2019)
    """
    n = C.shape[0]
    if X is None:
        u, v = - jnp.log(n) * jnp.ones(n), - jnp.log(n) * jnp.ones(n)
        Buv = jnp.diag(jnp.exp(u)) @ jnp.exp(-C / eta) @ jnp.diag(jnp.exp(v))
    else:
        Buv = X
        _ = X * jnp.exp(C / eta)
        u = jnp.log(jnp.sum(_, axis=1))
        v = jnp.log(jnp.sum(_, axis=0))
    rBuv = r_fun(Buv)
    cBuv = c_fun(Buv)

    def criterion(inps):
        n_iter, _, _, rBuv, cBuv, Buv = inps
        return ((jnp.linalg.norm(rBuv - r, ord=1) + jnp.linalg.norm(cBuv - c, ord=1)) > tol) & (n_iter < iter_max)

    def iter(inps):
        n_iter, u, v, rBuv, cBuv, Buv = inps
        I = jnp.argmax(rho(r, rBuv))
        J = jnp.argmax(rho(r, cBuv))
        rhoRbuv = rho(r.at[I].get(), rBuv.at[I].get())
        rhoCbuv = rho(r.at[J].get(), cBuv.at[J].get())
        u, Buv = jax.lax.cond(rhoRbuv > rhoCbuv,
                              lambda _: (
                                  u.at[I].set(u.at[I].get() + jnp.log(r.at[I].get()) - jnp.log(rBuv.at[I].get())),
                                  Buv.at[I].set(Buv.at[I].get() * r.at[I].get() / rBuv.at[I].get())
                              ),
                              lambda _: (u, Buv), None)
        v, Buv = jax.lax.cond(rhoRbuv > rhoCbuv,
                              lambda _: (v, Buv),
                              lambda _: (
                                  v.at[J].set(v.at[J].get() + jnp.log(c.at[J].get()) - jnp.log(cBuv.at[J].get())),
                                  Buv.at[:, J].set(Buv.at[:, J].get() * c.at[J].get() / cBuv.at[J].get())
                              ),
                              None)
        rBuv = r_fun(Buv)
        cBuv = c_fun(Buv)
        return n_iter + 1, u, v, rBuv, cBuv, Buv

    inps = (0, u, v, rBuv, cBuv, Buv)
    n_iter, _, _, _, _, Buv = jax.lax.while_loop(criterion, iter, inps)
    return Buv, n_iter
