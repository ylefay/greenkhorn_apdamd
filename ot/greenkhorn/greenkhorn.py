import jax.lax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def rho(a, b):
    return b - a + a * jnp.log(a / b)


def B(u, v, C, eta):
    return jnp.diag(jnp.exp(u)) @ jnp.exp(-C / eta) @ jnp.diag(jnp.exp(v))


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


def error(Buv, r, c):
    """
    Error function defined in Section 3.
    """
    return jnp.linalg.norm(r_fun(Buv) - r, ord=1) + jnp.linalg.norm(c_fun(Buv) - c, ord=1)


def greenkhorn(C, eta, r, c, tol, iter_max=100):
    """
    Algorithm 1. in Lin, Tianyi and Ho, Nhat and Jordan, Michael I. (2019)
    """
    n = C.shape[0]
    u, v = - jnp.log(n) * jnp.ones(n), - jnp.log(n) * jnp.ones(n)
    exp_minus_Cij_over_eta = jnp.exp(-C / eta)
    Buv = B(u, v, C, eta)
    rBuv = r_fun(Buv)
    cBuv = c_fun(Buv)

    def criterion(inps):
        n_iter, _, _, rBuv, Cbuv, _ = inps
        return (jnp.linalg.norm(rBuv - r, ord=1) + jnp.linalg.norm(cBuv - c, ord=1) > tol) & (n_iter < iter_max)

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
        Buv = Buv / jnp.sum(Buv)
        rBuv = r_fun(Buv)
        cBuv = c_fun(Buv)
        return n_iter + 1, u, v, rBuv, cBuv, Buv

    inps = (0, u, v, rBuv, cBuv, Buv)
    n_iter, _, _, _, _, Buv = jax.lax.while_loop(criterion, iter, inps)
    return Buv, n_iter, error(Buv, r, c)


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
    return F_pp + err_r @ err_c.T / jnp.linalg.norm(err_r, ord=1)


def OT(C, r, c, eps, iter_max=1000):
    """
    Algorithm 2. in Tianyi Lin, Nhat Ho, Michael I. Jordan (2019)
    """
    n = C.shape[0]
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    X_tilde, n_iter, err = greenkhorn(C, eta, r_tilde, c_tilde, eps_p / 2, iter_max)
    X_hat = Round(X_tilde, r, c)
    return X_hat, n_iter, eps
