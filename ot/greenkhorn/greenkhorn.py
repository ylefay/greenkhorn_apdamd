import jax.lax
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
    return jnp.linalg.norm(r_fun(Buv) - r, ord=1) + jnp.linalg.norm(c_fun(Buv) - c, ord=1)


def greenkhorn(C, eta, r, c, tol):
    """
    Algorithm 1. in Lin, Tianyi and Ho, Nhat and Jordan, Michael I. (2019)
    """
    n = C.shape[0]
    exp_minus_Cij_over_eta = jnp.exp(-C / eta)
    Buv = exp_minus_Cij_over_eta
    rBuv = r_fun(Buv)
    cBuv = c_fun(Buv)

    def rho_r(b):
        return jax.vmap(rho, in_axes=(0, 0))(r, b)

    def criterion(inps):
        _, _, _, rBuv, Cbuv, _ = inps
        return jnp.linalg.norm(rBuv - r, ord=1) + jnp.linalg.norm(cBuv - c, ord=1) > tol

    def iter(inps):
        n_iter, u, v, rBuv, cBuv, Buv = inps
        I = jnp.argmax(rho_r(rBuv))
        J = jnp.argmax(rho_r(cBuv))
        rhoRbuv = rho_r(rBuv.at[I].get())
        rhoCbuv = rho_r(cBuv.at[J].get())
        u, Buv = jax.lax.cond(rhoRbuv > rhoCbuv,
                              (u.at[I].set(u.at[I].get() + jnp.log(r.at[I].get()) - jnp.log(rBuv.at[I].get())),
                               Buv.at[I].set(Buv.at[I].get() * r.at[I].get() / rBuv.at[I].get())
                               ),
                              (u, Buv), None)
        v, Buv = jax.lax.cond(rhoRbuv > rhoCbuv,
                              (v, Buv),
                              (v.at[J].set(v.at[J].get() + jnp.log(c.at[J].get()) - jnp.log(cBuv.at[J].get())),
                               Buv.at[:, J].set(Buv.at[:, J].get() * c.at[J].get() / cBuv.at[J].get())
                               ),
                              None)
        rBuv = r_fun(Buv)
        cBuv = c_fun(Buv)
        return n_iter + 1, u, v, rBuv, cBuv, Buv

    inps = (0, jnp.zeros(n), jnp.zeros(n), rBuv, cBuv, Buv)
    _, _, _, _, _, Buv = jax.lax.while_loop(criterion, iter, inps)
    return Buv


def Round(F, r, c):
    """
    Algorithm 2. in Altschuler, Weed, Rigollet (2017)
    """
    rF = r_fun(F)
    X = jnp.diag(jnp.min(r / rF, 1))
    F_p = X @ F
    cF_p = c_fun(F)
    Y = jnp.diag(jnp.min(c / cF_p, 1))
    F_pp = F_p @ Y
    err_r = r - r_fun(F_pp)
    err_c = c - c_fun(F_pp)
    return F_pp + err_r @ err_c.T / jnp.linalg.norm(err_r, ord=1)


def OT(C, r, c, eps):
    """
    Algorithm 2. in Tianyi Lin, Nhat Ho, Michael I. Jordan (2019)
    """
    n = C.shape[0]
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, ord=jnp.inf))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    X_tilde = greenkhorn(C, eta, r_tilde, c_tilde, eps_p / 2)
    X_hat = Round(C, X_tilde)
    return X_hat
