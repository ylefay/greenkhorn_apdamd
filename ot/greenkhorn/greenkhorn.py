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
    return jnp.linalg.norm(r_fun(Buv) - r, order=1) + jnp.linalg.norm(c_fun(Buv) - c, order=1)


def greenkhorn(C, eta, r, c, tol):
    """
    Algorithm 1.
    """
    n = C.shape[0]
    exp_minus_Cij_over_eta = jnp.exp(-C / eta)
    Buv = exp_minus_Cij_over_eta
    rBuv = r_fun(Buv)
    cBuv = c_fun(Buv)
    Buv = exp_minus_Cij_over_eta

    def rho_r(b):
        return jax.vmap(rho, in_axes=(0, 0))(r, b)

    def criterion(inps):
        _, _, _, rBuv, Cbuv, _ = inps
        return jnp.linalg.norm(rBuv - r, order=1) + jnp.linalg.norm(cBuv - c, order=1) > tol

    def iter(inps):
        n_iter, u, v, rBuv, cBuv, Buv = inps
        I = jnp.argmax(rho_r(rBuv))
        J = jnp.argmax(rho(cBuv))
        rhoRbuv = rho(rBuv.at[I].get())
        rhoCbuv = rho(cBuv.at[J].get())
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


def Round(F, U):
    """
    Algorithm 2. in Altschuler, Weed, Rigollet (2017)
    """
    raise NotImplementedError


def OT(C, r, c, eps):
    n = C.shape[0]
    eta = eps / (4 * jnp.log(n))
    eps_p = eps / (8 * jnp.linalg.norm(C, order='inf'))
    r_tilde = (1 - eps_p / 8) * r + eps_p / (8 * n) * jnp.ones(n, )
    c_tilde = (1 - eps_p / 8) * c + eps_p / (8 * n) * jnp.ones(n, )
    X_tilde = greenkhorn(C, eta, r_tilde, c_tilde, eps_p / 2)
    X_hat = Round(C, X_tilde)
    return X_hat
