import jax.numpy as jnp
import jax


def sinkhorn(C, eta, r, c, tol, iter_max):
    """
    Sinkhorn Algorithm in a log-sum-exp style
    """
    n = C.shape[0]
    log_v = jnp.zeros(n)
    log_u = jnp.zeros(n)
    log_b1 = jnp.log(r)
    log_a1 = jnp.log(c)
    cost_over_reg = -C / eta

    def criterion(inps):
        n_iter, log_u, log_v, err = inps
        return (n_iter < iter_max) & (err > tol)

    def iter_fun(inps):
        n_iter, log_u, log_v, _ = inps

        # sinkhorn step 1
        _log_denominator_u = cost_over_reg + log_v.at[None, :].get()
        log_denominator_u = jax.scipy.special.logsumexp(_log_denominator_u, axis=1, keepdims=False)
        log_u = log_a1 - log_denominator_u
        # error computation
        _ = cost_over_reg + log_u.at[:, None].get()

        log_r = log_v + jax.scipy.special.logsumexp(_, axis=0, keepdims=False)
        error = jnp.linalg.norm(jnp.exp(log_r) - r, ord=1)
        # sinkhorn step 2
        _log_denominator_v = cost_over_reg + log_u.at[:, None].get()
        log_denominator_v = jax.scipy.special.logsumexp(_log_denominator_v, axis=0, keepdims=False)
        log_v = log_b1 - log_denominator_v
        # error computation
        _ = cost_over_reg + log_v.at[None, :].get()
        log_s = log_u + jax.scipy.special.logsumexp(_, axis=1, keepdims=False)
        error += jnp.linalg.norm(jnp.exp(log_s) - c, ord=1)
        return n_iter + 1, log_u, log_v, error




    inps = (0, log_u, log_v, jnp.inf)
    n_iter, log_u, log_v, _ = jax.lax.while_loop(criterion, iter_fun, inps)
    X = log_u.at[:, None].get() + cost_over_reg + log_v.at[None, :].get()
    X = jnp.exp(X)
    return X, n_iter

