import jax.numpy as jnp
import jax


def sinkhorn(C, eta, a1, b1, tol, iter_max):
    """
    Sinkhorn Algorithm in a log-sum-exp style
    """
    n = C.shape[0]
    v = jnp.ones(n) / n
    u = jnp.ones(n) / n
    cost_over_reg = jnp.exp(-C / eta)

    def criterion(inps):
        n_iter, u, v, err = inps
        return (n_iter < iter_max) & (err > tol)

    def iter_fun(inps):
        n_iter, u, v, _ = inps
        # sinkhorn step 1
        denominator_u = jnp.dot(cost_over_reg, v)
        u = a1 / denominator_u
        # error computation
        _ = cost_over_reg * u.at[:, None].get()
        r = v * jnp.sum(_, axis=0)
        error = jnp.linalg.norm(r - b1, ord=1)
        # sinkhorn step 2
        denominator_v = jnp.dot(cost_over_reg.T, u)
        v = b1 / denominator_v
        # error computation
        _ = cost_over_reg * v.at[None, :].get()
        s = u * jnp.sum(_, axis=1)
        error += jnp.linalg.norm(s - a1, ord=1)
        return n_iter + 1, u, v, error

    inps = (0, u, v, jnp.inf)
    n_iter, u, v, _ = jax.lax.while_loop(criterion, iter_fun, inps)
    X = jnp.diag(u) @ cost_over_reg @ jnp.diag(v)
    return X, n_iter
