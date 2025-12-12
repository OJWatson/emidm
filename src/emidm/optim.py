from __future__ import annotations


def optimize_params(*, loss_fn, init_params, n_steps: int = 200, learning_rate: float = 1e-2):
    """Generic gradient-based optimizer using Optax.

    This is intentionally lightweight: the user supplies a differentiable `loss_fn(params)`.

    Returns `(params, history)` where history is a dict with `loss` array.
    """
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "optimize_params requires 'jax' and 'optax'. Install them first (e.g. `pip install optax`)."
        ) from e

    params = init_params

    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for _ in range(int(n_steps)):
        params, opt_state, loss = step(params, opt_state)
        losses.append(loss)

    history = {"loss": jnp.stack(losses)}
    return params, history
