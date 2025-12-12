from __future__ import annotations


def run_blackjax_nuts(*, logdensity_fn, initial_position, rng_seed: int = 0, num_warmup: int = 64, num_samples: int = 64):
    """Minimal BlackJAX NUTS runner.

    This is a thin wrapper intended for examples/tests. It is optional and will raise if `blackjax` isn't installed.
    """
    try:
        import jax
        import blackjax
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "run_blackjax_nuts requires 'blackjax' (and jax). Install e.g. `pip install blackjax`."
        ) from e

    rng_key = jax.random.PRNGKey(rng_seed)

    nuts = blackjax.nuts(logdensity_fn)
    warmup = blackjax.window_adaptation(nuts, logdensity_fn)

    (state, params), _ = warmup.run(
        rng_key, initial_position, num_steps=num_warmup)

    def one_step(carry, rng_key):
        state, params = carry
        state, info = nuts.step(rng_key, state, params)
        return (state, params), state.position

    keys = jax.random.split(rng_key, num_samples)
    (_, _), positions = jax.lax.scan(one_step, (state, params), keys)
    return positions
