"""Bayesian inference utilities for emidm.

This module provides utilities for Bayesian inference using MCMC methods,
particularly Hamiltonian Monte Carlo via BlackJAX.
"""
from __future__ import annotations

from typing import Any, Callable


def run_blackjax_nuts(
    *,
    logdensity_fn: Callable[[Any], float],
    initial_position: Any,
    rng_seed: int = 0,
    num_warmup: int = 64,
    num_samples: int = 64,
) -> Any:
    """Run NUTS (No-U-Turn Sampler) using BlackJAX.

    This is a thin wrapper around BlackJAX's NUTS implementation with
    automatic window adaptation for step size and mass matrix tuning.

    Parameters
    ----------
    logdensity_fn : Callable
        Log-density function (unnormalized log-posterior).
        Should take a position and return a scalar.
    initial_position : array-like
        Initial parameter values for the sampler.
    rng_seed : int, default 0
        Random seed for reproducibility.
    num_warmup : int, default 64
        Number of warmup (adaptation) steps.
    num_samples : int, default 64
        Number of samples to draw after warmup.

    Returns
    -------
    array
        Array of posterior samples with shape (num_samples, ...).

    Raises
    ------
    ModuleNotFoundError
        If BlackJAX or JAX is not installed.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from emidm import run_blackjax_nuts
    >>> def log_posterior(x):
    ...     return -0.5 * jnp.sum(x ** 2)  # Standard normal
    >>> samples = run_blackjax_nuts(
    ...     logdensity_fn=log_posterior,
    ...     initial_position=jnp.zeros(2),
    ...     num_samples=100
    ... )
    """
    try:
        import jax
        import blackjax
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "run_blackjax_nuts requires 'blackjax' (and jax). Install e.g. `pip install blackjax`."
        ) from e

    rng_key = jax.random.PRNGKey(rng_seed)
    key_warmup, key_sample = jax.random.split(rng_key)

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)

    (state, params), _ = warmup.run(
        key_warmup, initial_position, num_steps=num_warmup)

    nuts = blackjax.nuts(logdensity_fn, **params)

    def one_step(state, rng_key):
        state, info = nuts.step(rng_key, state)
        return state, state.position

    sample_keys = jax.random.split(key_sample, num_samples)

    @jax.jit
    def _draw(initial_state, keys):
        _, positions = jax.lax.scan(one_step, initial_state, keys)
        return positions

    return _draw(state, sample_keys)
