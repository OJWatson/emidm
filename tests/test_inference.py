import pytest


jax = pytest.importorskip("jax")
blackjax = pytest.importorskip("blackjax")


def test_blackjax_nuts_runs_for_simple_gaussian():
    import jax.numpy as jnp

    from emidm.inference import run_blackjax_nuts

    def logdensity(x):
        # Standard normal
        return -0.5 * jnp.sum(x**2)

    samples = run_blackjax_nuts(logdensity_fn=logdensity, initial_position=jnp.array(
        [1.0, -1.0]), num_warmup=16, num_samples=16)
    assert samples.shape == (16, 2)
