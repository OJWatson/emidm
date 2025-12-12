import jax.numpy as jnp
from emidm.optim import optimize_params
import pytest

jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")


def test_optimize_params_reduces_simple_quadratic():
    def loss_fn(x):
        return (x - 3.0) ** 2

    params0 = jnp.array(0.0)
    params1, hist = optimize_params(
        loss_fn=loss_fn, init_params=params0, n_steps=50, learning_rate=0.2)

    assert hist["loss"][0] > hist["loss"][-1]
    assert jnp.abs(params1 - 3.0) < 0.5
