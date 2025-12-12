from emidm.diff import DiffConfig, run_diff_safir, run_diff_safir_simple, run_diff_sir
import pytest

jax = pytest.importorskip("jax")


def test_run_diff_sir_shapes_and_conservation():
    out = run_diff_sir(N_agents=50, I0=5, beta=0.3, gamma=0.2,
                       T=10, config=DiffConfig(tau=0.7))
    assert out["t"].shape[0] == 11
    assert out["S"].shape == out["t"].shape
    assert out["I"].shape == out["t"].shape
    assert out["R"].shape == out["t"].shape

    totals = out["S"] + out["I"] + out["R"]
    assert jax.numpy.all(totals == 50)


def test_run_diff_safir_shapes_and_conservation():
    population = [40, 60]
    contact_matrix = [[8.0, 2.0], [2.0, 6.0]]
    out = run_diff_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        time_horizon=10,
        dt=0.5,
        seed=0,
        n_seed=4,
        tau=0.7,
    )
    totals = out["S"] + out["E"] + out["I"] + out["R"] + out["D"]
    assert jax.numpy.all(totals == sum(map(int, population)))


def test_run_diff_safir_simple_conserves_population():
    out = run_diff_safir_simple(
        N_agents=60, I0=6, beta=0.25, gamma=0.15, ifr=0.1, T=10, config=DiffConfig(tau=0.7)
    )
    totals = out["S"] + out["I"] + out["R"] + out["D"]
    assert jax.numpy.all(totals == 60)


def test_diff_model_grad_is_finite():
    import jax.numpy as jnp

    def loss_fn(beta):
        out = run_diff_sir(N_agents=50, I0=5, beta=beta,
                           gamma=0.2, T=10, config=DiffConfig(tau=0.9))
        # Encourage lower peak infections
        return jnp.max(out["I"]) / 50.0

    g = jax.grad(loss_fn)(0.3)
    assert jnp.isfinite(g)
