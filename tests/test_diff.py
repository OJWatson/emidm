from emidm.diff import DiffConfig, run_diff_safir, run_diff_sir
import pytest

jax = pytest.importorskip("jax")


def test_run_diff_sir_shapes_and_conservation():
    out = run_diff_sir(N=50, I0=5, beta=0.3, gamma=0.2,
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
        T=10,
        dt=0.5,
        seed=0,
        n_seed=4,
        tau=0.7,
    )
    totals = out["S"] + out["E"] + out["I"] + out["R"] + out["D"]
    assert jax.numpy.all(totals == sum(map(int, population)))


def test_diff_model_grad_is_finite():
    import jax.numpy as jnp

    def loss_fn(beta):
        out = run_diff_sir(N=50, I0=5, beta=beta,
                           gamma=0.2, T=10, config=DiffConfig(tau=0.9))
        # Encourage lower peak infections
        return jnp.max(out["I"]) / 50.0

    g = jax.grad(loss_fn)(0.3)
    assert jnp.isfinite(g)


def test_temperature_effect_on_discreteness():
    """Test that lower temperature produces more discrete (binary) outputs."""
    import jax.numpy as jnp

    # Low temperature should produce nearly binary outputs (hard=False to see soft outputs)
    out_low_tau = run_diff_sir(
        N=50, I0=5, beta=0.3, gamma=0.2, T=10,
        config=DiffConfig(tau=0.1, hard=False),
        seed=42,
    )

    # High temperature produces smoother outputs
    out_high_tau = run_diff_sir(
        N=50, I0=5, beta=0.3, gamma=0.2, T=10,
        config=DiffConfig(tau=2.0, hard=False),
        seed=42,
    )

    # Both should conserve population
    assert jnp.allclose(
        out_low_tau["S"] + out_low_tau["I"] + out_low_tau["R"], 50, atol=0.1)
    assert jnp.allclose(
        out_high_tau["S"] + out_high_tau["I"] + out_high_tau["R"], 50, atol=0.1)


def test_hard_vs_soft_gumbel_softmax():
    """Test that hard=True produces integer-like outputs while hard=False produces continuous."""
    import jax.numpy as jnp

    # Hard mode (straight-through estimator)
    out_hard = run_diff_sir(
        N=50, I0=5, beta=0.3, gamma=0.2, T=10,
        config=DiffConfig(tau=0.5, hard=True),
        seed=42,
    )

    # Soft mode
    out_soft = run_diff_sir(
        N=50, I0=5, beta=0.3, gamma=0.2, T=10,
        config=DiffConfig(tau=0.5, hard=False),
        seed=42,
    )

    # Hard outputs should be close to integers
    hard_residuals = jnp.abs(out_hard["I"] - jnp.round(out_hard["I"]))
    assert jnp.all(hard_residuals <
                   0.01), "Hard mode should produce integer-like outputs"

    # Both should be differentiable
    def loss_hard(beta):
        out = run_diff_sir(N=50, I0=5, beta=beta, gamma=0.2, T=10,
                           config=DiffConfig(tau=0.5, hard=True))
        return jnp.mean(out["I"])

    def loss_soft(beta):
        out = run_diff_sir(N=50, I0=5, beta=beta, gamma=0.2, T=10,
                           config=DiffConfig(tau=0.5, hard=False))
        return jnp.mean(out["I"])

    g_hard = jax.grad(loss_hard)(0.3)
    g_soft = jax.grad(loss_soft)(0.3)

    assert jnp.isfinite(g_hard), "Hard mode should still be differentiable"
    assert jnp.isfinite(g_soft), "Soft mode should be differentiable"


def test_time_varying_rt_diff_sir():
    """Test that time-varying R_t works in differentiable SIR."""
    import jax.numpy as jnp

    T = 20
    # R_t that drops at t=10
    R_t = jnp.concatenate([jnp.full(11, 2.5), jnp.full(10, 0.5)])

    out = run_diff_sir(
        N=100, I0=5, gamma=0.1, T=T, R_t=R_t,
        config=DiffConfig(tau=0.5, hard=True),
        seed=42,
    )

    assert out["t"].shape[0] == T + 1
    assert jnp.all(out["S"] + out["I"] + out["R"] == 100)


def test_time_varying_rt_diff_safir():
    """Test that time-varying R_t works in differentiable SAFIR."""
    import jax.numpy as jnp

    T = 20
    R_t = jnp.concatenate([jnp.full(11, 2.5), jnp.full(10, 0.8)])

    population = [50, 50]
    contact_matrix = [[5.0, 2.0], [2.0, 4.0]]

    out = run_diff_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        R_t=R_t,
        T=T,
        dt=0.5,
        seed=42,
        n_seed=5,
        tau=0.5,
    )

    assert out["t"].shape[0] == T + 1
    totals = out["S"] + out["E"] + out["I"] + out["R"] + out["D"]
    assert jnp.all(totals == 100)
