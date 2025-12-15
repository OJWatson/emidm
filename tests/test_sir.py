import numpy as np

from emidm.sir import SIRModel, SIRState, run_sir_simulation
from emidm.utils import to_dataframe


def test_run_sir_simulation_conserves_population():
    result = run_sir_simulation(N=1000, I0=10, R_init=5, T=50, seed=0)
    totals = result["S"] + result["I"] + result["R"]
    assert np.all(totals == 1000)


def test_run_sir_simulation_is_reproducible_with_seed():
    result1 = run_sir_simulation(N=1000, I0=10, R_init=0, T=30, seed=123)
    result2 = run_sir_simulation(N=1000, I0=10, R_init=0, T=30, seed=123)
    np.testing.assert_array_equal(result1["S"], result2["S"])
    np.testing.assert_array_equal(result1["I"], result2["I"])
    np.testing.assert_array_equal(result1["R"], result2["R"])


def test_run_sir_simulation_replicates_shape():
    result = run_sir_simulation(N=200, I0=5, T=10, reps=3, seed=0)
    assert result["S"].shape == (3, 11)
    assert result["I"].shape == (3, 11)
    assert result["R"].shape == (3, 11)
    assert result["t"].shape == (11,)


def test_run_sir_simulation_has_expected_keys():
    result = run_sir_simulation(N=300, I0=3, R_init=0, T=10, reps=2, seed=0)
    assert set(["t", "S", "I", "R"]).issubset(result.keys())
    assert result["S"].shape == (2, 11)


def test_run_sir_simulation_time_varying_rt():
    R_t = [2.0] * 5 + [0.8] * 6
    result = run_sir_simulation(N=250, I0=5, R_init=0,
                                gamma=0.2, R_t=R_t, T=10, seed=0)
    totals = result["S"] + result["I"] + result["R"]
    assert np.all(totals == 250)


def test_to_dataframe_single_run():
    result = run_sir_simulation(N=100, I0=5, T=10, seed=0)
    df = to_dataframe(result)
    assert "t" in df.columns
    assert "S" in df.columns
    assert len(df) == 11


def test_to_dataframe_replicates():
    result = run_sir_simulation(N=100, I0=5, T=10, reps=3, seed=0)
    df = to_dataframe(result)
    assert "t" in df.columns
    assert "replicate" in df.columns
    assert len(df) == 33  # 3 reps * 11 time points


def test_sir_model_class_init_state():
    """Test SIRModel.init_state() returns correct initial state."""
    model = SIRModel(N=1000, I0=10, R_init=5, seed=42)
    state = model.init_state()

    assert isinstance(state, SIRState)
    assert state.S == 1000 - 10 - 5
    assert state.I == 10
    assert state.R == 5


def test_sir_model_class_step():
    """Test SIRModel.step() advances state correctly."""
    model = SIRModel(N=1000, I0=10, R_init=0, beta=0.3, gamma=0.1, seed=42)
    state = model.init_state()

    # Step once
    new_state = model.step(state, t_idx=0)

    # Population should be conserved
    assert state.S + state.I + state.R == new_state.S + \
        new_state.I + new_state.R == 1000


def test_sir_model_class_run():
    """Test SIRModel.run() produces correct output."""
    model = SIRModel(N=1000, I0=10, beta=0.3, gamma=0.1, T=50, seed=42)
    result = model.run(reps=3)

    assert result["S"].shape == (3, 51)
    assert result["I"].shape == (3, 51)
    assert result["R"].shape == (3, 51)

    # Check population conservation
    totals = result["S"] + result["I"] + result["R"]
    assert np.all(totals == 1000)
