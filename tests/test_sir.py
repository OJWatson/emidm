import numpy as np

from emidm.sir import run_model_with_replicates, run_sir, run_sir_replicates, simulate_sir
from emidm.utils import to_dataframe


def test_run_sir_conserves_population():
    result = run_sir(N=1000, I0=10, R0_init=5, T=50, seed=0)
    totals = result["S"] + result["I"] + result["R"]
    assert np.all(totals == 1000)


def test_run_sir_is_reproducible_with_seed():
    result1 = run_sir(N=1000, I0=10, R0_init=0, T=30, seed=123)
    result2 = run_sir(N=1000, I0=10, R0_init=0, T=30, seed=123)
    np.testing.assert_array_equal(result1["S"], result2["S"])
    np.testing.assert_array_equal(result1["I"], result2["I"])
    np.testing.assert_array_equal(result1["R"], result2["R"])


def test_run_sir_replicates_shape():
    result = run_sir_replicates(N=200, I0=5, T=10, reps=3, seed=0)
    assert result["S"].shape == (3, 11)
    assert result["I"].shape == (3, 11)
    assert result["R"].shape == (3, 11)
    assert result["t"].shape == (11,)


def test_run_model_with_replicates_has_replicate_column():
    df = run_model_with_replicates(run_sir, reps=3, N=200, I0=5, T=10, seed=0)
    assert "replicate" in df.columns
    assert set(df["replicate"].unique()) == {0, 1, 2}


def test_simulate_sir_has_expected_keys_and_replicates():
    result = simulate_sir(N=300, I0=3, R0_init=0, T=10, reps=2, seed=0)
    assert set(["t", "S", "I", "R"]).issubset(result.keys())
    assert result["S"].shape == (2, 11)


def test_simulate_sir_time_varying_rt_runs_and_conserves_population():
    R_t = [2.0] * 5 + [0.8] * 6
    result = simulate_sir(N=250, I0=5, R0_init=0,
                          gamma=0.2, R_t=R_t, T=10, seed=0)
    totals = result["S"] + result["I"] + result["R"]
    assert np.all(totals == 250)


def test_to_dataframe_single_run():
    result = run_sir(N=100, I0=5, T=10, seed=0)
    df = to_dataframe(result)
    assert "t" in df.columns
    assert "S" in df.columns
    assert len(df) == 11


def test_to_dataframe_replicates():
    result = run_sir_replicates(N=100, I0=5, T=10, reps=3, seed=0)
    df = to_dataframe(result)
    assert "t" in df.columns
    assert "replicate" in df.columns
    assert len(df) == 33  # 3 reps * 11 time points
