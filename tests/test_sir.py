import pandas as pd

from emidm.sir import run_model_with_replicates, run_sir, simulate_sir


def test_run_sir_conserves_population():
    df = run_sir(N=1000, I0=10, R0=5, T=50, seed=0)
    totals = df["S"] + df["I"] + df["R"]
    assert (totals == 1000).all()


def test_run_sir_is_reproducible_with_seed():
    df1 = run_sir(N=1000, I0=10, R0=0, T=30, seed=123)
    df2 = run_sir(N=1000, I0=10, R0=0, T=30, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_run_model_with_replicates_has_replicate_column():
    df = run_model_with_replicates(run_sir, reps=3, N=200, I0=5, T=10, seed=0)
    assert "replicate" in df.columns
    assert set(df["replicate"].unique()) == {0, 1, 2}


def test_simulate_sir_has_expected_columns_and_replicates():
    df = simulate_sir(N=300, I0=3, R0=0, T=10, n_replicates=2, seed=0)
    assert set(["t", "replicate", "S", "I", "R"]).issubset(df.columns)
    assert set(df["replicate"].unique()) == {0, 1}


def test_simulate_sir_time_varying_rt_runs_and_conserves_population():
    R_t = [2.0] * 5 + [0.8] * 6
    df = simulate_sir(N=250, I0=5, R0=0, gamma=0.2, R_t=R_t, T=10, seed=0)
    totals = df["S"] + df["I"] + df["R"]
    assert (totals == 250).all()
