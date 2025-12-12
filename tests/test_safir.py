import pandas as pd

from emidm.safir import run_safir, simulate_safir


def test_simulate_safir_conserves_population_by_age():
    population = [50, 70]
    contact_matrix = [[10.0, 2.0], [2.0, 8.0]]
    df = simulate_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        time_horizon=10,
        dt=0.5,
        seed=0,
        n_seed=5,
    )

    df = df.assign(total=df["S"] + df["E1"] + df["E2"] +
                   df["Iasy"] + df["Imild"] + df["Icase"] + df["R"] + df["D"])
    pop_int = pd.Series(population).astype(int)
    for age in pop_int.index:
        assert (df.loc[df["age"] == age, "total"] == pop_int.loc[age]).all()


def test_run_safir_conserves_population_aggregate():
    population = [40, 60]
    contact_matrix = [[5.0, 1.0], [1.0, 4.0]]
    df = run_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=1.5,
        time_horizon=8,
        dt=0.5,
        seed=0,
        n_seed=5,
    )
    totals = df["S"] + df["E"] + df["I"] + df["R"] + df["D"]
    assert (totals == sum(map(int, population))).all()


def test_simulate_safir_is_reproducible_with_seed():
    population = [30, 30]
    contact_matrix = [[8.0, 2.0], [2.0, 6.0]]
    df1 = simulate_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        time_horizon=6,
        dt=0.5,
        seed=123,
        n_seed=4,
    )
    df2 = simulate_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        time_horizon=6,
        dt=0.5,
        seed=123,
        n_seed=4,
    )
    pd.testing.assert_frame_equal(df1, df2)
