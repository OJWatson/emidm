import numpy as np

from emidm.safir import run_safir, run_safir_replicates, simulate_safir


def test_simulate_safir_conserves_population():
    population = [50, 70]
    contact_matrix = [[10.0, 2.0], [2.0, 8.0]]
    result = simulate_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        T=10,
        dt=0.5,
        seed=0,
        n_seed=5,
    )
    totals = result["S"] + result["E"] + \
        result["I"] + result["R"] + result["D"]
    assert np.all(totals == sum(map(int, population)))


def test_run_safir_conserves_population_aggregate():
    population = [40, 60]
    contact_matrix = [[5.0, 1.0], [1.0, 4.0]]
    result = run_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=1.5,
        T=8,
        dt=0.5,
        seed=0,
        n_seed=5,
    )
    totals = result["S"] + result["E"] + \
        result["I"] + result["R"] + result["D"]
    assert np.all(totals == sum(map(int, population)))


def test_simulate_safir_is_reproducible_with_seed():
    population = [30, 30]
    contact_matrix = [[8.0, 2.0], [2.0, 6.0]]
    result1 = simulate_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        T=6,
        dt=0.5,
        seed=123,
        n_seed=4,
    )
    result2 = simulate_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        T=6,
        dt=0.5,
        seed=123,
        n_seed=4,
    )
    np.testing.assert_array_equal(result1["S"], result2["S"])
    np.testing.assert_array_equal(result1["I"], result2["I"])


def test_run_safir_replicates_shape():
    population = [40, 60]
    contact_matrix = [[5.0, 1.0], [1.0, 4.0]]
    result = run_safir_replicates(
        population=population,
        contact_matrix=contact_matrix,
        R0=1.5,
        T=8,
        dt=0.5,
        seed=0,
        n_seed=5,
        reps=3,
    )
    assert result["S"].shape == (3, 9)
    assert result["I"].shape == (3, 9)
    assert result["t"].shape == (9,)
