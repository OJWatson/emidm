import numpy as np

from emidm.safir import SAFIRModel, SAFIRState, run_safir_simulation


def test_run_safir_simulation_conserves_population():
    population = [50, 70]
    contact_matrix = [[10.0, 2.0], [2.0, 8.0]]
    result = run_safir_simulation(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        T=10,
        dt=0.5,
        seed=0,
        I0=5,
    )
    totals = result["S"] + result["E"] + \
        result["I"] + result["R"] + result["D"]
    assert np.all(totals == sum(map(int, population)))


def test_run_safir_simulation_is_reproducible_with_seed():
    population = [30, 30]
    contact_matrix = [[8.0, 2.0], [2.0, 6.0]]
    result1 = run_safir_simulation(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        T=6,
        dt=0.5,
        seed=123,
        I0=4,
    )
    result2 = run_safir_simulation(
        population=population,
        contact_matrix=contact_matrix,
        R0=2.0,
        T=6,
        dt=0.5,
        seed=123,
        I0=4,
    )
    np.testing.assert_array_equal(result1["S"], result2["S"])
    np.testing.assert_array_equal(result1["I"], result2["I"])


def test_run_safir_simulation_replicates_shape():
    population = [40, 60]
    contact_matrix = [[5.0, 1.0], [1.0, 4.0]]
    result = run_safir_simulation(
        population=population,
        contact_matrix=contact_matrix,
        R0=1.5,
        T=8,
        dt=0.5,
        seed=0,
        I0=5,
        reps=3,
    )
    assert result["S"].shape == (3, 9)
    assert result["I"].shape == (3, 9)
    assert result["t"].shape == (9,)


def test_safir_model_class_init_state():
    """Test SAFIRModel.init_state() returns correct initial state."""
    population = np.array([500, 500])
    contact_matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    model = SAFIRModel(population=population,
                       contact_matrix=contact_matrix, I0=10, seed=42)
    state = model.init_state()

    assert isinstance(state, SAFIRState)
    assert state.S.sum() + state.E1.sum() == 1000  # Initial infections in E1
    assert state.E1.sum() == 10  # I0 individuals seeded


def test_safir_model_class_step():
    """Test SAFIRModel.step() advances state correctly."""
    population = np.array([500, 500])
    contact_matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    model = SAFIRModel(population=population,
                       contact_matrix=contact_matrix, I0=10, seed=42)
    state = model.init_state()

    # Step once
    new_state = model.step(state, day=0)

    # Population should be conserved
    total_old = state.S.sum() + state.E1.sum() + state.E2.sum() + state.Iasy.sum() + \
        state.Imild.sum() + state.Icase.sum() + state.R.sum() + state.D.sum()
    total_new = new_state.S.sum() + new_state.E1.sum() + new_state.E2.sum() + new_state.Iasy.sum() + \
        new_state.Imild.sum() + new_state.Icase.sum() + \
        new_state.R.sum() + new_state.D.sum()
    assert total_old == total_new == 1000


def test_safir_model_class_run():
    """Test SAFIRModel.run() produces correct output shape."""
    population = np.array([500, 500])
    contact_matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    model = SAFIRModel(population=population,
                       contact_matrix=contact_matrix, T=20, seed=42)
    result = model.run(reps=3)

    assert result["S"].shape == (3, 21)
    assert result["E"].shape == (3, 21)
    assert result["I"].shape == (3, 21)
    assert result["R"].shape == (3, 21)
    assert result["D"].shape == (3, 21)

    # Check population conservation
    totals = result["S"] + result["E"] + \
        result["I"] + result["R"] + result["D"]
    assert np.all(totals == 1000)
