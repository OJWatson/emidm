"""Tests for plotting functions."""
import numpy as np
import pandas as pd
import pytest


def test_plot_sir_with_dict():
    """Test plot_sir accepts dict input."""
    from emidm import run_sir
    from emidm.plotting import plot_sir

    result = run_sir(N=100, I0=5, T=20, seed=0)
    fig, ax = plot_sir(result)
    assert fig is not None
    assert ax is not None


def test_plot_sir_with_dataframe():
    """Test plot_sir accepts DataFrame input."""
    from emidm import run_sir, to_dataframe
    from emidm.plotting import plot_sir

    result = run_sir(N=100, I0=5, T=20, seed=0)
    df = to_dataframe(result)
    fig, ax = plot_sir(df)
    assert fig is not None
    assert ax is not None


def test_plot_safir_with_dict():
    """Test plot_safir accepts dict input."""
    from emidm import run_safir
    from emidm.plotting import plot_safir

    population = np.array([500, 500])
    contact_matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    result = run_safir(population=population,
                       contact_matrix=contact_matrix, T=20, seed=0)
    fig, ax = plot_safir(result)
    assert fig is not None
    assert ax is not None


def test_plot_replicates():
    """Test plot_replicates with replicate data."""
    from emidm import run_model_with_replicates, run_sir
    from emidm.plotting import plot_replicates

    df = run_model_with_replicates(model=run_sir, reps=3, N=100, I0=5, T=20)
    fig, ax = plot_replicates(df, compartment="I")
    assert fig is not None
    assert ax is not None


def test_plot_optimization_history():
    """Test plot_optimization_history with mock history."""
    from emidm.plotting import plot_optimization_history
    import numpy as np

    history = {
        "loss": np.array([1.0, 0.5, 0.3, 0.2, 0.15]),
        "params": np.array([0.1, 0.2, 0.25, 0.28, 0.3]),
    }
    fig, axes = plot_optimization_history(history)
    assert fig is not None
    assert len(axes) == 2


def test_plot_training_histories():
    """Test plot_training_histories with mock histories."""
    from emidm.plotting import plot_training_histories

    histories = {
        "Model1": {
            "epochs": list(range(1, 11)),
            "train_loss": [1.0 - i * 0.08 for i in range(10)],
            "val_loss": [1.1 - i * 0.07 for i in range(10)],
            "best_epoch": 8,
        },
    }
    fig, axes = plot_training_histories(histories)
    assert fig is not None
    assert len(axes) == 1


def test_to_dataframe_edge_cases():
    """Test to_dataframe with edge cases."""
    from emidm import to_dataframe

    # Single time point
    result = {"t": np.array([0]), "S": np.array(
        [100]), "I": np.array([0]), "R": np.array([0])}
    df = to_dataframe(result)
    assert len(df) == 1
    assert set(df.columns) == {"t", "S", "I", "R"}

    # Empty arrays should still work
    result = {"t": np.array([]), "S": np.array(
        []), "I": np.array([]), "R": np.array([])}
    df = to_dataframe(result)
    assert len(df) == 0


def test_to_dataframe_with_replicates():
    """Test to_dataframe correctly handles replicate data."""
    from emidm import to_dataframe
    import numpy as np

    # 2 replicates, 5 time points
    result = {
        "t": np.arange(5),
        "S": np.array([[100, 95, 90, 85, 80], [100, 94, 88, 82, 76]]),
        "I": np.array([[0, 5, 10, 15, 20], [0, 6, 12, 18, 24]]),
        "R": np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    }
    df = to_dataframe(result)

    # Should have 10 rows (2 reps * 5 time points)
    assert len(df) == 10
    assert "replicate" in df.columns
    assert df["replicate"].nunique() == 2


@pytest.mark.skipif(
    not pytest.importorskip("plotnine", reason="plotnine not installed"),
    reason="plotnine not installed"
)
def test_sir_facet_plot():
    """Test sir_facet_plot creates a plotnine plot."""
    from emidm import run_model_with_replicates, run_sir
    from emidm.plotting import sir_facet_plot
    from emidm.sampler import generate_lhs_samples
    import pandas as pd

    # Generate small sample
    samples = generate_lhs_samples(
        {"beta": [0.1, 0.3], "gamma": [0.05, 0.15]}, n_samples=2, seed=0)
    results = [
        run_model_with_replicates(
            model=run_sir, reps=2, N=100, T=10, **row.to_dict()).assign(**row.to_dict())
        for _, row in samples.iterrows()
    ]
    df = pd.concat(results)

    plot = sir_facet_plot(df, facet_by=["beta"])
    assert plot is not None
