"""Tests for the model registry."""
import pytest


def test_list_models_returns_builtin_models():
    from emidm import list_models

    models = list_models()
    assert "sir" in models
    assert "safir" in models
    assert "diff_sir" in models
    assert "diff_safir" in models


def test_list_models_filters_by_category():
    from emidm import list_models

    stochastic = list_models(category="stochastic")
    assert "sir" in stochastic
    assert "safir" in stochastic
    assert "diff_sir" not in stochastic

    differentiable = list_models(category="differentiable")
    assert "diff_sir" in differentiable
    assert "diff_safir" in differentiable
    assert "sir" not in differentiable


def test_get_model_returns_callable():
    from emidm import get_model

    sir_fn = get_model("sir")
    assert callable(sir_fn)

    # Should be able to call it
    result = sir_fn(N=100, I0=5, T=10, seed=0)
    assert "t" in result
    assert "S" in result
    assert "I" in result
    assert "R" in result


def test_get_model_raises_for_unknown():
    from emidm import get_model

    with pytest.raises(KeyError, match="not found"):
        get_model("nonexistent_model")


def test_get_model_info_returns_model_info():
    from emidm import get_model_info

    info = get_model_info("sir")
    assert info.name == "sir"
    assert info.category == "stochastic"
    assert "S" in info.compartments
    assert "I" in info.compartments
    assert "R" in info.compartments
    assert info.has_replicates is True
    assert info.replicate_func is not None


def test_model_summary_returns_string():
    from emidm import model_summary

    summary = model_summary()
    assert isinstance(summary, str)
    assert "sir" in summary.lower()
    assert "safir" in summary.lower()


def test_register_model_decorator():
    from emidm.registry import register_model, get_model, _MODEL_REGISTRY

    # Register a test model
    @register_model("test_model", category="test", compartments=["A", "B"])
    def run_test_model(*, x=1):
        return {"t": [0], "A": [x], "B": [1 - x]}

    # Should be retrievable
    assert "test_model" in _MODEL_REGISTRY
    fn = get_model("test_model")
    result = fn(x=0.5)
    assert result["A"] == [0.5]

    # Clean up
    del _MODEL_REGISTRY["test_model"]
