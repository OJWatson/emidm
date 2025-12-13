"""Model registry for emidm.

This module provides a registry system for epidemiological models, enabling
easy discovery, registration, and instantiation of models. This makes it
straightforward for contributors to add new models to the package.

Example usage:

    # Register a new model
    @register_model("my_seirs", category="stochastic")
    def run_my_seirs(*, N, I0, beta, gamma, sigma, T, seed=None):
        ...

    # List available models
    list_models()  # Returns all registered models
    list_models(category="stochastic")  # Filter by category

    # Get a model function
    model_fn = get_model("sir")
    result = model_fn(N=1000, I0=10, beta=0.3, gamma=0.1, T=100)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about a registered model.

    Attributes
    ----------
    name : str
        Unique identifier for the model.
    func : Callable
        The model function.
    category : str
        Category of the model (e.g., "stochastic", "differentiable").
    description : str
        Brief description of the model.
    compartments : list[str]
        List of compartment names returned by the model.
    parameters : list[str]
        List of required parameter names.
    has_replicates : bool
        Whether a replicate version exists.
    """

    name: str
    func: Callable
    category: str
    description: str = ""
    compartments: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    has_replicates: bool = False
    replicate_func: Optional[Callable] = None


# Global registry
_MODEL_REGISTRY: Dict[str, ModelInfo] = {}


def register_model(
    name: str,
    *,
    category: str = "stochastic",
    description: str = "",
    compartments: List[str] | None = None,
    parameters: List[str] | None = None,
    replicate_func: Callable | None = None,
):
    """Decorator to register a model function.

    Parameters
    ----------
    name : str
        Unique identifier for the model (e.g., "sir", "safir", "diff_sir").
    category : str, default "stochastic"
        Category: "stochastic", "differentiable", or custom.
    description : str, optional
        Brief description of the model.
    compartments : list[str], optional
        Compartment names (e.g., ["S", "I", "R"]). Auto-detected if not provided.
    parameters : list[str], optional
        Required parameter names. Auto-detected from function signature if not provided.
    replicate_func : Callable, optional
        Function for running multiple replicates.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @register_model("my_sir", category="stochastic", compartments=["S", "I", "R"])
    ... def run_my_sir(*, N, I0, beta, gamma, T, seed=None):
    ...     # Implementation
    ...     pass
    """

    def decorator(func: Callable) -> Callable:
        import inspect

        # Auto-detect parameters from function signature
        params = parameters
        if params is None:
            sig = inspect.signature(func)
            params = [
                p.name
                for p in sig.parameters.values()
                if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]

        # Default compartments based on category
        comps = compartments
        if comps is None:
            if "safir" in name.lower() or "seir" in name.lower():
                comps = ["S", "E", "I", "R", "D"]
            else:
                comps = ["S", "I", "R"]

        info = ModelInfo(
            name=name,
            func=func,
            category=category,
            description=description or func.__doc__ or "",
            compartments=comps,
            parameters=params,
            has_replicates=replicate_func is not None,
            replicate_func=replicate_func,
        )

        _MODEL_REGISTRY[name] = info
        return func

    return decorator


def get_model(name: str) -> Callable:
    """Get a model function by name.

    Parameters
    ----------
    name : str
        Model identifier.

    Returns
    -------
    Callable
        The model function.

    Raises
    ------
    KeyError
        If model is not found.

    Examples
    --------
    >>> model_fn = get_model("sir")
    >>> result = model_fn(N=1000, I0=10, beta=0.3, gamma=0.1, T=100)
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found. Available models: {available}")
    return _MODEL_REGISTRY[name].func


def get_model_info(name: str) -> ModelInfo:
    """Get full information about a registered model.

    Parameters
    ----------
    name : str
        Model identifier.

    Returns
    -------
    ModelInfo
        Model information dataclass.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found. Available models: {available}")
    return _MODEL_REGISTRY[name]


def list_models(category: str | None = None) -> List[str]:
    """List all registered models.

    Parameters
    ----------
    category : str, optional
        Filter by category (e.g., "stochastic", "differentiable").

    Returns
    -------
    list[str]
        List of model names.

    Examples
    --------
    >>> list_models()
    ['sir', 'safir', 'diff_sir', 'diff_safir']
    >>> list_models(category="differentiable")
    ['diff_sir', 'diff_safir']
    """
    if category is None:
        return list(_MODEL_REGISTRY.keys())
    return [name for name, info in _MODEL_REGISTRY.items() if info.category == category]


def model_summary() -> str:
    """Get a formatted summary of all registered models.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = ["Available Models", "=" * 50]

    categories = sorted(
        set(info.category for info in _MODEL_REGISTRY.values()))

    for cat in categories:
        lines.append(f"\n{cat.title()} Models:")
        lines.append("-" * 30)
        for name, info in _MODEL_REGISTRY.items():
            if info.category == cat:
                desc = info.description.split(
                    "\n")[0][:60] if info.description else ""
                rep_marker = " [+reps]" if info.has_replicates else ""
                lines.append(f"  {name}{rep_marker}: {desc}")

    return "\n".join(lines)


def _register_builtin_models():
    """Register all built-in emidm models."""
    from . import sir, safir, diff

    # Stochastic SIR
    register_model(
        "sir",
        category="stochastic",
        description="Stochastic SIR model",
        compartments=["S", "I", "R"],
        replicate_func=sir.run_sir_replicates,
    )(sir.run_sir)

    # Stochastic SAFIR
    register_model(
        "safir",
        category="stochastic",
        description="Age-structured SAFIR/SEIR model",
        compartments=["S", "E", "I", "R", "D"],
        replicate_func=safir.run_safir_replicates,
    )(safir.run_safir)

    # Differentiable SIR
    register_model(
        "diff_sir",
        category="differentiable",
        description="Differentiable SIR model (JAX/Gumbel-Softmax)",
        compartments=["S", "I", "R"],
        replicate_func=diff.run_diff_sir_replicates,
    )(diff.run_diff_sir)

    # Differentiable SAFIR
    register_model(
        "diff_safir",
        category="differentiable",
        description="Differentiable age-structured SAFIR model (JAX)",
        compartments=["S", "E", "I", "R", "D"],
        replicate_func=diff.run_diff_safir_replicates,
    )(diff.run_diff_safir)


# Register built-in models on import
_register_builtin_models()
