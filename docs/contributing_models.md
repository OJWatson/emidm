# Contributing New Models to emidm

This guide explains how to add new epidemiological models to `emidm`. The package is designed to make it easy for contributors to add new disease models while maintaining a consistent interface.

## Model Interface Requirements

All models in `emidm` follow a consistent interface pattern. Your model should:

### 1. Return a Dictionary

All models return a `dict` with:
- `"t"`: Time points array (shape: `(T+1,)` for single runs)
- Compartment arrays (e.g., `"S"`, `"I"`, `"R"`)

```python
def run_my_model(*, N, I0, beta, gamma, T, seed=None):
    # ... simulation logic ...
    return {
        "t": t_array,
        "S": susceptible_array,
        "I": infected_array,
        "R": recovered_array,
    }
```

### 2. Use Keyword-Only Arguments

Use `*` to force keyword-only arguments for clarity:

```python
def run_my_model(
    *,
    N: int = 1000,      # Population size
    I0: int = 10,       # Initial infected
    beta: float = 0.2,  # Transmission rate
    gamma: float = 0.1, # Recovery rate
    T: int = 100,       # Time horizon
    seed: int | None = None,  # Random seed
) -> dict:
```

### 3. Standard Parameter Names

Use these standard parameter names for consistency:

| Parameter | Description | Type |
|-----------|-------------|------|
| `N` | Total population size | `int` |
| `I0` | Initial infected | `int` |
| `beta` | Transmission rate | `float` |
| `gamma` | Recovery rate | `float` |
| `T` | Time horizon (days) | `int` |
| `dt` | Time step | `float` or `int` |
| `seed` | Random seed | `int \| None` |
| `reps` | Number of replicates | `int` |
| `R0` | Basic reproduction number | `float` |
| `R_t` | Time-varying reproduction number | `array-like` |

### 4. Provide a Replicates Function

For stochastic models, provide a `_replicates` version:

```python
def run_my_model_replicates(
    *,
    N: int = 1000,
    I0: int = 10,
    # ... other params ...
    reps: int = 10,
    seed: int | None = None,
) -> dict:
    """Run multiple replicates.
    
    Returns dict with arrays of shape (reps, T+1).
    """
```

## Step-by-Step Guide

### Step 1: Create Your Model File

Create a new file in `src/emidm/`, e.g., `src/emidm/seirs.py`:

```python
"""SEIRS model with waning immunity.

This module implements a stochastic SEIRS model where recovered
individuals can become susceptible again.
"""
from __future__ import annotations

import numpy as np


def run_seirs(
    *,
    N: int = 1000,
    I0: int = 10,
    E0: int = 0,
    beta: float = 0.3,
    sigma: float = 0.2,  # E -> I rate
    gamma: float = 0.1,  # I -> R rate
    omega: float = 0.01, # R -> S rate (waning)
    T: int = 100,
    dt: int = 1,
    seed: int | None = None,
) -> dict:
    """Run a stochastic SEIRS model simulation.

    Parameters
    ----------
    N : int, default 1000
        Total population size.
    I0 : int, default 10
        Initial number of infected individuals.
    E0 : int, default 0
        Initial number of exposed individuals.
    beta : float, default 0.3
        Transmission rate.
    sigma : float, default 0.2
        Rate of progression from E to I (1/latent period).
    gamma : float, default 0.1
        Recovery rate.
    omega : float, default 0.01
        Rate of immunity waning (R -> S).
    T : int, default 100
        Total simulation time.
    dt : int, default 1
        Time step size.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys 't', 'S', 'E', 'I', 'R' as numpy arrays.

    Examples
    --------
    >>> from emidm import run_seirs
    >>> result = run_seirs(N=10000, I0=10, beta=0.3, gamma=0.1, T=365)
    >>> result["I"].max()  # Peak infections
    """
    rng = np.random.default_rng(seed)
    
    n_times = (T // dt) + 1
    t_arr = np.arange(0, T + 1, dt)
    
    S_arr = np.zeros(n_times, dtype=np.int64)
    E_arr = np.zeros(n_times, dtype=np.int64)
    I_arr = np.zeros(n_times, dtype=np.int64)
    R_arr = np.zeros(n_times, dtype=np.int64)
    
    S, E, I, R = N - I0 - E0, E0, I0, 0
    
    for idx, t in enumerate(t_arr):
        S_arr[idx], E_arr[idx], I_arr[idx], R_arr[idx] = S, E, I, R
        
        if idx == n_times - 1:
            break
        
        # Transition probabilities
        p_SE = 1 - np.exp(-beta * I / N * dt)
        p_EI = 1 - np.exp(-sigma * dt)
        p_IR = 1 - np.exp(-gamma * dt)
        p_RS = 1 - np.exp(-omega * dt)
        
        # Stochastic transitions
        n_SE = rng.binomial(S, p_SE)
        n_EI = rng.binomial(E, p_EI)
        n_IR = rng.binomial(I, p_IR)
        n_RS = rng.binomial(R, p_RS)
        
        # Update compartments
        S = S - n_SE + n_RS
        E = E + n_SE - n_EI
        I = I + n_EI - n_IR
        R = R + n_IR - n_RS
    
    return {
        "t": t_arr,
        "S": S_arr,
        "E": E_arr,
        "I": I_arr,
        "R": R_arr,
    }
```

### Step 2: Register Your Model

Add registration in your module or in `registry.py`:

```python
from emidm.registry import register_model

@register_model(
    "seirs",
    category="stochastic",
    description="SEIRS model with waning immunity",
    compartments=["S", "E", "I", "R"],
)
def run_seirs(...):
    ...
```

Or register after import:

```python
# In registry.py or __init__.py
from .seirs import run_seirs, run_seirs_replicates

register_model(
    "seirs",
    category="stochastic",
    compartments=["S", "E", "I", "R"],
    replicate_func=run_seirs_replicates,
)(run_seirs)
```

### Step 3: Export from `__init__.py`

Add your model to `src/emidm/__init__.py`:

```python
from .seirs import run_seirs, run_seirs_replicates

__all__ = [
    # ... existing exports ...
    "run_seirs",
    "run_seirs_replicates",
]
```

### Step 4: Add Tests

Create `tests/test_seirs.py`:

```python
import numpy as np
from emidm.seirs import run_seirs


def test_run_seirs_conserves_population():
    result = run_seirs(N=1000, I0=10, T=50, seed=0)
    totals = result["S"] + result["E"] + result["I"] + result["R"]
    assert np.all(totals == 1000)


def test_run_seirs_is_reproducible():
    r1 = run_seirs(N=1000, I0=10, T=30, seed=42)
    r2 = run_seirs(N=1000, I0=10, T=30, seed=42)
    np.testing.assert_array_equal(r1["I"], r2["I"])


def test_run_seirs_returns_correct_keys():
    result = run_seirs(N=100, I0=5, T=10)
    assert set(result.keys()) == {"t", "S", "E", "I", "R"}
    assert result["t"].shape == (11,)
```

### Step 5: Add Documentation

Add a docstring following NumPy format (see example above) and optionally create a notebook in `docs/notebooks/`.

## Differentiable Models

For JAX-based differentiable models, follow the same pattern but:

1. Use `seed: int` parameter (converted to JAX key internally)
2. Import JAX lazily inside the function
3. Use `DiffConfig` for Gumbel-Softmax parameters
4. Ensure gradients flow through (use `hard=False` for training)

```python
from emidm.diff import DiffConfig

def run_diff_seirs(
    *,
    N: int = 200,
    I0: int = 5,
    beta: float = 0.3,
    # ... other params ...
    seed: int = 0,
    config: DiffConfig = DiffConfig(),
) -> dict:
    import jax
    import jax.numpy as jnp
    
    key = jax.random.PRNGKey(seed)
    # ... JAX implementation ...
```

## Checklist for New Models

- [ ] Function uses keyword-only arguments (`*`)
- [ ] Returns `dict` with `"t"` and compartment arrays
- [ ] Uses standard parameter names (`N`, `I0`, `beta`, `gamma`, `T`, `seed`, etc.)
- [ ] Has comprehensive NumPy-style docstring with examples
- [ ] Registered in model registry
- [ ] Exported from `__init__.py`
- [ ] Has unit tests for:
  - [ ] Population conservation
  - [ ] Reproducibility with seed
  - [ ] Correct return structure
- [ ] (Optional) Has `_replicates` version for multiple runs
- [ ] (Optional) Has differentiable version with `run_diff_` prefix

## Questions?

Open an issue on GitHub or check existing models in `src/emidm/sir.py` and `src/emidm/safir.py` for reference implementations.
