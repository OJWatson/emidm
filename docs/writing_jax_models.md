# Writing JAX Models for emidm

This practical guide walks you through implementing a new differentiable epidemiological model in `emidm`. We'll build a complete SEIR model step-by-step, explaining each decision.

## Prerequisites

Before starting, make sure you've read:
- [JAX Fundamentals](jax_fundamentals.md) - Core JAX concepts
- [Model Structure](model_structure.md) - How emidm models are organized
- [End-to-End Differentiability](end_to_end_differentiability.md) - Gumbel-Softmax trick

## Step 1: Define the Model Mathematically

Before writing code, clearly define your model's:

1. **Compartments**: What states can individuals be in?
2. **Transitions**: How do individuals move between states?
3. **Parameters**: What controls the dynamics?

### Example: SEIR Model

```
Compartments: S (Susceptible), E (Exposed), I (Infectious), R (Recovered)

Transitions:
  S → E: Infection (rate = β * I / N)
  E → I: Becoming infectious (rate = σ, where 1/σ = latent period)
  I → R: Recovery (rate = γ)

Parameters:
  β (beta): Transmission rate
  σ (sigma): Rate of becoming infectious (1/latent period)
  γ (gamma): Recovery rate
  N: Population size
  E0, I0: Initial exposed and infected
```

## Step 2: Create the File Structure

Create a new file or add to an existing module:

```python
# src/emidm/diff_seir.py
"""Differentiable SEIR model.

This module implements a differentiable SEIR (Susceptible-Exposed-Infectious-Recovered)
model using JAX and the Gumbel-Softmax trick for end-to-end differentiability.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Type hints that need imports go here
```

## Step 3: Implement Helper Functions

### 3.1 Gumbel-Softmax Sampling

You can reuse the existing implementation or define your own:

```python
def _gumbel_softmax(logits, key, tau, hard):
    """Sample from Gumbel-Softmax distribution.
    
    Args:
        logits: Log-probabilities (can be unnormalized)
        key: JAX random key
        tau: Temperature (lower = more discrete)
        hard: If True, use straight-through estimator
    
    Returns:
        Soft or hard sample from the categorical distribution
    """
    import jax
    import jax.numpy as jnp
    
    # Sample Gumbel noise
    u = jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))
    
    # Compute soft sample
    soft = jax.nn.softmax((logits + gumbel) / tau)
    
    if hard:
        # Straight-through: hard in forward, soft gradient in backward
        hard_sample = jax.nn.one_hot(jnp.argmax(soft), logits.shape[-1])
        return soft + jax.lax.stop_gradient(hard_sample - soft)
    return soft
```

### 3.2 Initial State Setup

Create one-hot encoded initial states:

```python
def _init_seir_state(N, E0, I0):
    """Initialize one-hot encoded SEIR state.
    
    Args:
        N: Population size (determines array size)
        E0: Initial number exposed
        I0: Initial number infectious
    
    Returns:
        Tuple of (S, E, I, R) one-hot arrays
    """
    import jax.numpy as jnp
    
    S0 = N - E0 - I0
    R0 = 0
    
    # One-hot encode each compartment
    S = jnp.zeros(N + 1).at[S0].set(1.0)  # N+1 to allow 0 to N
    E = jnp.zeros(N + 1).at[E0].set(1.0)
    I = jnp.zeros(N + 1).at[I0].set(1.0)
    R = jnp.zeros(N + 1).at[R0].set(1.0)
    
    return (S, E, I, R)
```

**Note**: We use `N + 1` sized arrays to represent counts from 0 to N inclusive.

### 3.3 Transition Logic

The key insight: we need to sample "how many transition" from each compartment.

```python
def _compute_transition_logits(state_onehot, p_transition, max_count):
    """Compute logits for number of transitions.
    
    Given a one-hot state and transition probability, compute the
    log-probability distribution over number of transitions.
    
    Args:
        state_onehot: One-hot encoded current count
        p_transition: Per-individual transition probability
        max_count: Maximum possible count (N)
    
    Returns:
        Logits for number of transitions (0 to max_count)
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
    
    # Get expected current count
    counts = jnp.arange(max_count + 1)
    current_count = jnp.sum(state_onehot * counts)
    
    # For simplicity, use Poisson approximation to binomial
    # Expected transitions = current_count * p_transition
    expected = current_count * p_transition
    
    # Poisson log-pmf: k*log(λ) - λ - log(k!)
    k = counts
    log_pmf = k * jnp.log(expected + 1e-10) - expected - gammaln(k + 1)
    
    # Mask impossible values (can't have more transitions than current count)
    mask = k <= current_count
    log_pmf = jnp.where(mask, log_pmf, -1e10)
    
    return log_pmf
```

**Design choice**: We use a Poisson approximation here for simplicity. For more accuracy, you could implement the full binomial distribution.

## Step 4: Implement the JIT-Compiled Core

This is the performance-critical inner loop:

```python
@partial(jax.jit, static_argnames=["N", "T", "dt", "tau", "hard"])
def _run_diff_seir_core(initial_state, key, beta, sigma, gamma, N, T, dt, tau, hard):
    """JIT-compiled SEIR simulation core.
    
    Args:
        initial_state: Tuple of (S, E, I, R) one-hot arrays
        key: JAX random key
        beta: Transmission rate (differentiable)
        sigma: E→I rate (differentiable)
        gamma: Recovery rate (differentiable)
        N: Population size (static)
        T: Time horizon (static)
        dt: Time step (static)
        tau: Gumbel-Softmax temperature (static)
        hard: Use straight-through estimator (static)
    
    Returns:
        Tuple of (times, S_trajectory, E_trajectory, I_trajectory, R_trajectory)
    """
    import jax
    import jax.numpy as jnp
    
    n_steps = int(T / dt)
    
    def step(carry, t):
        state, key = carry
        S, E, I, R = state
        
        # Split keys for each transition
        key, k1, k2, k3 = jax.random.split(key, 4)
        
        # Get current counts (soft expectations)
        counts = jnp.arange(N + 1)
        S_count = jnp.sum(S * counts)
        E_count = jnp.sum(E * counts)
        I_count = jnp.sum(I * counts)
        R_count = jnp.sum(R * counts)
        
        # Compute transition probabilities
        p_infect = 1 - jnp.exp(-beta * I_count / N * dt)
        p_become_infectious = 1 - jnp.exp(-sigma * dt)
        p_recover = 1 - jnp.exp(-gamma * dt)
        
        # Sample number of each transition
        # S → E transitions
        logits_SE = _compute_transition_logits(S, p_infect, N)
        n_SE_onehot = _gumbel_softmax(logits_SE, k1, tau, hard)
        n_SE = jnp.sum(n_SE_onehot * counts)
        
        # E → I transitions
        logits_EI = _compute_transition_logits(E, p_become_infectious, N)
        n_EI_onehot = _gumbel_softmax(logits_EI, k2, tau, hard)
        n_EI = jnp.sum(n_EI_onehot * counts)
        
        # I → R transitions
        logits_IR = _compute_transition_logits(I, p_recover, N)
        n_IR_onehot = _gumbel_softmax(logits_IR, k3, tau, hard)
        n_IR = jnp.sum(n_IR_onehot * counts)
        
        # Update counts
        new_S_count = S_count - n_SE
        new_E_count = E_count + n_SE - n_EI
        new_I_count = I_count + n_EI - n_IR
        new_R_count = R_count + n_IR
        
        # Clamp to valid range
        new_S_count = jnp.clip(new_S_count, 0, N)
        new_E_count = jnp.clip(new_E_count, 0, N)
        new_I_count = jnp.clip(new_I_count, 0, N)
        new_R_count = jnp.clip(new_R_count, 0, N)
        
        # Convert back to one-hot (approximately)
        # Use soft assignment based on distance to integer
        def soft_onehot(count, size):
            """Create soft one-hot from continuous count."""
            indices = jnp.arange(size)
            # Gaussian-like soft assignment
            distances = (indices - count) ** 2
            return jax.nn.softmax(-distances / 0.1)
        
        new_S = soft_onehot(new_S_count, N + 1)
        new_E = soft_onehot(new_E_count, N + 1)
        new_I = soft_onehot(new_I_count, N + 1)
        new_R = soft_onehot(new_R_count, N + 1)
        
        new_state = (new_S, new_E, new_I, new_R)
        output = (S_count, E_count, I_count, R_count)
        
        return (new_state, key), output
    
    # Run simulation
    times = jnp.arange(n_steps) * dt
    initial_carry = (initial_state, key)
    _, (S_traj, E_traj, I_traj, R_traj) = jax.lax.scan(step, initial_carry, times)
    
    return times, S_traj, E_traj, I_traj, R_traj
```

### Key Design Decisions

1. **Static arguments**: `N`, `T`, `dt`, `tau`, `hard` are static because they affect array shapes or control flow
2. **Dynamic arguments**: `beta`, `sigma`, `gamma` are dynamic because we want gradients w.r.t. them
3. **Key splitting**: We split the random key for each transition to ensure independence
4. **Soft one-hot conversion**: After updating counts, we convert back to soft one-hot for the next iteration

## Step 5: Implement the User-Facing Function

```python
def run_diff_seir(
    *,
    N: int = 200,
    E0: int = 0,
    I0: int = 5,
    beta: float = 0.4,
    sigma: float = 0.2,
    gamma: float = 0.1,
    T: int = 100,
    dt: float = 1.0,
    seed: int = 0,
    tau: float = 0.1,
    hard: bool = True,
) -> dict:
    """Run a differentiable SEIR model simulation.
    
    This model uses the Gumbel-Softmax trick to enable end-to-end
    differentiability through the stochastic simulation.
    
    Parameters
    ----------
    N : int, default 200
        Total population size.
    E0 : int, default 0
        Initial number of exposed individuals.
    I0 : int, default 5
        Initial number of infectious individuals.
    beta : float, default 0.4
        Transmission rate. This parameter is differentiable.
    sigma : float, default 0.2
        Rate of becoming infectious (1/latent period). Differentiable.
    gamma : float, default 0.1
        Recovery rate. This parameter is differentiable.
    T : int, default 100
        Total simulation time.
    dt : float, default 1.0
        Time step size.
    seed : int, default 0
        Random seed for reproducibility.
    tau : float, default 0.1
        Gumbel-Softmax temperature. Lower values give more discrete
        samples but may have gradient issues.
    hard : bool, default True
        If True, use straight-through estimator for discrete forward
        pass with continuous gradients.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 't': Time points array
        - 'S': Susceptible counts over time
        - 'E': Exposed counts over time
        - 'I': Infectious counts over time
        - 'R': Recovered counts over time
    
    Examples
    --------
    >>> result = run_diff_seir(N=1000, I0=10, beta=0.3, sigma=0.2, gamma=0.1, T=100)
    >>> result["I"].max()  # Peak infections
    
    >>> # Compute gradients
    >>> import jax
    >>> def loss(beta):
    ...     result = run_diff_seir(beta=beta, hard=False, tau=0.5)
    ...     return result["I"].sum()
    >>> grad_fn = jax.grad(loss)
    >>> grad_fn(0.3)  # Gradient of total infections w.r.t. beta
    """
    import jax
    
    # Convert seed to JAX key
    key = jax.random.PRNGKey(seed)
    
    # Initialize state
    initial_state = _init_seir_state(N, E0, I0)
    
    # Run simulation
    t, S, E, I, R = _run_diff_seir_core(
        initial_state, key, beta, sigma, gamma, N, T, dt, tau, hard
    )
    
    return {"t": t, "S": S, "E": E, "I": I, "R": R}
```

## Step 6: Add Tests

Create comprehensive tests:

```python
# tests/test_diff_seir.py
"""Tests for differentiable SEIR model."""
import numpy as np
import pytest


def test_run_diff_seir_shapes():
    """Test output shapes are correct."""
    from emidm.diff_seir import run_diff_seir
    
    result = run_diff_seir(N=100, I0=5, T=50, dt=1.0)
    
    assert "t" in result
    assert "S" in result
    assert "E" in result
    assert "I" in result
    assert "R" in result
    
    assert result["t"].shape == (50,)
    assert result["S"].shape == (50,)


def test_run_diff_seir_conservation():
    """Test population is conserved."""
    from emidm.diff_seir import run_diff_seir
    import numpy as np
    
    result = run_diff_seir(N=100, E0=5, I0=5, T=50, seed=42)
    
    total = result["S"] + result["E"] + result["I"] + result["R"]
    np.testing.assert_allclose(total, 100, atol=1)  # Allow small numerical error


def test_run_diff_seir_is_differentiable():
    """Test that gradients can be computed."""
    pytest.importorskip("jax")
    import jax
    from emidm.diff_seir import run_diff_seir
    
    def loss(beta):
        result = run_diff_seir(N=50, I0=5, T=30, beta=beta, hard=False, tau=0.5)
        return result["I"].sum()
    
    grad_fn = jax.grad(loss)
    grad = grad_fn(0.3)
    
    assert np.isfinite(grad)
    assert grad != 0  # Gradient should be non-zero


def test_run_diff_seir_reproducible():
    """Test same seed gives same results."""
    from emidm.diff_seir import run_diff_seir
    import numpy as np
    
    r1 = run_diff_seir(N=100, I0=5, T=30, seed=42)
    r2 = run_diff_seir(N=100, I0=5, T=30, seed=42)
    
    np.testing.assert_array_equal(r1["I"], r2["I"])
```

## Step 7: Register and Export

Add to the model registry and exports:

```python
# In src/emidm/__init__.py
from .diff_seir import run_diff_seir

__all__ = [
    # ... existing exports ...
    "run_diff_seir",
]

# In src/emidm/registry.py (add to auto-registration)
register_model(
    "diff_seir",
    category="differentiable",
    compartments=["S", "E", "I", "R"],
)(run_diff_seir)
```

## Common Pitfalls and Solutions

### Pitfall 1: Python Control Flow

```python
# ❌ BAD: Python if based on array value
if I_count > 0:
    p_infect = beta * I_count / N
else:
    p_infect = 0

# ✅ GOOD: Use jax.lax.cond or jnp.where
p_infect = jnp.where(I_count > 0, beta * I_count / N, 0.0)
```

### Pitfall 2: In-Place Modifications

```python
# ❌ BAD: In-place modification
state[0] = new_value

# ✅ GOOD: Create new array
state = state.at[0].set(new_value)
```

### Pitfall 3: Dynamic Shapes

```python
# ❌ BAD: Shape depends on runtime value
result = jnp.zeros(int(T / dt))  # T is dynamic

# ✅ GOOD: Make T static
@partial(jax.jit, static_argnames=["T", "dt"])
def simulate(params, T, dt):
    result = jnp.zeros(int(T / dt))  # Now T is known at compile time
```

### Pitfall 4: Forgetting to Split Keys

```python
# ❌ BAD: Reusing the same key
sample1 = jax.random.normal(key, (10,))
sample2 = jax.random.normal(key, (10,))  # Same as sample1!

# ✅ GOOD: Split keys
key, k1, k2 = jax.random.split(key, 3)
sample1 = jax.random.normal(k1, (10,))
sample2 = jax.random.normal(k2, (10,))  # Different!
```

### Pitfall 5: Gradient Through Hard Samples

```python
# ❌ BAD: No gradient flows
hard_sample = jax.nn.one_hot(soft.argmax(), n)

# ✅ GOOD: Straight-through estimator
hard_sample = soft + jax.lax.stop_gradient(
    jax.nn.one_hot(soft.argmax(), n) - soft
)
```

## Performance Tips

1. **Minimize Python overhead**: Keep all computation in JAX operations
2. **Use `scan` not loops**: `jax.lax.scan` is much faster than Python for-loops
3. **Batch operations**: Use `vmap` for multiple replicates instead of Python loops
4. **Profile with `jax.profiler`**: Find bottlenecks in your code
5. **Consider `float32`**: Faster than `float64` on GPUs

## Checklist for New Models

- [ ] Mathematical model clearly defined
- [ ] Two-layer architecture (user-facing + JIT core)
- [ ] Static arguments marked correctly
- [ ] Dynamic arguments for differentiable parameters
- [ ] Gumbel-Softmax for discrete sampling
- [ ] Straight-through estimator option
- [ ] Explicit random key handling
- [ ] Comprehensive docstring with examples
- [ ] Tests for shapes, conservation, differentiability, reproducibility
- [ ] Registered in model registry
- [ ] Exported from `__init__.py`

## Next Steps

- Study the existing implementations in `src/emidm/diff.py`
- Try modifying parameters and observing gradient behavior
- Experiment with different temperature values
- Consider adding age structure or spatial components
