# Model Structure in emidm

This document explains how the differentiable models in `emidm` are structured, focusing on the design patterns that enable JIT compilation and automatic differentiation.

## Overview: The Two-Layer Architecture

Every differentiable model in `emidm` follows a two-layer pattern:

```
┌─────────────────────────────────────────────────────────────┐
│  User-Facing Function (run_diff_sir_simulation)             │
│  - Accepts JAX PRNGKey for vmap/jit compatibility           │
│  - Sets up initial state                                    │
│  - Calls JIT-compiled core                                  │
│  - Returns user-friendly dict                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  JIT-Compiled Core (_run_diff_sir_core)                     │
│  - Pure function with explicit key passing                  │
│  - Uses jax.lax.scan for time stepping                      │
│  - All shapes determined by static arguments                │
│  - Returns JAX arrays                                       │
└─────────────────────────────────────────────────────────────┘
```

### Why Two Layers?

1. **User convenience**: The outer function provides a clean API with sensible defaults
2. **JIT efficiency**: The inner function is optimized for compilation
3. **Separation of concerns**: Setup logic vs. computation logic

## Layer 1: User-Facing Function

The user-facing function handles all the "messy" parts that don't need to be JIT-compiled:

```python
def run_diff_sir_simulation(
    *,
    N: int = 200,
    I0: int = 5,
    beta: float = 0.4,
    gamma: float = 0.1,
    T: int = 100,
    dt: float = 1.0,
    key,  # JAX PRNGKey - required for vmap/jit compatibility
    config: DiffConfig = DiffConfig(),
) -> dict:
    """Run a differentiable SIR model simulation."""
    import jax
    import jax.numpy as jnp
    
    # 1. Key is passed directly (enables vmap/jit)
    # 2. Set up initial state
    S0 = jnp.zeros(N).at[0].set(N - I0)  # One-hot encoding
    I0_arr = jnp.zeros(N).at[I0].set(1.0)
    R0_arr = jnp.zeros(N).at[0].set(1.0)
    initial_state = (S0, I0_arr, R0_arr)
    
    # 3. Call JIT-compiled core
    t, S, I, R = _run_diff_sir_core(
        initial_state, key, beta, gamma, N, T, dt, tau, hard
    )
    
    # 4. Return user-friendly dict
    return {"t": t, "S": S, "I": I, "R": R}
```

### Key Responsibilities

| Task | Why in Outer Function |
|------|----------------------|
| Key validation | Ensures valid JAX PRNGKey is provided |
| Initial state setup | May involve complex logic not needed in hot loop |
| Return dict formatting | Converts JAX arrays to user-friendly format |
| Default parameters | Provides sensible defaults without cluttering core |

## Layer 2: JIT-Compiled Core

The core function is where the actual simulation happens:

```python
from functools import partial

@partial(jax.jit, static_argnames=["N", "T", "dt", "tau", "hard"])
def _run_diff_sir_core(initial_state, key, beta, gamma, N, T, dt, tau, hard):
    """JIT-compiled core of the SIR simulation."""
    
    n_steps = int(T / dt)
    
    def step(carry, t):
        state, key = carry
        S, I, R = state
        
        # Split key for this step
        key, subkey = jax.random.split(key)
        
        # Compute transitions (see end_to_end_differentiability.md)
        # ... transition logic ...
        
        new_state = (new_S, new_I, new_R)
        
        # Return updated carry and output
        return (new_state, key), (S.argmax(), I.argmax(), R.argmax())
    
    # Run simulation with scan
    times = jnp.arange(n_steps) * dt
    _, (S_traj, I_traj, R_traj) = jax.lax.scan(
        step, (initial_state, key), times
    )
    
    return times, S_traj, I_traj, R_traj
```

### Static vs Dynamic Arguments

Understanding which arguments are static is crucial:

```python
@partial(jax.jit, static_argnames=["N", "T", "dt", "tau", "hard"])
def _run_diff_sir_core(initial_state, key, beta, gamma, N, T, dt, tau, hard):
    #                   ↑ dynamic ↑        ↑ static ↑
```

| Argument | Static? | Why |
|----------|---------|-----|
| `initial_state` | No | Array values can change |
| `key` | No | Different random seeds |
| `beta`, `gamma` | No | **These are what we optimize!** |
| `N` | Yes | Determines array shapes |
| `T`, `dt` | Yes | Determines number of time steps |
| `tau`, `hard` | Yes | Control flow / algorithm selection |

**Critical insight**: Parameters we want to differentiate with respect to (`beta`, `gamma`) must be **dynamic** arguments.

## The Simulation Loop: `jax.lax.scan`

The heart of every model is the time-stepping loop, implemented with `jax.lax.scan`:

```python
def step(carry, t):
    """One time step of the simulation.
    
    Args:
        carry: Tuple of (state, key) - things that update each step
        t: Current time (from the xs array)
    
    Returns:
        new_carry: Updated (state, key)
        output: Values to collect (trajectory)
    """
    state, key = carry
    
    # ... compute new_state ...
    
    return (new_state, key), output_for_trajectory

# Run the loop
initial_carry = (initial_state, key)
xs = jnp.arange(n_steps)  # "Inputs" for each step (often just indices)

final_carry, stacked_outputs = jax.lax.scan(step, initial_carry, xs)
```

### Why `scan` Instead of Python Loops?

```python
# ❌ Python loop - can't JIT, can't differentiate efficiently
trajectory = []
state = initial_state
for t in range(T):
    state = step(state)
    trajectory.append(state)

# ✅ jax.lax.scan - JIT-compatible, efficient autodiff
_, trajectory = jax.lax.scan(step, initial_state, jnp.arange(T))
```

Benefits of `scan`:
1. **JIT-compatible**: Compiles to efficient XLA operations
2. **Memory efficient**: Doesn't store intermediate computation graphs
3. **Differentiable**: JAX knows how to backpropagate through it

## State Representation: One-Hot Encoding

In `emidm`'s differentiable models, we represent compartment counts using **one-hot encoded probability distributions**:

```python
# Traditional representation: scalar counts
S, I, R = 990, 10, 0

# One-hot representation: probability distributions
# S[i] = probability that S = i
S = jnp.zeros(N)  # S[0]=0, S[1]=0, ..., S[990]=1, ...
S = S.at[990].set(1.0)  # One-hot: all probability mass at 990

I = jnp.zeros(N)
I = I.at[10].set(1.0)  # One-hot: all probability mass at 10
```

### Why One-Hot?

1. **Enables Gumbel-Softmax**: We can sample from these distributions differentiably
2. **Represents uncertainty**: During soft sampling, probability spreads across values
3. **Gradient flow**: Changes in parameters affect the full distribution

### Converting Back to Counts

```python
# Get the most likely count (argmax)
S_count = S.argmax()  # Returns 990

# Or get expected value
S_expected = jnp.sum(S * jnp.arange(N))  # Weighted average
```

## Putting It All Together: Complete Model Structure

Here's the complete structure of a differentiable model:

```python
"""Differentiable SIR model."""
from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

# ============================================================
# LAYER 2: JIT-COMPILED CORE
# ============================================================

@partial(jax.jit, static_argnames=["N", "T", "dt", "tau", "hard"])
def _run_diff_sir_core(initial_state, key, beta, gamma, N, T, dt, tau, hard):
    """JIT-compiled simulation core.
    
    All array shapes are determined by static arguments.
    Dynamic arguments (beta, gamma) can be differentiated.
    """
    import jax.numpy as jnp
    
    n_steps = int(T / dt)
    
    def step(carry, t):
        (S, I, R), key = carry
        key, subkey = jax.random.split(key)
        
        # Compute transition probabilities
        S_count = jnp.sum(S * jnp.arange(N))
        I_count = jnp.sum(I * jnp.arange(N))
        
        p_infect = 1 - jnp.exp(-beta * I_count / N * dt)
        p_recover = 1 - jnp.exp(-gamma * dt)
        
        # Sample transitions (using Gumbel-Softmax)
        # ... (see end_to_end_differentiability.md)
        
        new_state = (new_S, new_I, new_R)
        output = (S_count, I_count, R_count)
        
        return (new_state, key), output
    
    # Run simulation
    times = jnp.arange(n_steps) * dt
    _, (S_traj, I_traj, R_traj) = jax.lax.scan(
        step, (initial_state, key), times
    )
    
    return times, S_traj, I_traj, R_traj


# ============================================================
# HELPER: INITIAL STATE SETUP
# ============================================================

def _init_state(N, I0):
    """Create one-hot encoded initial state."""
    import jax.numpy as jnp
    
    S0 = jnp.zeros(N).at[N - I0].set(1.0)
    I0_arr = jnp.zeros(N).at[I0].set(1.0)
    R0_arr = jnp.zeros(N).at[0].set(1.0)
    
    return (S0, I0_arr, R0_arr)


# ============================================================
# LAYER 1: USER-FACING FUNCTION
# ============================================================

def run_diff_sir(
    *,
    N: int = 200,
    I0: int = 5,
    beta: float = 0.4,
    gamma: float = 0.1,
    T: int = 100,
    dt: float = 1.0,
    seed: int = 0,
    tau: float = 0.1,
    hard: bool = True,
) -> dict:
    """Run a differentiable SIR model.
    
    Parameters
    ----------
    N : int
        Population size.
    I0 : int
        Initial infected.
    beta : float
        Transmission rate (differentiable).
    gamma : float
        Recovery rate (differentiable).
    T : int
        Time horizon.
    dt : float
        Time step.
    seed : int
        Random seed.
    tau : float
        Gumbel-Softmax temperature.
    hard : bool
        If True, use straight-through estimator.
    
    Returns
    -------
    dict
        Dictionary with 't', 'S', 'I', 'R' arrays.
    """
    import jax
    
    # Setup
    key = jax.random.PRNGKey(seed)
    initial_state = _init_state(N, I0)
    
    # Run core
    t, S, I, R = _run_diff_sir_core(
        initial_state, key, beta, gamma, N, T, dt, tau, hard
    )
    
    # Return dict
    return {"t": t, "S": S, "I": I, "R": R}
```

## Design Patterns Summary

| Pattern | Purpose | Example |
|---------|---------|--------|
| Two-layer architecture | Separate user API from JIT core | `run_diff_sir_simulation` → `_run_diff_sir_core` |
| Static arguments | Enable JIT with dynamic shapes | `static_argnames=["N", "T"]` |
| `jax.lax.scan` | Differentiable time stepping | `scan(step, init, times)` |
| One-hot encoding | Enable Gumbel-Softmax sampling | `S = zeros(N).at[count].set(1.0)` |
| Explicit key passing | Functional random number generation | `key, subkey = split(key)` |

## Next Steps

- [End-to-End Differentiability](end_to_end_differentiability.md) - How we handle discrete/stochastic operations
- [Writing JAX Models](writing_jax_models.md) - Practical guide to implementing new models
