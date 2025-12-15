# JAX Fundamentals for Epidemiological Modelling

This document explains the core JAX concepts used in `emidm`'s differentiable models. Understanding these fundamentals will help you read, modify, and extend the codebase.

## Why JAX?

Traditional epidemiological models are written in NumPy or R and run forward simulations. While useful for prediction, they can't be directly used for:

1. **Gradient-based optimization** - Finding parameters that minimize a loss function
2. **Bayesian inference** - Computing gradients of log-posteriors for HMC/NUTS
3. **Sensitivity analysis** - Understanding how outputs change with inputs

JAX solves this by providing:
- **Automatic differentiation** (`jax.grad`) - Compute gradients of any function
- **Just-in-time compilation** (`jax.jit`) - Compile Python to fast machine code
- **Vectorization** (`jax.vmap`) - Automatically batch operations
- **GPU/TPU support** - Same code runs on accelerators

## Core Concept 1: Functional Programming

JAX requires **pure functions** - functions that:
1. Always return the same output for the same input
2. Have no side effects (don't modify external state)

### Why This Matters

```python
# ❌ BAD: Impure function (modifies external state)
state = {"count": 0}

def increment():
    state["count"] += 1  # Side effect!
    return state["count"]

# ✅ GOOD: Pure function (no side effects)
def increment(state):
    return {"count": state["count"] + 1}
```

In `emidm`, this means:
- **No in-place array modifications** - Always create new arrays
- **Explicit state passing** - Pass state as arguments, return new state
- **Random keys must be passed explicitly** - No global random state

### Random Number Generation in JAX

Unlike NumPy's global random state, JAX uses explicit random keys:

```python
import jax
import jax.random as random

# Create a PRNG key from a seed
key = random.PRNGKey(42)

# Split the key to get new keys (never reuse keys!)
key, subkey = random.split(key)

# Use subkey for random operations
samples = random.normal(subkey, shape=(10,))
```

In `emidm` models, we accept a `seed: int` parameter and convert it internally:

```python
def run_diff_sir(*, seed: int = 0, ...):
    key = jax.random.PRNGKey(seed)
    # ... use key for random operations
```

## Core Concept 2: Just-In-Time Compilation (JIT)

`jax.jit` compiles Python functions to XLA (Accelerated Linear Algebra) for fast execution.

### How JIT Works

```python
import jax
import jax.numpy as jnp

def slow_function(x):
    for i in range(1000):
        x = x + 1
    return x

# First call: traces and compiles (slow)
# Subsequent calls: runs compiled code (fast)
fast_function = jax.jit(slow_function)
```

### JIT Constraints

For JIT to work, functions must be **traceable** - JAX must be able to determine the computation graph at compile time.

**What breaks JIT:**

```python
# ❌ BAD: Python control flow depending on array values
@jax.jit
def bad_function(x):
    if x > 0:  # Can't trace: depends on runtime value
        return x * 2
    else:
        return x * 3

# ✅ GOOD: Use jax.lax.cond for conditional logic
@jax.jit
def good_function(x):
    return jax.lax.cond(
        x > 0,
        lambda x: x * 2,  # True branch
        lambda x: x * 3,  # False branch
        x
    )
```

**What else breaks JIT:**

```python
# ❌ BAD: Dynamic shapes
@jax.jit
def bad_dynamic(n):
    return jnp.zeros(n)  # Shape depends on runtime value

# ✅ GOOD: Static shapes (use static_argnums if needed)
@jax.jit
def good_static():
    return jnp.zeros(100)  # Shape known at compile time
```

### Static Arguments

When some arguments determine shapes or control flow, mark them as static:

```python
from functools import partial

@partial(jax.jit, static_argnums=(1,))  # Second arg is static
def create_array(value, size):
    return jnp.full(size, value)

# Or use static_argnames
@partial(jax.jit, static_argnames=["T"])
def simulate(params, T):
    return jnp.zeros(T)
```

In `emidm`, we use this pattern for time horizon `T` and other shape-determining parameters.

## Core Concept 3: Automatic Differentiation

JAX can compute gradients of any differentiable function:

```python
import jax

def loss(params):
    return (params["beta"] - 0.3) ** 2 + (params["gamma"] - 0.1) ** 2

# Get gradient function
grad_loss = jax.grad(loss)

# Compute gradients
params = {"beta": 0.5, "gamma": 0.2}
grads = grad_loss(params)
# grads = {"beta": 0.4, "gamma": 0.2}
```

### Forward vs Reverse Mode

JAX supports both:
- **Reverse mode** (`jax.grad`): Efficient for many inputs → one output (common in ML)
- **Forward mode** (`jax.jvp`): Efficient for one input → many outputs

For optimization (scalar loss), reverse mode is almost always what you want.

### Gradients Through Loops

JAX can differentiate through `jax.lax.scan` (the functional equivalent of a for-loop):

```python
def simulate(params, T):
    def step(state, t):
        # One simulation step
        new_state = state + params["rate"]
        return new_state, new_state
    
    initial_state = 0.0
    final_state, trajectory = jax.lax.scan(step, initial_state, jnp.arange(T))
    return trajectory

# This is differentiable!
grad_fn = jax.grad(lambda p: simulate(p, 100).sum())
```

## Core Concept 4: PyTrees

JAX operates on **PyTrees** - nested structures of arrays:

```python
# All of these are valid PyTrees
params = {"beta": 0.3, "gamma": 0.1}
params = [0.3, 0.1]
params = {"model": {"beta": 0.3}, "prior": {"sigma": 1.0}}
```

`jax.grad` automatically handles PyTrees:

```python
def loss(params):
    return params["beta"] ** 2 + params["gamma"] ** 2

grads = jax.grad(loss)({"beta": 0.3, "gamma": 0.1})
# grads = {"beta": 0.6, "gamma": 0.2}
```

## Putting It Together: A Simple Example

Here's a minimal differentiable SIR model showing all concepts:

```python
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=["T", "dt"])
def simple_sir(beta, gamma, N, I0, T, dt=1.0):
    """A simple differentiable SIR model."""
    
    # Initial state
    S0 = N - I0
    R0 = 0.0
    state = jnp.array([S0, I0, R0])
    
    def step(state, t):
        S, I, R = state
        
        # Compute rates (continuous approximation)
        infection_rate = beta * S * I / N
        recovery_rate = gamma * I
        
        # Update state (Euler method)
        dS = -infection_rate * dt
        dI = (infection_rate - recovery_rate) * dt
        dR = recovery_rate * dt
        
        new_state = state + jnp.array([dS, dI, dR])
        return new_state, new_state
    
    # Run simulation using scan
    times = jnp.arange(0, T, dt)
    _, trajectory = jax.lax.scan(step, state, times)
    
    return trajectory

# This is fully differentiable!
def loss(beta, gamma, observed_I):
    trajectory = simple_sir(beta, gamma, N=1000, I0=10, T=100)
    predicted_I = trajectory[:, 1]  # I compartment
    return jnp.mean((predicted_I - observed_I) ** 2)

# Compute gradients
grad_fn = jax.grad(loss, argnums=(0, 1))
d_beta, d_gamma = grad_fn(0.3, 0.1, observed_data)
```

## Key Takeaways

1. **Write pure functions** - No side effects, explicit state passing
2. **Use JAX primitives** - `jax.lax.scan` instead of Python loops, `jax.lax.cond` instead of if/else
3. **Mark static arguments** - Anything that determines shapes or control flow
4. **Pass random keys explicitly** - No global random state
5. **Think in arrays** - Vectorize operations, avoid Python loops over array elements

## Next Steps

- [Model Structure](model_structure.md) - How emidm's differentiable models are organized
- [End-to-End Differentiability](end_to_end_differentiability.md) - Handling discrete/stochastic operations
- [Writing JAX Models](writing_jax_models.md) - Practical guide to implementing new models
