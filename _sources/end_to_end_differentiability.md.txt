# End-to-End Differentiability in Epidemiological Models

This document explains the core challenge of making stochastic, discrete epidemiological models differentiable, and how `emidm` solves it using the Gumbel-Softmax trick.

## The Problem: Discrete Operations Break Gradients

Epidemiological models involve **discrete** operations that have zero or undefined gradients:

```python
# Stochastic SIR: sample number of new infections
new_infections = np.random.binomial(S, p_infect)  # Discrete!

# The gradient of binomial sampling is undefined
# ∂(new_infections)/∂(p_infect) = ???
```

This is problematic because:
1. **Sampling is non-differentiable**: You can't backpropagate through `random.binomial`
2. **Argmax is non-differentiable**: `argmax` has zero gradient almost everywhere
3. **Integer operations**: Counts are integers, gradients need continuous values

## The Solution: Gumbel-Softmax (Concrete Distribution)

The **Gumbel-Softmax** trick (also called the Concrete distribution) provides a differentiable approximation to categorical sampling.

### How Categorical Sampling Works

To sample from a categorical distribution with probabilities `[p₀, p₁, ..., pₙ]`:

```python
# Standard approach: sample category index
category = np.random.choice(n, p=probs)  # Returns integer 0, 1, ..., n-1
```

This returns a **hard** one-hot vector (all zeros except one 1), but the sampling operation has no gradient.

### The Gumbel-Max Trick

The Gumbel-Max trick reformulates categorical sampling:

```python
# Equivalent to categorical sampling:
gumbel_noise = -log(-log(uniform(0, 1)))  # Gumbel(0, 1) samples
logits = log(probs)  # Convert probs to log-odds
category = argmax(logits + gumbel_noise)  # Sample!
```

This is mathematically equivalent to categorical sampling, but still has the non-differentiable `argmax`.

### The Gumbel-Softmax Trick

Replace `argmax` with `softmax` (which IS differentiable):

```python
def gumbel_softmax(logits, key, tau):
    """Differentiable approximation to categorical sampling."""
    # Sample Gumbel noise
    gumbel_noise = -jnp.log(-jnp.log(
        jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0)
    ))
    
    # Add noise to logits
    noisy_logits = (logits + gumbel_noise) / tau
    
    # Softmax instead of argmax
    return jax.nn.softmax(noisy_logits)
```

The **temperature** `τ` (tau) controls the approximation:
- `τ → 0`: Output approaches one-hot (like argmax), but gradient vanishes
- `τ → ∞`: Output approaches uniform distribution
- `τ ≈ 0.1-1.0`: Good balance between discreteness and gradient flow

### Visualization

```
Temperature Effect on Gumbel-Softmax Output:

τ = 0.01 (nearly discrete):  [0.00, 0.00, 0.99, 0.01, 0.00]  ← Almost one-hot
τ = 0.1  (typical):          [0.02, 0.05, 0.78, 0.12, 0.03]  ← Peaked but soft
τ = 1.0  (soft):             [0.10, 0.15, 0.35, 0.25, 0.15]  ← Spread out
τ = 10.0 (very soft):        [0.18, 0.20, 0.22, 0.21, 0.19]  ← Nearly uniform
```

## The Straight-Through Estimator

For the forward pass, we often want **discrete** outputs (actual counts), but for the backward pass, we need **continuous** gradients.

The **straight-through estimator** achieves this:

```python
def straight_through_gumbel_softmax(logits, key, tau):
    """Hard samples in forward pass, soft gradients in backward pass."""
    # Soft sample (differentiable)
    soft = gumbel_softmax(logits, key, tau)
    
    # Hard sample (one-hot)
    hard = jax.nn.one_hot(soft.argmax(), len(soft))
    
    # Trick: use hard for forward, soft gradient for backward
    # jax.lax.stop_gradient(hard - soft) has zero gradient
    return soft + jax.lax.stop_gradient(hard - soft)
```

This works because:
- **Forward pass**: Returns `hard` (the one-hot vector)
- **Backward pass**: Gradient flows through `soft` (ignoring the stopped gradient term)

### In emidm

```python
def _gumbel_softmax_sample(logits, key, tau, hard):
    """Sample from Gumbel-Softmax distribution.
    
    Args:
        logits: Log-probabilities (unnormalized)
        key: JAX random key
        tau: Temperature parameter
        hard: If True, use straight-through estimator
    """
    # Sample Gumbel noise
    u = jax.random.uniform(key, logits.shape, minval=1e-10, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))
    
    # Soft sample
    soft = jax.nn.softmax((logits + gumbel) / tau)
    
    if hard:
        # Straight-through: hard forward, soft backward
        hard_sample = jax.nn.one_hot(soft.argmax(), logits.shape[0])
        return soft + jax.lax.stop_gradient(hard_sample - soft)
    else:
        return soft
```

## Applying to Epidemiological Transitions

### The Challenge

In an SIR model, we need to sample:
1. How many susceptibles get infected (binomial)
2. How many infected recover (binomial)

These are **binomial** distributions, not categorical. How do we apply Gumbel-Softmax?

### The Solution: One-Hot State Representation

We represent counts as **one-hot probability distributions**:

```python
# Instead of: S = 990 (integer)
# We use: S = [0, 0, ..., 0, 1, 0, ..., 0]  (one-hot, 1 at position 990)
#              ↑ position 0    ↑ position 990
```

Now, transitions become **categorical** operations on these distributions:

```python
def transition_S_to_I(S_onehot, p_infect, key, tau, hard):
    """Transition some susceptibles to infected.
    
    S_onehot: One-hot distribution over possible S counts
    p_infect: Probability of infection per susceptible
    """
    N = len(S_onehot)
    
    # Current S count (expected value or argmax)
    S_count = jnp.sum(S_onehot * jnp.arange(N))
    
    # For each possible current S value, compute distribution over new infections
    # This creates a transition matrix
    
    # Simplified: sample new S directly
    # Compute logits for each possible new S value
    # ... (complex transition logic)
    
    # Sample new S using Gumbel-Softmax
    new_S_onehot = _gumbel_softmax_sample(logits, key, tau, hard)
    
    return new_S_onehot
```

### Transition Matrices

For efficiency, we can precompute **transition matrices** that encode all possible transitions:

```python
def build_infection_matrix(N, p_infect):
    """Build matrix M where M[i,j] = P(new_S = j | old_S = i)."""
    # For each starting S value i, compute distribution over ending S values j
    # This involves binomial probabilities
    
    matrix = jnp.zeros((N, N))
    for i in range(N):
        # If S = i, new infections ~ Binomial(i, p_infect)
        # So new_S ~ i - Binomial(i, p_infect)
        for k in range(i + 1):  # k = number of new infections
            prob = binom_pmf(k, i, p_infect)
            new_S = i - k
            matrix = matrix.at[i, new_S].set(prob)
    
    return matrix
```

Then apply the transition:

```python
# Transition: multiply state by transition matrix
new_S_probs = S_onehot @ transition_matrix  # Matrix multiplication!

# Sample from the resulting distribution
new_S_onehot = _gumbel_softmax_sample(jnp.log(new_S_probs + 1e-10), key, tau, hard)
```

## Complete Example: Differentiable SIR Step

Here's how a single time step works in `emidm`:

```python
def sir_step(carry, t):
    """One differentiable SIR time step."""
    (S, I, R), key = carry
    key, k1, k2 = jax.random.split(key, 3)
    
    N = len(S)
    
    # Get current counts (as soft expectations)
    S_count = jnp.sum(S * jnp.arange(N))
    I_count = jnp.sum(I * jnp.arange(N))
    
    # Compute transition probabilities
    p_infect = 1 - jnp.exp(-beta * I_count / N * dt)
    p_recover = 1 - jnp.exp(-gamma * dt)
    
    # Build transition distributions
    # (Simplified - actual implementation uses efficient matrix ops)
    
    # Sample new infections using Gumbel-Softmax
    infection_logits = compute_infection_logits(S, p_infect)
    new_infections_onehot = _gumbel_softmax_sample(infection_logits, k1, tau, hard)
    
    # Sample recoveries using Gumbel-Softmax  
    recovery_logits = compute_recovery_logits(I, p_recover)
    new_recoveries_onehot = _gumbel_softmax_sample(recovery_logits, k2, tau, hard)
    
    # Update compartments
    new_S = update_S(S, new_infections_onehot)
    new_I = update_I(I, new_infections_onehot, new_recoveries_onehot)
    new_R = update_R(R, new_recoveries_onehot)
    
    # Output counts for trajectory
    output = (S_count, I_count, N - S_count - I_count)
    
    return ((new_S, new_I, new_R), key), output
```

## Gradient Flow Visualization

```
Forward Pass (with hard=True):
┌─────────┐    ┌──────────────┐    ┌─────────┐    ┌──────┐
│  beta   │───▶│ p_infect     │───▶│ logits  │───▶│ hard │───▶ Loss
│  gamma  │    │ computation  │    │         │    │sample│
└─────────┘    └──────────────┘    └─────────┘    └──────┘
                                         │
                                   (discrete output)

Backward Pass:
┌─────────┐    ┌──────────────┐    ┌─────────┐    ┌──────┐
│ ∂L/∂β   │◀───│   gradient   │◀───│ ∂L/∂log │◀───│ soft │◀─── ∂L/∂out
│ ∂L/∂γ   │    │   flows!     │    │         │    │ grad │
└─────────┘    └──────────────┘    └─────────┘    └──────┘
                                         │
                                   (continuous gradient)
```

## Training vs Inference

| Mode | `hard` | `tau` | Use Case |
|------|--------|-------|----------|
| Training | `False` | 0.1-1.0 | Gradient-based optimization |
| Inference | `True` | 0.1 | Discrete samples for prediction |
| Annealing | `False` → `True` | 1.0 → 0.1 | Curriculum learning |

### Temperature Annealing

During training, you might anneal temperature:

```python
# Start soft, gradually become more discrete
for epoch in range(n_epochs):
    tau = max(0.1, 1.0 - epoch * 0.01)  # Anneal from 1.0 to 0.1
    
    loss, grads = jax.value_and_grad(loss_fn)(params, tau=tau)
    params = update(params, grads)
```

## Key Takeaways

1. **Gumbel-Softmax** provides differentiable categorical sampling
2. **Temperature** controls the softness of the approximation
3. **Straight-through estimator** gives discrete forward, continuous backward
4. **One-hot encoding** converts count-based models to categorical form
5. **Transition matrices** efficiently encode binomial transitions

## Mathematical Details

### Gumbel Distribution

The Gumbel(0, 1) distribution has CDF: `F(x) = exp(-exp(-x))`

To sample: `g = -log(-log(u))` where `u ~ Uniform(0, 1)`

### Why Gumbel-Max Works

For logits `z₁, ..., zₙ` and Gumbel noise `g₁, ..., gₙ`:

```
P(argmax(zᵢ + gᵢ) = k) = exp(zₖ) / Σⱼ exp(zⱼ) = softmax(z)ₖ
```

This is the **Gumbel-Max theorem** - adding Gumbel noise and taking argmax is equivalent to sampling from the softmax distribution.

### Gradient of Softmax

The softmax function has well-defined gradients:

```
∂softmax(z)ᵢ/∂zⱼ = softmax(z)ᵢ * (δᵢⱼ - softmax(z)ⱼ)
```

This allows gradients to flow through the Gumbel-Softmax operation.

## References

- Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR.
- Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). The Concrete Distribution. ICLR.
- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. arXiv.

## Next Steps

- [Writing JAX Models](writing_jax_models.md) - Practical guide to implementing new models
- [JAX Fundamentals](jax_fundamentals.md) - Core JAX concepts
