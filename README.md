<!-- badges: start -->

![Publish Website](https://github.com/ojwatson/emidm/actions/workflows/publish.yml/badge.svg)
![Tests](https://github.com/ojwatson/emidm/actions/workflows/tests.yml/badge.svg)
![Docs](https://github.com/ojwatson/emidm/actions/workflows/docs.yml/badge.svg)
[![Website](https://img.shields.io/badge/View-Website-blue?logo=githubpages&style=flat-square)](https://ojwatson.github.io/emidm/)

<!-- badges: end -->

# emidm

**Differentiable Epidemiological Modelling in Python**

`emidm` provides stochastic and differentiable infectious disease models built with JAX, enabling gradient-based calibration and Bayesian inference for epidemiological parameters.

## Features

- **Stochastic Models**: SIR and age-structured SAFIR/SEIR models with contact matrices
- **Differentiable Models**: JAX-based implementations using Gumbel-Softmax for gradient-based optimization
- **Optimization**: Built-in Optax integration for parameter calibration
- **Bayesian Inference**: BlackJAX scaffolding for MCMC sampling

## Installation

```bash
pip install git+https://github.com/OJWatson/emidm.git
```

For differentiable models and optimization:

```bash
pip install "emidm[jax] @ git+https://github.com/OJWatson/emidm.git"
```

## Quick Start

### Stochastic SIR Model

```python
from emidm import simulate_sir

# Run a basic SIR simulation
results = simulate_sir(N=10000, I0=10, beta=0.3, gamma=0.1, T=100)
print(results[["t", "S", "I", "R"]].head())
```

### Time-Varying Reproduction Number

```python
from emidm import simulate_sir

# Define R(t) that decreases over time (e.g., due to interventions)
def R_t(t):
    return 2.5 if t < 30 else 1.2

results = simulate_sir(N=10000, I0=10, R_t=R_t, gamma=0.1, T=100)
```

### Differentiable Model & Calibration

```python
import jax
import jax.numpy as jnp
from emidm.diff import run_diff_sir, DiffConfig
from emidm.optim import optimize_params

# Generate synthetic data
key = jax.random.PRNGKey(0)
data = run_diff_sir(N_agents=200, I0=5, beta=0.35, gamma=0.2, T=30,
                    config=DiffConfig(tau=0.8, hard=True), key=key)

# Define loss function
def loss_fn(beta):
    pred = run_diff_sir(N_agents=200, I0=5, beta=beta, gamma=0.2, T=30,
                        config=DiffConfig(tau=0.8, hard=True), 
                        key=jax.random.PRNGKey(0))
    return jnp.mean((pred["I"] - data["I"]) ** 2)

# Fit beta using gradient descent
beta_hat, history = optimize_params(loss_fn=loss_fn, 
                                    init_params=jnp.array(0.15),
                                    n_steps=150, learning_rate=0.01)
print(f"Fitted β: {float(beta_hat):.3f}")  # Should recover ~0.35
```

## Documentation

Full documentation, tutorials, and API reference available at: **[ojwatson.github.io/emidm](https://ojwatson.github.io/emidm/)**

## License

`emidm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
