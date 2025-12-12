User Guide
==========

This guide provides an overview of **emidm** and explains the key concepts behind
differentiable epidemiological modeling.

Overview
--------

**emidm** provides two types of epidemiological models:

1. **Stochastic models** (NumPy-based): Traditional discrete-event simulations
2. **Differentiable models** (JAX-based): Models that support automatic differentiation

The differentiable models enable gradient-based optimization for parameter calibration,
which is significantly faster than derivative-free methods for high-dimensional problems.

Models
------

SIR Model
^^^^^^^^^

The SIR (Susceptible-Infected-Recovered) model is the simplest compartmental model:

- **S** → **I**: Susceptible individuals become infected at rate β·I/N
- **I** → **R**: Infected individuals recover at rate γ

.. code-block:: python

   from emidm import simulate_sir, run_diff_sir

   # Stochastic SIR
   df = simulate_sir(N=10000, I0=10, beta=0.3, gamma=0.1, T=100)

   # Differentiable SIR
   result = run_diff_sir(N_agents=200, I0=5, beta=0.3, gamma=0.1, T=50)

SAFIR Model
^^^^^^^^^^^

The SAFIR model is an age-structured SEIR model with hospitalization and death:

- **S** → **E1** → **E2**: Susceptible → Exposed (two stages)
- **E2** → **Iasy** | **Imild** | **Icase**: Exposed → Asymptomatic, Mild, or Severe infection
- **Iasy/Imild** → **R**: Recovery
- **Icase** → **R** | **D**: Recovery or Death

.. code-block:: python

   import numpy as np
   from emidm import simulate_safir, run_diff_safir

   population = np.array([1000, 2000, 1500, 1000])  # 4 age groups
   contact_matrix = np.array([
       [3.0, 1.5, 0.5, 0.3],
       [1.5, 2.5, 1.0, 0.5],
       [0.5, 1.0, 2.0, 1.0],
       [0.3, 0.5, 1.0, 1.5],
   ])

   # Stochastic SAFIR
   df = simulate_safir(
       population=population,
       contact_matrix=contact_matrix,
       R0=2.5,
       time_horizon=200,
   )

   # Differentiable SAFIR
   result = run_diff_safir(
       population=population,
       contact_matrix=contact_matrix,
       R0=2.5,
       time_horizon=100,
   )

Time-Varying Reproduction Number
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both SIR and SAFIR models support time-varying R(t):

.. code-block:: python

   import numpy as np
   from emidm import simulate_sir

   # R_t drops from 2.5 to 0.8 at day 50 (e.g., intervention)
   R_t = np.concatenate([np.full(50, 2.5), np.full(51, 0.8)])

   df = simulate_sir(N=10000, I0=10, R_t=R_t, gamma=0.1, T=100)

Differentiable Modeling
-----------------------

Why Differentiable Models?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Traditional epidemiological models use discrete stochastic events (e.g., binomial draws
for infections). These are not differentiable, making gradient-based optimization impossible.

**emidm** uses the **Gumbel-Softmax** reparameterization trick to make these discrete
events differentiable while preserving the stochastic nature of the model.

The Gumbel-Softmax Trick
^^^^^^^^^^^^^^^^^^^^^^^^

The Gumbel-Softmax (or Concrete) distribution provides a continuous relaxation of
discrete categorical distributions. For a Bernoulli random variable with probability p:

1. Sample Gumbel noise: g ~ Gumbel(0, 1)
2. Compute soft sample: y = softmax((log(p) + g) / τ)
3. For hard samples: use straight-through estimator (argmax in forward, softmax in backward)

The **temperature** parameter τ controls the discreteness:

- **Low τ** (e.g., 0.1): Nearly discrete outputs
- **High τ** (e.g., 2.0): Smoother, more continuous outputs

.. code-block:: python

   from emidm import run_diff_sir, DiffConfig

   # Low temperature: more discrete
   result_discrete = run_diff_sir(
       N_agents=100, I0=5, beta=0.3, gamma=0.1, T=50,
       config=DiffConfig(tau=0.1, hard=True),
   )

   # Higher temperature: smoother gradients
   result_smooth = run_diff_sir(
       N_agents=100, I0=5, beta=0.3, gamma=0.1, T=50,
       config=DiffConfig(tau=1.0, hard=True),
   )

The ``hard`` parameter controls whether to use the straight-through estimator:

- ``hard=True``: Forward pass uses discrete samples, backward pass uses continuous gradients
- ``hard=False``: Both passes use continuous (soft) samples

Parameter Calibration
---------------------

The main advantage of differentiable models is efficient parameter calibration using
gradient descent.

Basic Calibration
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from emidm import run_diff_sir, DiffConfig, optimize_params, mse_loss

   # Generate synthetic "observed" data
   true_beta = 0.35
   key = jax.random.PRNGKey(42)
   observed = run_diff_sir(
       N_agents=200, I0=5, beta=true_beta, gamma=0.1, T=50,
       config=DiffConfig(tau=0.5, hard=True),
       key=key,
   )

   # Define loss function
   def loss_fn(beta):
       pred = run_diff_sir(
           N_agents=200, I0=5, beta=beta, gamma=0.1, T=50,
           config=DiffConfig(tau=0.5, hard=True),
           key=jax.random.PRNGKey(0),  # Fixed key for reproducibility
       )
       return mse_loss(pred["I"], observed["I"])

   # Optimize
   beta_init = jnp.array(0.15)  # Initial guess
   beta_hat, history = optimize_params(
       loss_fn=loss_fn,
       init_params=beta_init,
       n_steps=150,
       learning_rate=0.01,
   )

   print(f"True β: {true_beta}, Estimated β: {float(beta_hat):.3f}")

Using the Loss Function Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**emidm** provides several loss functions:

- ``mse_loss``: Mean squared error
- ``poisson_nll``: Poisson negative log-likelihood (for count data)
- ``gaussian_nll``: Gaussian negative log-likelihood
- ``make_sir_loss``: Helper to create loss functions for SIR models

.. code-block:: python

   from emidm import make_sir_loss, run_diff_sir, DiffConfig

   # Create a loss function automatically
   loss_fn = make_sir_loss(
       observed_I=observed["I"],
       model_fn=run_diff_sir,
       model_kwargs={
           "N_agents": 200,
           "I0": 5,
           "gamma": 0.1,
           "T": 50,
           "config": DiffConfig(tau=0.5, hard=True),
           "key": jax.random.PRNGKey(0),
       },
       loss_type="mse",  # or "poisson", "gaussian"
   )

Plotting
--------

**emidm** provides plotting functions for visualizing model outputs:

.. code-block:: python

   from emidm import (
       simulate_sir,
       run_diff_sir,
       plot_sir,
       plot_safir,
       plot_model_comparison,
       plot_optimization_history,
       plot_replicates,
   )

   # Plot SIR trajectories (works with both dict and DataFrame)
   df = simulate_sir(N=10000, I0=10, beta=0.3, gamma=0.1, T=100)
   fig, ax = plot_sir(df, title="SIR Epidemic")

   # Plot differentiable model output
   result = run_diff_sir(N_agents=200, I0=5, beta=0.3, gamma=0.1, T=50)
   fig, ax = plot_sir(result, title="Differentiable SIR")

   # Compare observed vs fitted
   fig, ax = plot_model_comparison(observed, fitted, compartment="I")

   # Plot optimization history
   fig, axes = plot_optimization_history(history, true_params=0.35)

   # Plot multiple replicates with confidence intervals
   df_reps = simulate_sir(N=10000, I0=10, beta=0.3, gamma=0.1, T=100, n_replicates=50)
   fig, ax = plot_replicates(df_reps, compartment="I", show_ci=True)

Bayesian Inference
------------------

For uncertainty quantification, **emidm** provides integration with BlackJAX for
Hamiltonian Monte Carlo (HMC) sampling:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from emidm import run_diff_sir, DiffConfig, mse_loss, run_blackjax_nuts

   # Define log-density (negative loss + prior)
   def log_density(beta):
       # Likelihood
       pred = run_diff_sir(
           N_agents=200, I0=5, beta=beta, gamma=0.1, T=50,
           config=DiffConfig(tau=0.5, hard=True),
           key=jax.random.PRNGKey(0),
       )
       log_lik = -mse_loss(pred["I"], observed["I"])

       # Prior: beta ~ Normal(0.3, 0.1)
       log_prior = -0.5 * ((beta - 0.3) / 0.1) ** 2

       return log_lik + log_prior

   # Run NUTS sampler
   samples = run_blackjax_nuts(
       logdensity_fn=log_density,
       initial_position=jnp.array(0.25),
       rng_seed=42,
       num_warmup=500,
       num_samples=1000,
   )

   print(f"Posterior mean: {samples.mean():.3f}")
   print(f"Posterior std: {samples.std():.3f}")

References
----------

- Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR.
- Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). The Concrete Distribution. ICLR.
- Quera-Bofarull, A., et al. (2023). BlackBIRDS: Black-Box Inference foR Differentiable Simulators. JOSS.
