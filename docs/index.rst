emidm
=====

**Differentiable Epidemiological Modelling in Python**

``emidm`` provides stochastic and differentiable infectious disease models built with JAX,
enabling gradient-based calibration and Bayesian inference for epidemiological parameters.

Key Features
------------

- **Stochastic Models**: SIR and age-structured SAFIR/SEIR models with contact matrices
- **Differentiable Models**: JAX-based implementations using Gumbel-Softmax for gradient-based optimization  
- **Optimization**: Built-in Optax integration for parameter calibration
- **Bayesian Inference**: BlackJAX scaffolding for MCMC sampling

Quick Example
-------------

.. code-block:: python

   from emidm import run_sir_simulation, to_dataframe
   
   # Run a basic SIR simulation
   results = run_sir_simulation(N=10000, I0=10, beta=0.3, gamma=0.1, T=100)
   df = to_dataframe(results)
   print(df[["t", "S", "I", "R"]].head())

For differentiable models and gradient-based calibration:

.. code-block:: python

   import jax
   from emidm import run_diff_sir_simulation, DiffConfig
   from emidm.optim import optimize_params
   
   # Fit parameters using gradients
   key = jax.random.PRNGKey(0)
   beta_hat, history = optimize_params(
       loss_fn=my_loss_fn,
       init_params=jax.numpy.array(0.15),
       n_steps=150
   )

Getting Started
---------------

Install with pip:

.. code-block:: bash

   pip install git+https://github.com/OJWatson/emidm.git

For JAX support (differentiable models):

.. code-block:: bash

   pip install "emidm[jax] @ git+https://github.com/OJWatson/emidm.git"

.. toctree::
   :maxdepth: 2
   :caption: Contents

   user_guide
   tutorials
   api
   development
   slides
