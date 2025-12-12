Development Guide
=================

This guide covers how to contribute to **emidm**, run tests, and understand the
project structure.

Project Structure
-----------------

::

   emidm/
   ├── src/emidm/           # Main package source code
   │   ├── __init__.py      # Package exports
   │   ├── sir.py           # Stochastic SIR model
   │   ├── safir.py         # Stochastic SAFIR/SEIR model
   │   ├── diff.py          # Differentiable models (JAX)
   │   ├── optim.py         # Optimization utilities
   │   ├── inference.py     # Bayesian inference (BlackJAX)
   │   └── plotting.py      # Visualization functions
   ├── tests/               # Test suite
   ├── docs/                # Sphinx documentation
   ├── examples/            # Example scripts
   └── pyproject.toml       # Package configuration

Installation for Development
----------------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/OJWatson/emidm.git
   cd emidm

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or: .venv\Scripts\activate  # Windows

   # Install with all development dependencies
   pip install -e ".[jax,docs]"

Running Tests
-------------

Tests are run with pytest:

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run with verbose output
   pytest tests/ -v

   # Run specific test file
   pytest tests/test_sir.py

   # Run with coverage
   pytest tests/ --cov=emidm --cov-report=html

Some tests require JAX and will be skipped if JAX is not installed.

Building Documentation
----------------------

Documentation is built with Sphinx:

.. code-block:: bash

   # Install docs dependencies
   pip install -e ".[docs]"

   # Build HTML docs
   cd docs
   sphinx-build -b html . _build/html

   # View locally
   open _build/html/index.html  # Mac
   # or: xdg-open _build/html/index.html  # Linux

Code Style
----------

The project follows standard Python conventions:

- Use type hints for function signatures
- Write docstrings in NumPy style
- Keep functions focused and modular
- Prefer keyword-only arguments for public APIs

Example function signature:

.. code-block:: python

   def simulate_sir(
       *,
       I0: int = 10,
       N: int = 1000,
       beta: float = 0.2,
       gamma: float = 0.1,
       T: int = 100,
       seed: int | None = None,
   ) -> pd.DataFrame:
       """Simulate a stochastic SIR model.

       Parameters
       ----------
       I0 : int, default 10
           Initial number of infected individuals.
       ...

       Returns
       -------
       pd.DataFrame
           Simulation results.
       """

Adding New Models
-----------------

To add a new epidemiological model:

1. **Stochastic version**: Add to ``sir.py`` or create a new module
2. **Differentiable version**: Add to ``diff.py`` using Gumbel-Softmax
3. **Tests**: Add tests in ``tests/``
4. **Documentation**: Update docstrings and add to user guide

Key considerations for differentiable models:

- Use JAX arrays (``jnp``) instead of NumPy
- Use ``jax.lax.scan`` for time loops (enables efficient compilation)
- Use ``_gumbel_softmax_bernoulli`` or ``_gumbel_softmax_categorical`` for stochastic events
- Ensure all operations are differentiable (no Python control flow on JAX values)

Example pattern:

.. code-block:: python

   def run_diff_new_model(*, params, key, config: DiffConfig):
       _require_jax()
       import jax
       import jax.numpy as jnp

       def step(carry, idx):
           state, key = carry
           # ... compute transition probabilities ...

           key, subkey = jax.random.split(key)
           # Use Gumbel-Softmax for stochastic transitions
           outcome = _gumbel_softmax_bernoulli(
               subkey, prob, tau=config.tau, hard=config.hard
           )
           # ... update state ...
           return (new_state, key), output

       # Use scan for efficient compilation
       (final_state, _), outputs = jax.lax.scan(step, init, indices)
       return outputs

Continuous Integration
----------------------

The project uses GitHub Actions for CI:

- **tests.yml**: Runs pytest on Python 3.10, 3.11, 3.12
- **docs.yml**: Builds Sphinx documentation
- **publish.yml**: Deploys docs to GitHub Pages on push to main
- **preview.yml**: Creates PR preview deployments

All CI checks must pass before merging PRs.

Release Process
---------------

1. Update version in ``src/emidm/__about__.py``
2. Update CHANGELOG (if present)
3. Create a git tag: ``git tag v0.x.x``
4. Push tag: ``git push origin v0.x.x``
5. CI will build and deploy documentation

Contributing
------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make changes and add tests
4. Run tests: ``pytest tests/``
5. Commit with clear messages
6. Push and create a Pull Request

Please ensure:

- All tests pass
- New code has tests
- Docstrings are complete
- Code follows project style
