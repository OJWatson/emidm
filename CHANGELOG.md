# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Model registry system for discovering and accessing models programmatically
  - `get_model()`, `list_models()`, `model_summary()`, `register_model()`, `get_model_info()`
- `run_diff_safir_replicates()` for running multiple replicates of differentiable SAFIR model
- `sir_facet_plot()` for faceted plotting with plotnine
- `to_dataframe()` utility for converting model outputs to pandas DataFrames
- Comprehensive JAX documentation:
  - JAX Fundamentals guide
  - Model Structure guide
  - End-to-End Differentiability guide
  - Writing JAX Models practical guide
- Contributing guidelines for new models
- Test coverage for plotting functions

### Changed
- **Breaking**: Standardized parameter names across all models:
  - `N_agents` → `N` (population size)
  - `key` → `seed` (random seed as integer)
  - `config` → `tau`, `hard` (Gumbel-Softmax parameters as direct arguments)
  - `n_replicates` → `reps` (number of replicates)
- All model functions now return dictionaries instead of DataFrames
- Updated all documentation and examples to use new parameter names

### Fixed
- Documentation toctree warnings

## [0.1.13] - 2025-01-XX

### Added
- Initial public release
- Stochastic SIR model with time-varying R(t)
- Age-structured SAFIR/SEIR model with contact matrices
- Differentiable SIR and SAFIR models using Gumbel-Softmax
- Optimization utilities with Optax integration
- BlackJAX NUTS sampler integration
- Plotting utilities for model outputs
- Latin Hypercube Sampling for parameter exploration
- Sphinx documentation with tutorials

[Unreleased]: https://github.com/OJWatson/emidm/compare/v0.1.13...HEAD
[0.1.13]: https://github.com/OJWatson/emidm/releases/tag/v0.1.13
