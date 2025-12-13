# SPDX-FileCopyrightText: 2025-present OJWatson <o.watson15@imperial.ac.uk>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__

# Utility functions
from .utils import to_dataframe

# SIR models
from .sir import (
    run_sir,
    run_sir_replicates,
    simulate_sir,
    run_model_with_replicates,
    plot_model_outputs,
)

# SAFIR models
from .safir import run_safir, run_safir_replicates, simulate_safir

# Differentiable models
from .diff import (
    DiffConfig,
    run_diff_safir,
    run_diff_safir_replicates,
    run_diff_sir,
    run_diff_sir_replicates,
)

# Inference
from .inference import run_blackjax_nuts

# Optimization
from .optim import optimize_params, mse_loss, poisson_nll, gaussian_nll, make_sir_loss

# Plotting
from .plotting import (
    plot_sir,
    plot_safir,
    plot_model_comparison,
    plot_optimization_history,
    plot_replicates,
    plot_training_histories,
    sir_facet_plot,
)

# Sampling utilities
from .sampler import generate_lhs_samples

# Model registry
from .registry import (
    get_model,
    get_model_info,
    list_models,
    model_summary,
    register_model,
)

__all__ = [
    "__version__",
    # Utilities
    "to_dataframe",
    # SIR models
    "run_sir",
    "run_sir_replicates",
    "simulate_sir",
    "run_model_with_replicates",
    "plot_model_outputs",
    # SAFIR models
    "run_safir",
    "run_safir_replicates",
    "simulate_safir",
    # Differentiable models
    "DiffConfig",
    "run_diff_sir",
    "run_diff_sir_replicates",
    "run_diff_safir",
    "run_diff_safir_replicates",
    # Optimization & inference
    "optimize_params",
    "mse_loss",
    "poisson_nll",
    "gaussian_nll",
    "make_sir_loss",
    "run_blackjax_nuts",
    # Plotting
    "plot_sir",
    "plot_safir",
    "plot_model_comparison",
    "plot_optimization_history",
    "plot_replicates",
    "plot_training_histories",
    "sir_facet_plot",
    # Sampling
    "generate_lhs_samples",
    # Model registry
    "get_model",
    "get_model_info",
    "list_models",
    "model_summary",
    "register_model",
]
