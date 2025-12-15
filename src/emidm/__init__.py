# SPDX-FileCopyrightText: 2025-present OJWatson <o.watson15@imperial.ac.uk>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__

# Utility functions
from .utils import to_dataframe

# SIR models
from .sir import (
    SIRModel,
    SIRState,
    run_sir_simulation,
)

# SAFIR models
from .safir import (
    SAFIRModel,
    SAFIRState,
    run_safir_simulation,
)

# Differentiable models
from .diff import (
    DiffConfig,
    make_diff_safir_model,
    run_diff_safir_simulation,
    run_diff_sir_simulation,
)

# Inference
from .inference import run_blackjax_nuts

# Optimization
from .optim import optimize_params, mse_loss, poisson_nll, gaussian_nll, binomial_nll, make_sir_loss

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
    "SIRModel",
    "SIRState",
    "run_sir_simulation",
    # SAFIR models
    "SAFIRModel",
    "SAFIRState",
    "run_safir_simulation",
    # Differentiable models
    "DiffConfig",
    "make_diff_safir_model",
    "run_diff_sir_simulation",
    "run_diff_safir_simulation",
    # Optimization & inference
    "optimize_params",
    "mse_loss",
    "poisson_nll",
    "gaussian_nll",
    "binomial_nll",
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
