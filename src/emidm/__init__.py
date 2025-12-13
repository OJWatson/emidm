# SPDX-FileCopyrightText: 2025-present OJWatson <o.watson15@imperial.ac.uk>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__

from .sir import run_sir, run_model_with_replicates, simulate_sir
from .safir import run_safir, simulate_safir
from .diff import DiffConfig, run_diff_safir, run_diff_sir, run_diff_sir_replicates
from .inference import run_blackjax_nuts
from .optim import optimize_params, mse_loss, poisson_nll, gaussian_nll, make_sir_loss
from .plotting import (
    plot_sir,
    plot_safir,
    plot_model_comparison,
    plot_optimization_history,
    plot_replicates,
)

__all__ = [
    "__version__",
    # SIR models
    "run_sir",
    "simulate_sir",
    "run_model_with_replicates",
    # SAFIR models
    "run_safir",
    "simulate_safir",
    # Differentiable models
    "DiffConfig",
    "run_diff_sir",
    "run_diff_safir",
    "run_diff_sir_replicates",
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
]
