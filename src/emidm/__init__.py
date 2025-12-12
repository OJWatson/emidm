# SPDX-FileCopyrightText: 2025-present OJWatson <o.watson15@imperial.ac.uk>
#
# SPDX-License-Identifier: MIT

from .__about__ import __version__

from .sir import run_sir, run_model_with_replicates, simulate_sir
from .safir import run_safir, simulate_safir
from .diff import DiffConfig, run_diff_safir, run_diff_safir_simple, run_diff_sir
from .inference import run_blackjax_nuts
from .optim import optimize_params

__all__ = [
    "__version__",
    "run_sir",
    "simulate_sir",
    "run_model_with_replicates",
    "run_safir",
    "simulate_safir",
    "DiffConfig",
    "run_diff_sir",
    "run_diff_safir",
    "run_diff_safir_simple",
    "optimize_params",
    "run_blackjax_nuts",
]
