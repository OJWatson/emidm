from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.stats import qmc

if TYPE_CHECKING:
    import pandas as pd


def generate_lhs_samples(
    param_ranges: dict, n_samples: int, seed: int | None = None
) -> "pd.DataFrame":
    """Generate Latin Hypercube Samples for parameter space exploration.

    Latin Hypercube Sampling (LHS) provides better coverage of the parameter
    space than random sampling, making it ideal for sensitivity analysis
    and surrogate model training.

    Parameters
    ----------
    param_ranges : dict
        Dictionary with parameter names as keys and [min, max] lists as values.
        Example: {"beta": [0.1, 0.5], "gamma": [0.05, 0.2]}
    n_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each parameter, containing n_samples rows.

    Examples
    --------
    >>> from emidm import generate_lhs_samples
    >>> param_ranges = {"beta": [0.1, 0.5], "gamma": [0.05, 0.2]}
    >>> samples = generate_lhs_samples(param_ranges, n_samples=100, seed=42)
    >>> samples.head()
           beta     gamma
    0  0.123456  0.087654
    ...
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    sample_unit = sampler.random(n=n_samples)

    import pandas as pd

    samples = {}
    for i, param in enumerate(param_ranges):
        lower, upper = param_ranges[param]
        samples[param] = lower + sample_unit[:, i] * (upper - lower)

    return pd.DataFrame(samples)
