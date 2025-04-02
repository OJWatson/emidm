import pandas as pd
from scipy.stats import qmc


def generate_lhs_samples(
    param_ranges: dict, n_samples: int, seed: int = None
) -> pd.DataFrame:
    """
    Generate Latin Hypercube Samples as a DataFrame for given parameter ranges.

    Args:
        param_ranges (dict): Dictionary with parameter names as keys and [min, max] lists as values.
        n_samples (int): Number of samples to generate.
        seed (int, optional): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame of sampled parameter values.
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    sample_unit = sampler.random(n=n_samples)

    samples = {}
    for i, param in enumerate(param_ranges):
        lower, upper = param_ranges[param]
        samples[param] = lower + sample_unit[:, i] * (upper - lower)

    return pd.DataFrame(samples)
