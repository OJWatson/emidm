"""Utility functions for emidm.

This module provides utility functions for converting between data formats
and other common operations used across the package.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def to_dataframe(data: dict, include_replicate: bool = True) -> "pd.DataFrame":
    """Convert model output dictionary to a pandas DataFrame.

    This utility function converts the standard dict output from emidm models
    to a pandas DataFrame for easier data manipulation and analysis.

    Parameters
    ----------
    data : dict
        Model output dictionary with keys like 't', 'S', 'I', 'R', etc.
        Arrays can be 1D (single run) or 2D (multiple replicates where
        first dimension is replicates).
    include_replicate : bool, default True
        If True and data contains 2D arrays (replicates), include a
        'replicate' column in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with time column and compartment columns.
        If input has replicates (2D arrays), the DataFrame will be in
        long format with a 'replicate' column.

    Examples
    --------
    >>> from emidm import run_sir, to_dataframe
    >>> result = run_sir(N=1000, I0=10, beta=0.3, gamma=0.1, T=50)
    >>> df = to_dataframe(result)
    >>> df.head()
       t    S   I   R
    0  0  990  10   0
    1  1  988  11   1
    ...

    >>> # With replicates
    >>> from emidm import run_sir_replicates, to_dataframe
    >>> result = run_sir_replicates(N=1000, I0=10, beta=0.3, gamma=0.1, T=50, reps=10)
    >>> df = to_dataframe(result)
    >>> df.head()
       t  replicate    S   I   R
    0  0          0  990  10   0
    ...
    """
    import pandas as pd

    # Convert any JAX arrays to numpy
    converted = {}
    for k, v in data.items():
        try:
            converted[k] = np.asarray(v)
        except Exception:
            converted[k] = v

    # Determine if we have replicates (2D arrays for compartments)
    time_key = "t" if "t" in converted else "day"
    t = converted.get(time_key, None)

    if t is None:
        raise ValueError("Data must contain 't' or 'day' key for time")

    # Check dimensionality of non-time arrays
    sample_key = None
    for k in converted:
        if k not in (time_key,) and isinstance(converted[k], np.ndarray):
            sample_key = k
            break

    if sample_key is None:
        # No array data besides time
        return pd.DataFrame(converted)

    sample_arr = converted[sample_key]

    if sample_arr.ndim == 1:
        # Single run - simple conversion
        return pd.DataFrame(converted)

    elif sample_arr.ndim == 2:
        # Multiple replicates: shape is (reps, time_steps)
        n_reps, n_times = sample_arr.shape

        rows = []
        for rep in range(n_reps):
            row_data = {time_key: t}
            if include_replicate:
                row_data["replicate"] = rep
            for k, v in converted.items():
                if k == time_key:
                    continue
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    row_data[k] = v[rep, :]
                else:
                    row_data[k] = v
            rows.append(pd.DataFrame(row_data))

        return pd.concat(rows, ignore_index=True)

    else:
        raise ValueError(f"Unexpected array dimensionality: {sample_arr.ndim}")


def dict_to_dataframe(data: dict, include_replicate: bool = True) -> "pd.DataFrame":
    """Alias for to_dataframe for backwards compatibility.

    See :func:`to_dataframe` for full documentation.
    """
    return to_dataframe(data, include_replicate=include_replicate)
