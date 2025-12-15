from __future__ import annotations

import numpy as np
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _rt_at_time(R_t: Sequence[float] | Callable[[int], float], t: int) -> float:
    if callable(R_t):
        return float(R_t(t))
    return float(R_t[t])


def simulate_sir(
    *,
    N: int = 1000,
    I0: int = 10,
    R0_init: int = 0,
    beta: float = 0.2,
    gamma: float = 0.1,
    R_t: Sequence[float] | Callable[[int], float] | None = None,
    T: int = 100,
    dt: int = 1,
    reps: int = 1,
    seed: int | None = None,
) -> dict:
    """Simulate a stochastic SIR model with optional time-varying reproduction number.

    This is the detailed SIR simulation function supporting multiple replicates and
    time-varying transmission rates.

    Parameters
    ----------
    N : int, default 1000
        Total population size.
    I0 : int, default 10
        Initial number of infected individuals.
    R0_init : int, default 0
        Initial number of recovered individuals (not the reproduction number).
    beta : float, default 0.2
        Transmission rate (used if R_t is None).
    gamma : float, default 0.1
        Recovery rate.
    R_t : Sequence or Callable, optional
        Time-varying reproduction number. Can be:
        - A sequence of length T+1 with R_t values at each time step
        - A callable R_t(t) returning the reproduction number at time t
        If provided, beta is computed as R_t * gamma at each time step.
    T : int, default 100
        Total simulation time.
    dt : int, default 1
        Time step size.
    reps : int, default 1
        Number of stochastic replicates to run.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - 't': time points array
        - 'S': susceptible counts (shape: (reps, T+1) if reps > 1, else (T+1,))
        - 'I': infected counts
        - 'R': recovered counts
        - 'replicate': replicate indices (only if reps > 1)

    Examples
    --------
    >>> from emidm import simulate_sir
    >>>
    >>> # Basic simulation
    >>> result = simulate_sir(N=10000, I0=10, beta=0.3, gamma=0.1, T=100)
    >>> result["I"].max()  # Peak infections
    >>>
    >>> # With time-varying R_t
    >>> import numpy as np
    >>> R_t = np.concatenate([np.full(50, 2.5), np.full(51, 0.8)])  # Intervention at t=50
    >>> result = simulate_sir(N=10000, I0=10, R_t=R_t, gamma=0.1, T=100)
    >>>
    >>> # Multiple replicates
    >>> result = simulate_sir(N=10000, I0=10, beta=0.3, gamma=0.1, T=100, reps=100, seed=42)
    >>> result["I"].shape  # (100, 101)
    """
    I0 = int(I0)
    R0_init = int(R0_init)
    N = int(N)
    T = int(T)
    dt = int(dt)
    reps = int(reps)

    n_times = (T // dt) + 1
    t_arr = np.arange(0, T + 1, dt)

    S_all = np.zeros((reps, n_times), dtype=np.int64)
    I_all = np.zeros((reps, n_times), dtype=np.int64)
    R_all = np.zeros((reps, n_times), dtype=np.int64)

    for rep in range(reps):
        rep_seed = None if seed is None else seed + rep
        rng = np.random.default_rng(rep_seed)

        I = I0
        S = N - I0 - R0_init
        R = R0_init

        for idx, t in enumerate(t_arr):
            S_all[rep, idx] = S
            I_all[rep, idx] = I
            R_all[rep, idx] = R

            if idx == n_times - 1:
                break

            beta_t = beta if R_t is None else _rt_at_time(R_t, t) * gamma
            pSI = 1 - np.exp(-beta_t * I / N * dt)
            pIR = 1 - np.exp(-gamma * dt)

            n_SE = rng.binomial(S, pSI)
            n_IR = rng.binomial(I, pIR)

            S = S - n_SE
            I = I + n_SE - n_IR
            R = R + n_IR

    # Return 1D arrays if single replicate, 2D if multiple
    if reps == 1:
        return {
            "t": t_arr,
            "S": S_all[0],
            "I": I_all[0],
            "R": R_all[0],
        }
    else:
        return {
            "t": t_arr,
            "S": S_all,
            "I": I_all,
            "R": R_all,
        }


def run_sir(
    *,
    N: int = 1000,
    I0: int = 10,
    R0_init: int = 0,
    beta: float = 0.2,
    gamma: float = 0.1,
    R_t: Sequence[float] | Callable[[int], float] | None = None,
    T: int = 100,
    dt: int = 1,
    seed: int | None = None,
) -> dict:
    """Run a stochastic SIR model simulation.

    This is the primary interface for running a single SIR simulation.

    Parameters
    ----------
    N : int, default 1000
        Total population size.
    I0 : int, default 10
        Initial number of infected individuals.
    R0_init : int, default 0
        Initial number of recovered individuals (not the reproduction number).
    beta : float, default 0.2
        Transmission rate (used if R_t is None).
    gamma : float, default 0.1
        Recovery rate.
    R_t : Sequence or Callable, optional
        Time-varying reproduction number.
    T : int, default 100
        Total simulation time.
    dt : int, default 1
        Time step size.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys 't', 'S', 'I', 'R' as numpy arrays.

    Examples
    --------
    >>> from emidm import run_sir
    >>> result = run_sir(N=1000, I0=10, beta=0.3, gamma=0.1, T=50)
    >>> result["I"].max()  # Peak infections
    """
    return simulate_sir(
        N=N,
        I0=I0,
        R0_init=R0_init,
        beta=beta,
        gamma=gamma,
        R_t=R_t,
        T=T,
        dt=dt,
        reps=1,
        seed=seed,
    )


def run_sir_replicates(
    *,
    N: int = 1000,
    I0: int = 10,
    R0_init: int = 0,
    beta: float = 0.2,
    gamma: float = 0.1,
    R_t: Sequence[float] | Callable[[int], float] | None = None,
    T: int = 100,
    dt: int = 1,
    reps: int = 10,
    seed: int | None = None,
) -> dict:
    """Run multiple replicates of the stochastic SIR model.

    Parameters
    ----------
    N : int, default 1000
        Total population size.
    I0 : int, default 10
        Initial number of infected individuals.
    R0_init : int, default 0
        Initial number of recovered individuals.
    beta : float, default 0.2
        Transmission rate (used if R_t is None).
    gamma : float, default 0.1
        Recovery rate.
    R_t : Sequence or Callable, optional
        Time-varying reproduction number.
    T : int, default 100
        Total simulation time.
    dt : int, default 1
        Time step size.
    reps : int, default 10
        Number of stochastic replicates to run.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - 't': time points array (shape: (T+1,))
        - 'S': susceptible counts (shape: (reps, T+1))
        - 'I': infected counts (shape: (reps, T+1))
        - 'R': recovered counts (shape: (reps, T+1))

    Examples
    --------
    >>> from emidm import run_sir_replicates
    >>> result = run_sir_replicates(N=1000, I0=10, beta=0.3, gamma=0.1, T=50, reps=100)
    >>> result["I"].mean(axis=0)  # Mean infection trajectory
    """
    return simulate_sir(
        N=N,
        I0=I0,
        R0_init=R0_init,
        beta=beta,
        gamma=gamma,
        R_t=R_t,
        T=T,
        dt=dt,
        reps=reps,
        seed=seed,
    )


def run_model_with_replicates(
    model: callable = None, reps: int = 10, seed: int | None = None, **kwargs
) -> "pd.DataFrame":
    """Run a model multiple times and return results as a DataFrame.

    .. deprecated::
        This function is deprecated. Use model-specific replicate functions
        (e.g., run_sir_replicates) with to_dataframe() instead.

    Parameters
    ----------
    model : callable
        The model function to run for each replicate.
    reps : int
        The number of times to run the model.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs :
        Arbitrary keyword arguments to pass to the model function.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the concatenated results of each replicate,
        with an additional column indicating the replicate number.
    """
    import pandas as pd
    from .utils import to_dataframe

    if model is None:
        model = run_sir

    results = []
    for rep in range(reps):
        rep_seed = None if seed is None else seed + rep
        result = model(**kwargs, seed=rep_seed)
        if isinstance(result, dict):
            df = to_dataframe(result)
            df["replicate"] = rep
        else:
            df = result.copy()
            df["replicate"] = rep
        results.append(df)
    return pd.concat(results, axis=0, ignore_index=True)


def plot_model_outputs(
    data,
    x: str = "t",
    replicate: str = "replicate",
    columns: list = None,
    show: bool = True,
):
    """Plot the outputs of a model run.

    Parameters
    ----------
    data : dict or pd.DataFrame
        The model output (dict from model functions or DataFrame).
    x : str
        The name of the column to use as the x-axis.
    replicate : str
        The name of the column to use as the replicate identifier.
    columns : list
        The names of the columns to plot. If None, will plot S, I, R.
    show : bool
        Whether to show the plot as well as returning it.

    Returns
    -------
    plotnine.ggplot
        The plot object.
    """
    try:
        import plotnine as p9
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "plot_model_outputs requires 'plotnine' to be installed. "
            "Install it via `pip install plotnine` (or install emidm with its dependencies)."
        ) from e

    from .utils import to_dataframe

    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        df = to_dataframe(data)
    else:
        df = data.copy()

    if columns is None:
        # Default to S, I, R columns
        columns = [c for c in ["S", "I", "R", "E", "D"] if c in df.columns]

    # Check if we have replicates
    if replicate in df.columns:
        # Reshape dataframe into tidy long-format
        df_long = df.melt(
            id_vars=[x, replicate],
            value_vars=columns,
            var_name="Compartment",
            value_name="Value",
        )

        # Add unique identifier for group plotting
        df_long = df_long.assign(
            uid=df_long["Compartment"] + df_long[replicate].astype(str)
        )

        # Plot: color by compartment, lines grouped by replicate
        p = p9.ggplot(
            df_long,
            p9.aes(x=x, y="Value", group="uid", color="Compartment"),
        ) + p9.geom_line(alpha=0.7)
    else:
        # Single run - simpler plot
        df_long = df.melt(
            id_vars=[x],
            value_vars=columns,
            var_name="Compartment",
            value_name="Value",
        )
        p = p9.ggplot(
            df_long,
            p9.aes(x=x, y="Value", color="Compartment"),
        ) + p9.geom_line(alpha=0.9, size=1)

    # Explicitly plot
    if show:
        p9.ggplot.show(p)

    # Return the plot object
    return p
