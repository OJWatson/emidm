import pandas as pd
import numpy as np
from collections.abc import Callable, Sequence


def _rt_at_time(R_t: Sequence[float] | Callable[[int], float], t: int) -> float:
    if callable(R_t):
        return float(R_t(t))
    return float(R_t[t])


def simulate_sir(
    *,
    I0: int = 10,
    R0: int = 0,
    N: int = 1000,
    beta: float = 0.2,
    gamma: float = 0.1,
    R_t: Sequence[float] | Callable[[int], float] | None = None,
    T: int = 100,
    dt: int = 1,
    n_replicates: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate a stochastic SIR model with optional time-varying reproduction number.

    This is the main SIR simulation function supporting multiple replicates and
    time-varying transmission rates.

    Parameters
    ----------
    I0 : int, default 10
        Initial number of infected individuals.
    R0 : int, default 0
        Initial number of recovered individuals (not the reproduction number).
    N : int, default 1000
        Total population size.
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
    n_replicates : int, default 1
        Number of stochastic replicates to run.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: t, replicate, N, I0, beta, beta_t, gamma, S, I, R.

    Examples
    --------
    >>> from emidm import simulate_sir
    >>>
    >>> # Basic simulation
    >>> df = simulate_sir(N=10000, I0=10, beta=0.3, gamma=0.1, T=100)
    >>>
    >>> # With time-varying R_t
    >>> import numpy as np
    >>> R_t = np.concatenate([np.full(50, 2.5), np.full(51, 0.8)])  # Intervention at t=50
    >>> df = simulate_sir(N=10000, I0=10, R_t=R_t, gamma=0.1, T=100)
    >>>
    >>> # Multiple replicates
    >>> df = simulate_sir(N=10000, I0=10, beta=0.3, gamma=0.1, T=100, n_replicates=100, seed=42)
    """
    I0 = int(I0)
    R0 = int(R0)
    N = int(N)
    T = int(T)
    dt = int(dt)
    n_replicates = int(n_replicates)

    rows: list[dict] = []
    for replicate_index in range(n_replicates):
        replicate_seed = None if seed is None else seed + replicate_index
        rng = np.random.default_rng(replicate_seed)

        I = I0
        S = N - I0 - R0
        R = R0

        for t in range(0, T + 1, dt):
            beta_t = beta if R_t is None else _rt_at_time(R_t, t) * gamma
            rows.append(
                {
                    "t": t,
                    "replicate": replicate_index,
                    "N": N,
                    "I0": I0,
                    "beta": beta,
                    "beta_t": float(beta_t),
                    "gamma": gamma,
                    "S": S,
                    "I": I,
                    "R": R,
                }
            )

            if t == T:
                break

            pSI = 1 - np.exp(-beta_t * I / N * dt)
            pIR = 1 - np.exp(-gamma * dt)

            n_SE = rng.binomial(S, pSI)
            n_IR = rng.binomial(I, pIR)

            S = S - n_SE
            I = I + n_SE - n_IR
            R = R + n_IR

    return pd.DataFrame(rows)


def run_sir(
    I0: int = 10,
    R0: int = 0,
    N: int = 1000,
    beta: float = 0.2,
    gamma: float = 0.1,
    T: int = 100,
    dt: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:

    # set up our results
    """
    Run an SIR model from the given parameters

    Parameters
    ----------
    I0 : int
        The initial number of infected individuals
    R0 : int
        The initial number of recovered individuals
    N : int
        The total population size
    beta : float
        The transmission rate
    gamma : float
        The recovery rate
    T : int
        The total time
    dt : int
        The time step

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns "t", "N", "S", "I", and "R"
        giving the number of susceptible, infected, and recovered individuals
        at each time step
    """

    # Integer convert
    I0 = int(I0)
    R0 = int(R0)
    N = int(N)
    T = int(T)

    df = simulate_sir(
        I0=I0,
        R0=R0,
        N=N,
        beta=beta,
        gamma=gamma,
        T=T,
        dt=dt,
        n_replicates=1,
        seed=seed,
    )
    return df.drop(columns=["replicate", "I0", "beta", "beta_t", "gamma"]).reset_index(
        drop=True
    )


def run_model_with_replicates(
    model: callable = run_sir, reps: int = 10, seed: int | None = None, **kwargs
) -> pd.DataFrame:
    """
    Runs the specified model multiple times and returns a concatenated DataFrame
    with results from each replicate labeled.

    Parameters
    ----------
    model : callable
        The model function to run for each replicate.
    reps : int
        The number of times to run the model.
    **kwargs :
        Arbitrary keyword arguments to pass to the model function.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the concatenated results of each replicate,
        with an additional column indicating the replicate number.
    """
    results = []
    for replicate_index in range(reps):
        replicate_seed = None if seed is None else seed + replicate_index
        results.append(
            model(**kwargs, seed=replicate_seed).assign(replicate=replicate_index)
        )
    return pd.concat(results, axis=0)


def plot_model_outputs(
    df: pd.DataFrame,
    x: str = "t",
    replicate: str = "replicate",
    columns: list = None,
    show: bool = True,
):
    """
    Plot the outputs of a model run.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the model output.
    x : str
        The name of the column to use as the x-axis.
    replicate : str
        The name of the column to use as the replicate identifier.
    columns : list
        The names of the columns to plot. If None, will plot all columns
        except x and replicate.
    show : bool
        Logical for whether to show the plot as well as returning
    """
    try:
        import plotnine as p9
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "plot_model_outputs requires 'plotnine' to be installed. "
            "Install it via `pip install plotnine` (or install emidm with its dependencies)."
        ) from e

    if columns is None:
        columns = df.columns.drop([x, replicate])

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
        p9.aes(x="t", y="Value", group="uid", color="Compartment"),
    ) + p9.geom_line(alpha=0.7)

    # Explicitly plot
    if show:
        p9.ggplot.show(p)

    # Return the plot object
    return p
