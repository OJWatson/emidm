import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_line, facet_grid


def run_sir(
    I0: int = 10,
    R0: int = 0,
    N: int = 1000,
    beta: float = 0.2,
    gamma: float = 0.1,
    T: int = 100,
    dt: int = 1,
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

    # initialise compartments
    res = pd.DataFrame(
        {
            "t": np.arange(start=0, stop=T + 1, step=dt),
            "N": N,
            "S": N - I0,
            "I": I0,
            "R": 0,
        }
    )

    # create our initial vars
    I = I0
    S = N - I
    R = 0

    # enumerate timesteps
    for step in res.index[1:]:

        # define parameters for this timestep
        pSI = 1 - np.exp(-beta * I / N * dt)
        pIR = 1 - np.exp(-gamma * dt)

        # calculate compartment changes
        n_SE = np.random.binomial(S, pSI)
        n_IR = np.random.binomial(I, pIR)

        # update compartments
        S = S - n_SE
        I = I + n_SE - n_IR
        R = R + n_IR

        # store results
        res.loc[step, "S"] = S
        res.loc[step, "I"] = I
        res.loc[step, "R"] = R

    return res


def run_model_with_replicates(
    model: callable = run_sir, reps: int = 10, **kwargs
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
    results = [
        model(**kwargs).assign(replicate=replicate_index)
        for replicate_index in range(reps)
    ]
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
    if columns is None:
        columns = df.columns.drop([x])

    # Reshape dataframe into tidy long-format
    df_long = df.melt(
        id_vars=[x, "replicate"],
        value_vars=columns,
        var_name="Compartment",
        value_name="Value",
    )

    # Add unique identifier for group plotting
    df_long = df_long.assign(
        uid=df_long["Compartment"] + df_long["replicate"].astype(str)
    )

    # Plot: color by compartment, lines grouped by replicate
    p = ggplot(
        df_long,
        aes(x="t", y="Value", group="uid", color="Compartment"),
    ) + geom_line(alpha=0.7)

    # Explicitly plot
    if show:
        ggplot.show(p)

    # Return the plot object
    return p
