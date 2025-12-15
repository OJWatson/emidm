"""Plotting utilities for epidemiological model outputs.

This module provides publication-ready plotting functions using matplotlib and seaborn
for visualizing SIR and SAFIR model outputs, including both stochastic and differentiable
model results.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import to_dataframe as _to_dataframe_util

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _to_dataframe(data, model_type: str = "sir") -> pd.DataFrame:
    """Convert model output (dict or DataFrame) to a standardized DataFrame.

    Parameters
    ----------
    data : dict or pd.DataFrame
        Model output. If dict (from differentiable models), converts to DataFrame.
    model_type : str
        Either "sir" or "safir" to determine expected columns.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with time column 't' or 'day'.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()

    # Use the centralized to_dataframe utility
    if isinstance(data, dict):
        return _to_dataframe_util(data, include_replicate=False)

    raise TypeError(f"Expected dict or DataFrame, got {type(data)}")


def plot_sir(
    data,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
    alpha: float = 0.8,
    linewidth: float = 2.0,
    colors: dict | None = None,
) -> tuple[Figure, Axes]:
    """Plot SIR model trajectories.

    Parameters
    ----------
    data : dict or pd.DataFrame
        Model output with columns/keys: t, S, I, R.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title.
    show_legend : bool, default True
        Whether to show legend.
    alpha : float, default 0.8
        Line transparency.
    linewidth : float, default 2.0
        Line width.
    colors : dict, optional
        Custom colors for compartments. Default: {"S": "blue", "I": "red", "R": "green"}.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.
    """
    df = _to_dataframe(data, "sir")

    if colors is None:
        colors = {"S": "#1f77b4", "I": "#d62728", "R": "#2ca02c"}

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    time_col = "t" if "t" in df.columns else "day"

    for compartment, color in colors.items():
        if compartment in df.columns:
            ax.plot(
                df[time_col],
                df[compartment],
                label=compartment,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    if show_legend:
        ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    return fig, ax


def plot_safir(
    data,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
    alpha: float = 0.8,
    linewidth: float = 2.0,
    compartments: list[str] | None = None,
    colors: dict | None = None,
) -> tuple[Figure, Axes]:
    """Plot SAFIR/SEIR model trajectories.

    Parameters
    ----------
    data : dict or pd.DataFrame
        Model output with columns/keys: t/day, S, E, I, R, D.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title.
    show_legend : bool, default True
        Whether to show legend.
    alpha : float, default 0.8
        Line transparency.
    linewidth : float, default 2.0
        Line width.
    compartments : list[str], optional
        Which compartments to plot. Default: ["S", "E", "I", "R", "D"].
    colors : dict, optional
        Custom colors for compartments.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.
    """
    df = _to_dataframe(data, "safir")

    if compartments is None:
        compartments = ["S", "E", "I", "R", "D"]

    if colors is None:
        colors = {
            "S": "#1f77b4",  # blue
            "E": "#ff7f0e",  # orange
            "I": "#d62728",  # red
            "R": "#2ca02c",  # green
            "D": "#7f7f7f",  # gray
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    time_col = "t" if "t" in df.columns else "day"

    for compartment in compartments:
        if compartment in df.columns:
            ax.plot(
                df[time_col],
                df[compartment],
                label=compartment,
                color=colors.get(compartment, None),
                alpha=alpha,
                linewidth=linewidth,
            )

    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    if show_legend:
        ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    return fig, ax


def plot_model_comparison(
    observed,
    fitted,
    *,
    compartment: str = "I",
    ax: Axes | None = None,
    title: str | None = None,
    observed_label: str = "Observed",
    fitted_label: str = "Fitted",
) -> tuple[Figure, Axes]:
    """Plot comparison between observed and fitted model outputs.

    Parameters
    ----------
    observed : dict or pd.DataFrame
        Observed/true data.
    fitted : dict or pd.DataFrame
        Fitted model output.
    compartment : str, default "I"
        Which compartment to compare.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    observed_label : str, default "Observed"
        Label for observed data.
    fitted_label : str, default "Fitted"
        Label for fitted data.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.
    """
    df_obs = _to_dataframe(observed)
    df_fit = _to_dataframe(fitted)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    time_col = "t" if "t" in df_obs.columns else "day"

    ax.scatter(
        df_obs[time_col],
        df_obs[compartment],
        label=observed_label,
        color="black",
        alpha=0.7,
        s=30,
        zorder=5,
    )
    ax.plot(
        df_fit[time_col],
        df_fit[compartment],
        label=fitted_label,
        color="#2ca02c",
        linewidth=2.5,
        zorder=4,
    )

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(f"Number {compartment}", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    return fig, ax


def plot_optimization_history(
    history: dict,
    *,
    figsize: tuple[float, float] = (12, 4),
    true_params: float | np.ndarray | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot optimization history showing loss and parameter trajectories.

    Parameters
    ----------
    history : dict
        Optimization history with keys "loss" and optionally "params".
    figsize : tuple, default (12, 4)
        Figure size.
    true_params : float or array, optional
        True parameter value(s) to show as horizontal line.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The matplotlib figure and list of axes objects.
    """
    has_params = "params" in history and history["params"] is not None

    n_plots = 2 if has_params else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Loss curve
    loss = np.asarray(history["loss"])
    axes[0].plot(loss, color="steelblue", linewidth=2)
    axes[0].set_xlabel("Optimization step", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    # Parameter trajectory
    if has_params:
        params = np.asarray(history["params"])
        axes[1].plot(params, color="steelblue", linewidth=2, label="Estimated")
        if true_params is not None:
            axes[1].axhline(
                y=float(true_params),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"True = {float(true_params):.3f}",
            )
        axes[1].set_xlabel("Optimization step", fontsize=12)
        axes[1].set_ylabel("Parameter value", fontsize=12)
        axes[1].set_title("Parameter Trajectory", fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_replicates(
    df: pd.DataFrame,
    *,
    compartment: str = "I",
    ax: Axes | None = None,
    title: str | None = None,
    alpha: float = 0.3,
    show_mean: bool = True,
    show_ci: bool = True,
    ci_level: float = 0.95,
) -> tuple[Figure, Axes]:
    """Plot multiple replicates with mean and confidence interval.

    Parameters
    ----------
    df : pd.DataFrame
        Model output with columns: t, replicate, and compartment columns.
    compartment : str, default "I"
        Which compartment to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    alpha : float, default 0.3
        Transparency for individual trajectories.
    show_mean : bool, default True
        Whether to show mean trajectory.
    show_ci : bool, default True
        Whether to show confidence interval.
    ci_level : float, default 0.95
        Confidence interval level.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    time_col = "t" if "t" in df.columns else "day"

    # Plot individual replicates
    for rep in df["replicate"].unique():
        rep_data = df[df["replicate"] == rep]
        ax.plot(
            rep_data[time_col],
            rep_data[compartment],
            color="steelblue",
            alpha=alpha,
            linewidth=1,
        )

    # Compute and plot mean
    if show_mean or show_ci:
        grouped = df.groupby(time_col)[compartment]
        mean = grouped.mean()
        times = mean.index

        if show_mean:
            ax.plot(times, mean, color="darkblue", linewidth=2.5, label="Mean")

        if show_ci:
            std = grouped.std()
            z = 1.96 if ci_level == 0.95 else 1.645  # approximate
            lower = mean - z * std
            upper = mean + z * std
            ax.fill_between(
                times, lower, upper, color="steelblue", alpha=0.2, label=f"{int(ci_level*100)}% CI"
            )

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(f"Number {compartment}", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    return fig, ax


def plot_training_histories(histories: dict) -> tuple[Figure, list[Axes]]:
    """Plot training and validation losses for one or multiple models.

    Parameters
    ----------
    histories : dict
        Dictionary where keys are model names (str) and values are dictionaries
        containing 'epochs', 'train_loss', 'val_loss', and 'best_epoch'.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The matplotlib figure and list of axes objects.
    """
    num_models = len(histories)

    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))
    if num_models == 1:
        axes = [axes]

    for idx, (model_name, history) in enumerate(histories.items()):
        ax = axes[idx]
        ax.plot(history["epochs"], history["train_loss"], label="Train Loss")
        ax.plot(history["epochs"], history["val_loss"],
                label="Validation Loss")
        ax.axvline(
            x=history["best_epoch"],
            color="r",
            linestyle="--",
            label=f"Best epoch ({history['best_epoch']})",
        )
        ax.set_title(f"{model_name} Model Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, axes


def sir_facet_plot(
    df: pd.DataFrame,
    *,
    compartments: list[str] | None = None,
    facet_by: str | list[str] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Create a faceted plot of SIR model outputs using plotnine.

    This function creates a faceted plot showing epidemic trajectories
    for different parameter combinations, useful for visualizing
    Latin Hypercube Sampling results or sensitivity analyses.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SIR model outputs with columns 't', 'S', 'I', 'R',
        and optionally 'replicate' and parameter columns for faceting.
    compartments : list[str], optional
        Compartments to plot. Default is ["S", "I", "R"].
    facet_by : str or list[str], optional
        Column(s) to facet by. If None, attempts to auto-detect parameter columns.
    figsize : tuple[float, float], optional
        Figure size (width, height) in inches.

    Returns
    -------
    plotnine.ggplot
        A plotnine ggplot object.

    Examples
    --------
    >>> from emidm import run_model_with_replicates, run_sir
    >>> from emidm.plotting import sir_facet_plot
    >>> from emidm.sampler import generate_lhs_samples
    >>> samples = generate_lhs_samples({"beta": [0.1, 0.5], "gamma": [0.05, 0.2]}, n_samples=4)
    >>> results = [run_model_with_replicates(model=run_sir, reps=3, **row.to_dict()).assign(**row.to_dict())
    ...            for _, row in samples.iterrows()]
    >>> df = pd.concat(results)
    >>> plot = sir_facet_plot(df, facet_by=["beta", "gamma"])
    """
    try:
        from plotnine import ggplot, aes, geom_line, facet_wrap, labs, theme_bw
    except ImportError:
        raise ImportError(
            "sir_facet_plot requires plotnine. Install with: pip install plotnine"
        )

    if compartments is None:
        compartments = ["S", "I", "R"]

    # Melt the dataframe to long format for plotting
    id_vars = ["t"]
    if "replicate" in df.columns:
        id_vars.append("replicate")

    # Add faceting columns to id_vars
    if facet_by is None:
        # Auto-detect parameter columns (exclude standard SIR columns)
        standard_cols = {"t", "S", "I", "R", "E", "D", "N", "replicate"}
        facet_by = [c for c in df.columns if c not in standard_cols]

    if isinstance(facet_by, str):
        facet_by = [facet_by]

    id_vars.extend([c for c in facet_by if c in df.columns])

    # Melt to long format
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=compartments,
        var_name="compartment",
        value_name="count",
    )

    # Create facet formula
    if len(facet_by) == 1:
        facet_formula = f"~ {facet_by[0]}"
    elif len(facet_by) == 2:
        facet_formula = f"{facet_by[0]} ~ {facet_by[1]}"
    else:
        # For more than 2, just wrap by first
        facet_formula = f"~ {facet_by[0]}"

    # Build plot
    if "replicate" in df.columns:
        p = (
            ggplot(df_long, aes(x="t", y="count",
                   color="compartment", group="replicate"))
            + geom_line(alpha=0.5)
        )
    else:
        p = (
            ggplot(df_long, aes(x="t", y="count", color="compartment"))
            + geom_line()
        )

    p = (
        p
        + facet_wrap(facet_formula, labeller="label_both")
        + labs(x="Time", y="Count", color="Compartment")
        + theme_bw()
    )

    return p
