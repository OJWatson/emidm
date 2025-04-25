from typing import Optional

import pandas as pd
from plotnine import ggplot, aes, geom_line, facet_wrap, theme_bw  # main grammar


def sir_facet_plot(
    df: pd.DataFrame,
    *,
    show: bool = True,
) -> Optional["ggplot"]:
    """
    Plot S, I, R trajectories for each replicate and return the ggplot object.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide-format data with columns::

            t, replicate, beta, gamma, I0, S, I, R

    show : bool, default True
        When *True* the figure is rendered immediately via ``p.draw()``;
        when *False* plotting is suppressed (useful for batch scripts).

    Returns
    -------
    plotnine.ggplot or None
        The constructed ggplot object if *show* is True, otherwise *None*.

    Notes
    -----
    * Lines are coloured by epidemiological compartment (S, I, R).
    * Each replicate gets a unique grouping key so its trajectory is drawn
      as a continuous line.
    * Panels are faceted by the precise (beta, gamma) parameter pair.
    """
    # ---------------------------------------------------------------------
    # 1. Reshape to 'long' (tidy) format
    # ---------------------------------------------------------------------
    df_long = df.melt(
        id_vars=["t", "replicate", "gamma", "beta", "I0"],
        value_vars=["S", "I", "R"],
        var_name="Compartment",
        value_name="Value",
    ).assign(
        uid=lambda d: d["Compartment"] + d["replicate"].astype(str),
        facet=lambda d: (
            "beta = "
            + d["beta"].round(3).astype(str)
            + ",\n"
            + "gamma = "
            + d["gamma"].round(3).astype(str)
            + ",\n"
            + "I0 = "
            + int(d["I0"]).astype(str)
        ),
    )

    # ---------------------------------------------------------------------
    # 2. Build the ggplot figure
    # ---------------------------------------------------------------------
    p = (
        ggplot(
            df_long,
            aes(x="t", y="Value", group="uid", colour="Compartment"),
        )
        + geom_line(alpha=0.7)
        + facet_wrap("facet")
        + theme_bw()
    )

    # ---------------------------------------------------------------------
    # 3. Render and/or return
    # ---------------------------------------------------------------------
    if show:
        # Explicit draw to make sure the figure appears in scripts/IDEs
        p.draw()
        return p
    else:
        # Skip drawing (e.g. during testing or pipelines)
        return None


import matplotlib.pyplot as plt


def plot_training_histories(histories):
    """
    Plot training and validation losses for one or multiple models.

    Args:
        histories (dict):
            Dictionary where keys are model names (str) and values are dictionaries
            containing 'epochs', 'train_loss', 'val_loss', and 'best_epoch'.
            Example:
                {
                    "FFNN": {"epochs": [...], "train_loss": [...], "val_loss": [...], "best_epoch": int},
                    "GRU": {"epochs": [...], "train_loss": [...], "val_loss": [...], "best_epoch": int},
                    ...
                }

    Returns:
        None.
        Displays the plot.
    """
    num_models = len(histories)

    plt.figure(
        figsize=(5 * num_models, 4)
    )  # Adjust figure size based on number of models

    for idx, (model_name, history) in enumerate(histories.items(), start=1):
        plt.subplot(1, num_models, idx)

        plt.plot(history["epochs"], history["train_loss"], label="Train Loss")
        plt.plot(history["epochs"], history["val_loss"], label="Validation Loss")
        plt.axvline(
            x=history["best_epoch"],
            color="r",
            linestyle="--",
            label=f"Best epoch ({history['best_epoch']})",
        )

        plt.title(f"{model_name} Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
