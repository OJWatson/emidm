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

            t, replicate, beta, gamma, S, I, R

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
        id_vars=["t", "replicate", "gamma", "beta"],
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
