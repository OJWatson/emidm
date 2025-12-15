"""Stochastic SIR model implementation.

This module provides a class-based SIR model following the ABM pattern:
- SIRModel: Model class with init_state(), step(), run() methods
- Convenience functions: run_sir(), run_sir_replicates(), simulate_sir()
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _rt_at_time(R_t: Sequence[float] | Callable[[int], float], t: int) -> float:
    """Get R_t value at time t."""
    if callable(R_t):
        return float(R_t(t))
    return float(R_t[t])


@dataclass
class SIRState:
    """State of the SIR model at a single time point.

    Attributes
    ----------
    S : int
        Number of susceptible individuals.
    I : int
        Number of infected individuals.
    R : int
        Number of recovered individuals.
    """
    S: int
    I: int
    R: int


class SIRModel:
    """Stochastic SIR (Susceptible-Infected-Recovered) model.

    This class implements a discrete-time stochastic SIR model with optional
    time-varying reproduction number. It follows the ABM pattern with:
    - init_state(): Initialize model state
    - step(): Advance model by one time step
    - run(): Run full simulation with optional replicates

    Parameters
    ----------
    N : int, default 1000
        Total population size.
    I0 : int, default 10
        Initial number of infected individuals.
    R_init : int, default 0
        Initial number of recovered individuals.
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
    dt : float, default 1.0
        Time step size.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from emidm.sir import SIRModel
    >>> model = SIRModel(N=1000, I0=10, beta=0.3, gamma=0.1, T=50, seed=42)
    >>> state = model.init_state()
    >>> state = model.step(state, t_idx=0)  # One step
    >>> result = model.run(reps=10)  # Full simulation with 10 replicates
    """

    def __init__(
        self,
        *,
        N: int = 1000,
        I0: int = 10,
        R_init: int = 0,
        beta: float = 0.2,
        gamma: float = 0.1,
        R_t: Sequence[float] | Callable[[int], float] | None = None,
        T: int = 100,
        dt: float = 1.0,
        seed: int | None = None,
    ):
        self.N = int(N)
        self.I0 = int(I0)
        self.R_init = int(R_init)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.R_t = R_t
        self.T = int(T)
        self.dt = float(dt)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def init_state(self) -> SIRState:
        """Initialize the model state.

        Returns
        -------
        SIRState
            Initial state with S = N - I0 - R_init, I = I0, R = R_init.
        """
        return SIRState(
            S=self.N - self.I0 - self.R_init,
            I=self.I0,
            R=self.R_init,
        )

    def step(self, state: SIRState, t_idx: int) -> SIRState:
        """Advance the model by one time step.

        Parameters
        ----------
        state : SIRState
            Current model state.
        t_idx : int
            Current time index (for looking up time-varying R_t).

        Returns
        -------
        SIRState
            Updated model state after one dt step.
        """
        # Get transmission rate (time-varying or constant)
        if self.R_t is not None:
            beta_t = _rt_at_time(self.R_t, t_idx) * self.gamma
        else:
            beta_t = self.beta

        # Compute transition probabilities
        pSI = 1 - np.exp(-beta_t * state.I / self.N * self.dt)
        pIR = 1 - np.exp(-self.gamma * self.dt)

        # Stochastic transitions
        n_SI = self.rng.binomial(state.S, pSI)
        n_IR = self.rng.binomial(state.I, pIR)

        return SIRState(
            S=state.S - n_SI,
            I=state.I + n_SI - n_IR,
            R=state.R + n_IR,
        )

    def run(self, reps: int = 1) -> dict:
        """Run the full simulation.

        Parameters
        ----------
        reps : int, default 1
            Number of stochastic replicates to run.

        Returns
        -------
        dict
            Dictionary with keys:
            - 't': time points array (shape: (n_times,))
            - 'S': susceptible counts
            - 'I': infected counts
            - 'R': recovered counts
            If reps > 1, compartment arrays have shape (reps, n_times).
            If reps == 1, compartment arrays have shape (n_times,).
        """
        n_times = int(self.T / self.dt) + 1
        t_arr = np.arange(0, self.T + self.dt, self.dt)[:n_times]

        S_all = np.zeros((reps, n_times), dtype=np.int64)
        I_all = np.zeros((reps, n_times), dtype=np.int64)
        R_all = np.zeros((reps, n_times), dtype=np.int64)

        for rep in range(reps):
            # Reset RNG for each replicate
            rep_seed = None if self.seed is None else self.seed + rep
            self.rng = np.random.default_rng(rep_seed)

            # Initialize state
            state = self.init_state()

            # Run simulation
            for t_idx in range(n_times):
                S_all[rep, t_idx] = state.S
                I_all[rep, t_idx] = state.I
                R_all[rep, t_idx] = state.R

                if t_idx < n_times - 1:
                    state = self.step(state, t_idx)

        # Return 1D arrays if single replicate, 2D if multiple
        if reps == 1:
            return {"t": t_arr, "S": S_all[0], "I": I_all[0], "R": R_all[0]}
        else:
            return {"t": t_arr, "S": S_all, "I": I_all, "R": R_all}


# -----------------------------------------------------------------------------
# User-friendly simulation function
# -----------------------------------------------------------------------------

def run_sir_simulation(
    *,
    N: int = 1000,
    I0: int = 10,
    R_init: int = 0,
    beta: float = 0.2,
    gamma: float = 0.1,
    R_t: Sequence[float] | Callable[[int], float] | None = None,
    T: int = 100,
    dt: float = 1.0,
    reps: int = 1,
    seed: int | None = None,
) -> dict:
    """Run a stochastic SIR model simulation.

    This is the primary user-friendly interface for running SIR simulations.
    For more control over the simulation, use the SIRModel class directly.

    Parameters
    ----------
    N : int, default 1000
        Total population size.
    I0 : int, default 10
        Initial number of infected individuals.
    R_init : int, default 0
        Initial number of recovered individuals.
    beta : float, default 0.2
        Transmission rate (used if R_t is None).
    gamma : float, default 0.1
        Recovery rate.
    R_t : Sequence or Callable, optional
        Time-varying reproduction number. If provided, beta is computed as
        R_t * gamma at each time step.
    T : int, default 100
        Total simulation time (number of time steps).
    dt : float, default 1.0
        Time step size.
    reps : int, default 1
        Number of stochastic replicates to run.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - 't': time points array (shape: (n_times,))
        - 'S': susceptible counts
        - 'I': infected counts
        - 'R': recovered counts
        If reps == 1, compartment arrays have shape (n_times,).
        If reps > 1, compartment arrays have shape (reps, n_times).

    Examples
    --------
    >>> from emidm import run_sir_simulation
    >>> # Single run
    >>> result = run_sir_simulation(N=1000, I0=10, beta=0.3, gamma=0.1, T=50)
    >>> result["I"].max()  # Peak infections
    >>> # Multiple replicates
    >>> result = run_sir_simulation(N=1000, I0=10, beta=0.3, gamma=0.1, T=50, reps=100)
    >>> result["I"].mean(axis=0)  # Mean infection trajectory
    """
    model = SIRModel(
        N=N, I0=I0, R_init=R_init, beta=beta, gamma=gamma,
        R_t=R_t, T=T, dt=dt, seed=seed,
    )
    return model.run(reps=reps)
