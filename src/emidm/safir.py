"""Stochastic age-structured SAFIR/SEIR model implementation.

This module provides a class-based SAFIR model following the ABM pattern:
- SAFIRModel: Model class with init_state(), step(), run() methods
- Convenience functions: run_safir(), run_safir_replicates(), simulate_safir()

The SAFIR model implements an age-structured SEIR model with hospitalization and death:
Compartments: S -> E1 -> E2 -> (Iasy | Imild | Icase) -> R or D
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


_DEFAULT_PROB_HOSP = np.array(
    [
        0.000840764,
        0.001182411,
        0.001662887,
        0.002338607,
        0.003288907,
        0.004625365,
        0.006504897,
        0.009148183,
        0.012865577,
        0.018093546,
        0.025445917,
        0.035785947,
        0.050327683,
        0.0707785,
        0.099539573,
        0.1399878,
        0.233470395,
    ],
    dtype=float,
)

_DEFAULT_PROB_NON_SEV_DEATH = np.array(
    [
        0.181354223,
        0.181354223,
        0.181354223,
        0.137454906,
        0.121938236,
        0.122775613,
        0.136057441,
        0.160922182,
        0.196987378,
        0.242011054,
        0.289368845,
        0.326537862,
        0.337229819,
        0.309082553,
        0.243794865,
        0.160480254,
        0.057084366,
    ],
    dtype=float,
)

_DEFAULT_PROB_SEV_DEATH = np.array(
    [
        0.226668959,
        0.252420241,
        0.281097009,
        0.413005389,
        0.518451493,
        0.573413613,
        0.576222065,
        0.54253573,
        0.493557696,
        0.447376527,
        0.416666608,
        0.411186639,
        0.443382594,
        0.538718871,
        0.570434076,
        0.643352843,
        0.992620047,
    ],
    dtype=float,
)


def _interp_to_n_age(arr: np.ndarray, n_age: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == n_age:
        return arr
    x_old = np.linspace(0.0, 1.0, arr.shape[0])
    x_new = np.linspace(0.0, 1.0, n_age)
    return np.interp(x_new, x_old, arr)


def _compute_beta_from_R0(*, R0: float, contact_matrix: np.ndarray, rel_inf_period: np.ndarray) -> float:
    contact_matrix = np.asarray(contact_matrix, dtype=float)
    rel_inf_period = np.asarray(rel_inf_period, dtype=float)
    M = contact_matrix * rel_inf_period[None, :]
    eigvals = np.linalg.eigvals(M)
    rho = float(np.max(np.real(eigvals)))
    if rho <= 0:
        return 0.0
    return float(R0) / rho


@dataclass
class SAFIRState:
    """State of the SAFIR model at a single time point.

    All arrays are per age group.

    Attributes
    ----------
    S : np.ndarray
        Susceptible counts by age.
    E1 : np.ndarray
        First exposed compartment by age.
    E2 : np.ndarray
        Second exposed compartment by age.
    Iasy : np.ndarray
        Asymptomatic infectious by age.
    Imild : np.ndarray
        Mild symptomatic by age.
    Icase : np.ndarray
        Severe/hospitalized by age.
    R : np.ndarray
        Recovered by age.
    D : np.ndarray
        Dead by age.
    """
    S: np.ndarray
    E1: np.ndarray
    E2: np.ndarray
    Iasy: np.ndarray
    Imild: np.ndarray
    Icase: np.ndarray
    R: np.ndarray
    D: np.ndarray


class SAFIRModel:
    """Stochastic age-structured SAFIR/SEIR model.

    This class implements a discrete-time stochastic age-structured SEIR model
    with hospitalization and death. It follows the ABM pattern with:
    - init_state(): Initialize model state
    - step(): Advance model by one sub-daily time step
    - run(): Run full simulation with optional replicates

    Compartments: S -> E1 -> E2 -> (Iasy | Imild | Icase) -> R or D

    Parameters
    ----------
    population : array-like
        Population size per age group.
    contact_matrix : array-like
        Contact matrix (n_age x n_age).
    R0 : float, default 2.0
        Basic reproduction number.
    R_t : array-like, optional
        Time-varying reproduction number. If provided, must have length T + 1.
    T : int, default 200
        Number of days to simulate.
    dt : float, default 0.1
        Sub-daily time step (must evenly divide 1 day).
    seed : int, optional
        Random seed for reproducibility.
    I0 : int, default 10
        Initial number of infected individuals.
    prob_hosp : array-like, optional
        Age-specific probability of hospitalization.
    prob_asymp : float, default 0.3
        Probability of asymptomatic infection.
    prob_non_sev_death : array-like, optional
        Age-specific death probability for non-severe cases.
    prob_sev_death : array-like, optional
        Age-specific death probability for severe (ICU) cases.
    frac_ICU : float, default 0.3
        Fraction of hospitalized cases requiring ICU.
    dur_E : float, default 4.6
        Mean duration of exposed period (days).
    dur_IMild : float, default 2.1
        Mean duration of mild/asymptomatic infectious period.
    dur_ICase : float, default 4.5
        Mean duration from symptom onset to hospitalization.

    Examples
    --------
    >>> import numpy as np
    >>> from emidm.safir import SAFIRModel
    >>> population = np.array([1000, 2000, 1500])
    >>> contact_matrix = np.array([[3, 1, 0.5], [1, 2, 1], [0.5, 1, 1.5]])
    >>> model = SAFIRModel(population=population, contact_matrix=contact_matrix, R0=2.5, T=100, seed=42)
    >>> state = model.init_state()
    >>> state = model.step(state, day=0)  # One sub-daily step
    >>> result = model.run(reps=10)  # Full simulation with 10 replicates
    """

    def __init__(
        self,
        *,
        population: np.ndarray,
        contact_matrix: np.ndarray,
        R0: float = 2.0,
        R_t: np.ndarray | None = None,
        T: int = 200,
        dt: float = 0.1,
        seed: int | None = None,
        I0: int = 10,
        prob_hosp: np.ndarray | None = None,
        prob_asymp: float = 0.3,
        prob_non_sev_death: np.ndarray | None = None,
        prob_sev_death: np.ndarray | None = None,
        frac_ICU: float = 0.3,
        dur_E: float = 4.6,
        dur_IMild: float = 2.1,
        dur_ICase: float = 4.5,
    ):
        # Store basic parameters
        self.population = np.asarray(population, dtype=float)
        self.contact_matrix = np.asarray(contact_matrix, dtype=float)
        self.n_age = int(self.population.shape[0])
        self.R0 = float(R0)
        self.R_t = R_t
        self.T = int(T)
        self.dt = float(dt)
        self.seed = seed
        self.I0 = int(I0)
        self.prob_asymp = float(prob_asymp)
        self.frac_ICU = float(frac_ICU)
        self.dur_E = float(dur_E)
        self.dur_IMild = float(dur_IMild)
        self.dur_ICase = float(dur_ICase)

        # Validate inputs
        if self.contact_matrix.shape != (self.n_age, self.n_age):
            raise ValueError("contact_matrix must have shape (n_age, n_age)")

        self.pop_int = np.floor(self.population).astype(int)
        if (self.pop_int < 0).any():
            raise ValueError("population must be non-negative")

        # Process age-specific probabilities
        if prob_hosp is None:
            self.prob_hosp = _interp_to_n_age(_DEFAULT_PROB_HOSP, self.n_age)
        else:
            self.prob_hosp = _interp_to_n_age(
                np.asarray(prob_hosp, dtype=float), self.n_age)

        if prob_non_sev_death is None:
            self.prob_non_sev_death = _interp_to_n_age(
                _DEFAULT_PROB_NON_SEV_DEATH, self.n_age)
        else:
            self.prob_non_sev_death = _interp_to_n_age(
                np.asarray(prob_non_sev_death, dtype=float), self.n_age)

        if prob_sev_death is None:
            self.prob_sev_death = _interp_to_n_age(
                _DEFAULT_PROB_SEV_DEATH, self.n_age)
        else:
            self.prob_sev_death = _interp_to_n_age(
                np.asarray(prob_sev_death, dtype=float), self.n_age)

        # Compute derived parameters
        self.prob_die = self.frac_ICU * self.prob_sev_death + \
            (1.0 - self.frac_ICU) * self.prob_non_sev_death
        self.prob_die = np.clip(self.prob_die, 0.0, 1.0)

        rel_inf_period = (1.0 - self.prob_hosp) * \
            self.dur_IMild + self.prob_hosp * self.dur_ICase
        self.beta_base = _compute_beta_from_R0(
            R0=self.R0, contact_matrix=self.contact_matrix, rel_inf_period=rel_inf_period
        )

        # Prepare time-varying beta
        if self.R_t is not None:
            R_t_arr = np.asarray(self.R_t, dtype=float)
            if R_t_arr.shape[0] != self.T + 1:
                raise ValueError(
                    f"R_t must have length T + 1 = {self.T + 1}, got {R_t_arr.shape[0]}")
            self.beta_t_daily = self.beta_base * R_t_arr / self.R0
        else:
            self.beta_t_daily = np.full(
                self.T + 1, self.beta_base, dtype=float)

        # Validate dt
        self.steps_per_day = int(round(1.0 / self.dt))
        if not np.isclose(self.steps_per_day * self.dt, 1.0):
            raise ValueError(
                "dt must evenly divide 1 day (e.g. 0.1, 0.2, 0.25)")

        # Transition probabilities
        self.p_E = 1.0 - np.exp(-self.dt / (self.dur_E / 2.0))
        self.p_Iasym = 1.0 - np.exp(-self.dt / self.dur_IMild)
        self.p_Imild = 1.0 - np.exp(-self.dt / self.dur_IMild)
        self.p_Icase = 1.0 - np.exp(-self.dt / self.dur_ICase)

        # Category probabilities
        self.P_Icase = np.clip(self.prob_hosp, 0.0, 1.0)
        self.P_Iasym = (1.0 - self.P_Icase) * \
            np.clip(self.prob_asymp, 0.0, 1.0)
        self.P_Imild = (1.0 - self.P_Icase) * \
            (1.0 - np.clip(self.prob_asymp, 0.0, 1.0))

        # Initialize RNG
        self.rng = np.random.default_rng(seed)

    def init_state(self) -> SAFIRState:
        """Initialize the model state.

        Seeds initial infections across age groups proportional to population.

        Returns
        -------
        SAFIRState
            Initial state with seeded infections in E1 compartment.
        """
        S = self.pop_int.copy()
        E1 = np.zeros(self.n_age, dtype=int)
        E2 = np.zeros(self.n_age, dtype=int)
        Iasy = np.zeros(self.n_age, dtype=int)
        Imild = np.zeros(self.n_age, dtype=int)
        Icase = np.zeros(self.n_age, dtype=int)
        R = np.zeros(self.n_age, dtype=int)
        D = np.zeros(self.n_age, dtype=int)

        # Seed initial exposures across age groups proportional to population
        I0_eff = int(min(self.I0, int(self.pop_int.sum())))
        if I0_eff > 0:
            probs = self.pop_int / self.pop_int.sum()
            seed_by_age = self.rng.multinomial(I0_eff, probs)
            seed_by_age = np.minimum(seed_by_age, S)
            S = S - seed_by_age
            E1 = E1 + seed_by_age

        return SAFIRState(S=S, E1=E1, E2=E2, Iasy=Iasy, Imild=Imild, Icase=Icase, R=R, D=D)

    def step(self, state: SAFIRState, day: int) -> SAFIRState:
        """Advance the model by one sub-daily time step.

        Parameters
        ----------
        state : SAFIRState
            Current model state.
        day : int
            Current day (for looking up time-varying beta).

        Returns
        -------
        SAFIRState
            Updated model state after one dt step.
        """
        # Compute force of infection
        I_tot = state.Iasy + state.Imild + state.Icase
        with np.errstate(divide="ignore", invalid="ignore"):
            I_frac = np.where(self.pop_int > 0, I_tot / self.pop_int, 0.0)

        beta_t = self.beta_t_daily[day]
        lambda_age = beta_t * (self.contact_matrix @ I_frac)
        p_inf = 1.0 - np.exp(-lambda_age * self.dt)
        p_inf = np.clip(p_inf, 0.0, 1.0)

        # Transitions
        n_SE1 = self.rng.binomial(state.S, p_inf)
        n_E1E2 = self.rng.binomial(state.E1, self.p_E)
        n_E2_leave = self.rng.binomial(state.E2, self.p_E)

        # E2 -> I categories
        n_to_Iasy = np.zeros(self.n_age, dtype=int)
        n_to_Imild = np.zeros(self.n_age, dtype=int)
        n_to_Icase = np.zeros(self.n_age, dtype=int)
        for a in range(self.n_age):
            if n_E2_leave[a] == 0:
                continue
            split = self.rng.multinomial(
                n_E2_leave[a], [self.P_Iasym[a], self.P_Imild[a], self.P_Icase[a]])
            n_to_Iasy[a], n_to_Imild[a], n_to_Icase[a] = split

        # I -> R/D transitions
        n_IasyR = self.rng.binomial(state.Iasy, self.p_Iasym)
        n_ImildR = self.rng.binomial(state.Imild, self.p_Imild)
        n_Icase_leave = self.rng.binomial(state.Icase, self.p_Icase)

        n_IcaseD = np.zeros(self.n_age, dtype=int)
        n_IcaseR = np.zeros(self.n_age, dtype=int)
        for a in range(self.n_age):
            if n_Icase_leave[a] == 0:
                continue
            split = self.rng.multinomial(
                n_Icase_leave[a], [self.prob_die[a], 1.0 - self.prob_die[a]])
            n_IcaseD[a], n_IcaseR[a] = split

        # Apply updates
        return SAFIRState(
            S=state.S - n_SE1,
            E1=state.E1 + n_SE1 - n_E1E2,
            E2=state.E2 + n_E1E2 - n_E2_leave,
            Iasy=state.Iasy + n_to_Iasy - n_IasyR,
            Imild=state.Imild + n_to_Imild - n_ImildR,
            Icase=state.Icase + n_to_Icase - n_Icase_leave,
            R=state.R + n_IasyR + n_ImildR + n_IcaseR,
            D=state.D + n_IcaseD,
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
            - 't': time points array (shape: (T+1,))
            - 'S', 'E', 'I', 'R', 'D': compartment totals
            If reps > 1, compartment arrays have shape (reps, T+1).
            If reps == 1, compartment arrays have shape (T+1,).
        """
        n_days = self.T + 1

        S_all = np.zeros((reps, n_days), dtype=np.int64)
        E_all = np.zeros((reps, n_days), dtype=np.int64)
        I_all = np.zeros((reps, n_days), dtype=np.int64)
        R_all = np.zeros((reps, n_days), dtype=np.int64)
        D_all = np.zeros((reps, n_days), dtype=np.int64)

        for rep in range(reps):
            # Reset RNG for each replicate
            rep_seed = None if self.seed is None else self.seed + rep
            self.rng = np.random.default_rng(rep_seed)

            # Initialize state
            state = self.init_state()

            # Run simulation day by day
            for day in range(n_days):
                # Record daily totals
                S_all[rep, day] = state.S.sum()
                E_all[rep, day] = state.E1.sum() + state.E2.sum()
                I_all[rep, day] = state.Iasy.sum() + state.Imild.sum() + \
                    state.Icase.sum()
                R_all[rep, day] = state.R.sum()
                D_all[rep, day] = state.D.sum()

                if day < self.T:
                    # Run sub-daily steps
                    for _ in range(self.steps_per_day):
                        state = self.step(state, day)

        t_arr = np.arange(n_days)

        # Return 1D arrays if single replicate, 2D if multiple
        if reps == 1:
            return {"t": t_arr, "S": S_all[0], "E": E_all[0], "I": I_all[0], "R": R_all[0], "D": D_all[0]}
        else:
            return {"t": t_arr, "S": S_all, "E": E_all, "I": I_all, "R": R_all, "D": D_all}


# -----------------------------------------------------------------------------
# User-friendly simulation function
# -----------------------------------------------------------------------------

def run_safir_simulation(
    *,
    population: np.ndarray,
    contact_matrix: np.ndarray,
    R0: float = 2.0,
    R_t: np.ndarray | None = None,
    T: int = 200,
    dt: float = 0.1,
    seed: int | None = 0,
    I0: int = 10,
    prob_hosp: np.ndarray | None = None,
    prob_asymp: float = 0.3,
    prob_non_sev_death: np.ndarray | None = None,
    prob_sev_death: np.ndarray | None = None,
    frac_ICU: float = 0.3,
    dur_E: float = 4.6,
    dur_IMild: float = 2.1,
    dur_ICase: float = 4.5,
    reps: int = 1,
) -> dict:
    """Run an age-structured SAFIR/SEIR model simulation.

    This is the primary user-friendly interface for running SAFIR simulations.
    For more control over the simulation, use the SAFIRModel class directly.

    The model implements an age-structured SEIR model with hospitalization and death:
    Compartments: S -> E1 -> E2 -> (Iasy | Imild | Icase) -> R or D

    Parameters
    ----------
    population : array-like
        Population size per age group.
    contact_matrix : array-like
        Contact matrix (n_age x n_age).
    R0 : float, default 2.0
        Basic reproduction number.
    R_t : array-like, optional
        Time-varying reproduction number. If provided, must have length T + 1.
    T : int, default 200
        Number of days to simulate.
    dt : float, default 0.1
        Sub-daily time step (must evenly divide 1 day).
    seed : int, optional
        Random seed for reproducibility.
    I0 : int, default 10
        Initial number of infected individuals.
    prob_hosp : array-like, optional
        Age-specific probability of hospitalization.
    prob_asymp : float, default 0.3
        Probability of asymptomatic infection.
    prob_non_sev_death : array-like, optional
        Age-specific death probability for non-severe cases.
    prob_sev_death : array-like, optional
        Age-specific death probability for severe (ICU) cases.
    frac_ICU : float, default 0.3
        Fraction of hospitalized cases requiring ICU.
    dur_E : float, default 4.6
        Mean duration of exposed period (days).
    dur_IMild : float, default 2.1
        Mean duration of mild/asymptomatic infectious period.
    dur_ICase : float, default 4.5
        Mean duration from symptom onset to hospitalization.
    reps : int, default 1
        Number of stochastic replicates to run.

    Returns
    -------
    dict
        Dictionary with keys:
        - 't': time points array (shape: (T+1,))
        - 'S': susceptible counts
        - 'E': exposed counts (E1 + E2)
        - 'I': infected counts (Iasy + Imild + Icase)
        - 'R': recovered counts
        - 'D': dead counts
        If reps == 1, compartment arrays have shape (T+1,).
        If reps > 1, compartment arrays have shape (reps, T+1).

    Examples
    --------
    >>> import numpy as np
    >>> from emidm import run_safir_simulation
    >>> population = np.array([1000, 2000])
    >>> contact_matrix = np.array([[3, 1], [1, 2]])
    >>> # Single run
    >>> result = run_safir_simulation(population=population, contact_matrix=contact_matrix, T=50)
    >>> result["I"].max()  # Peak infections
    >>> # Multiple replicates
    >>> result = run_safir_simulation(population=population, contact_matrix=contact_matrix, T=50, reps=100)
    >>> result["I"].mean(axis=0)  # Mean infection trajectory
    """
    model = SAFIRModel(
        population=population,
        contact_matrix=contact_matrix,
        R0=R0,
        R_t=R_t,
        T=T,
        dt=dt,
        seed=seed,
        I0=I0,
        prob_hosp=prob_hosp,
        prob_asymp=prob_asymp,
        prob_non_sev_death=prob_non_sev_death,
        prob_sev_death=prob_sev_death,
        frac_ICU=frac_ICU,
        dur_E=dur_E,
        dur_IMild=dur_IMild,
        dur_ICase=dur_ICase,
    )
    return model.run(reps=reps)
