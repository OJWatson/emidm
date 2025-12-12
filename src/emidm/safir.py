import numpy as np
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


def simulate_safir(
    *,
    population: np.ndarray,
    contact_matrix: np.ndarray,
    R0: float = 2.0,
    time_horizon: int = 200,
    dt: float = 0.1,
    seed: int | None = 0,
    n_seed: int = 10,
    prob_hosp: np.ndarray | None = None,
    prob_asymp: float = 0.3,
    prob_non_sev_death: np.ndarray | None = None,
    prob_sev_death: np.ndarray | None = None,
    frac_ICU: float = 0.3,
    dur_E: float = 4.6,
    dur_IMild: float = 2.1,
    dur_ICase: float = 4.5,
    n_replicates: int = 1,
) -> pd.DataFrame:
    population = np.asarray(population, dtype=float)
    contact_matrix = np.asarray(contact_matrix, dtype=float)
    n_age = int(population.shape[0])
    n_replicates = int(n_replicates)

    if contact_matrix.shape != (n_age, n_age):
        raise ValueError("contact_matrix must have shape (n_age, n_age)")

    pop_int = np.floor(population).astype(int)
    if (pop_int < 0).any():
        raise ValueError("population must be non-negative")

    if prob_hosp is None:
        prob_hosp = _interp_to_n_age(_DEFAULT_PROB_HOSP, n_age)
    else:
        prob_hosp = _interp_to_n_age(np.asarray(prob_hosp, dtype=float), n_age)

    if prob_non_sev_death is None:
        prob_non_sev_death = _interp_to_n_age(
            _DEFAULT_PROB_NON_SEV_DEATH, n_age)
    else:
        prob_non_sev_death = _interp_to_n_age(
            np.asarray(prob_non_sev_death, dtype=float), n_age)

    if prob_sev_death is None:
        prob_sev_death = _interp_to_n_age(_DEFAULT_PROB_SEV_DEATH, n_age)
    else:
        prob_sev_death = _interp_to_n_age(
            np.asarray(prob_sev_death, dtype=float), n_age)

    prob_asymp = float(prob_asymp)
    frac_ICU = float(frac_ICU)

    prob_die = frac_ICU * prob_sev_death + \
        (1.0 - frac_ICU) * prob_non_sev_death
    prob_die = np.clip(prob_die, 0.0, 1.0)

    rel_inf_period = (1.0 - prob_hosp) * dur_IMild + prob_hosp * dur_ICase
    beta = _compute_beta_from_R0(
        R0=R0, contact_matrix=contact_matrix, rel_inf_period=rel_inf_period)

    steps_per_day = int(round(1.0 / dt))
    if not np.isclose(steps_per_day * dt, 1.0):
        raise ValueError("dt must evenly divide 1 day (e.g. 0.1, 0.2, 0.25)")

    p_E = 1.0 - np.exp(-dt / (dur_E / 2.0))
    p_Iasym = 1.0 - np.exp(-dt / dur_IMild)
    p_Imild = 1.0 - np.exp(-dt / dur_IMild)
    p_Icase = 1.0 - np.exp(-dt / dur_ICase)

    rows: list[dict] = []
    for rep in range(n_replicates):
        rng = np.random.default_rng(None if seed is None else seed + rep)

        S = pop_int.copy()
        E1 = np.zeros(n_age, dtype=int)
        E2 = np.zeros(n_age, dtype=int)
        Iasy = np.zeros(n_age, dtype=int)
        Imild = np.zeros(n_age, dtype=int)
        Icase = np.zeros(n_age, dtype=int)
        R = np.zeros(n_age, dtype=int)
        D = np.zeros(n_age, dtype=int)

        # Seed initial exposures across age groups proportional to population
        n_seed_eff = int(min(n_seed, int(pop_int.sum())))
        if n_seed_eff > 0:
            probs = pop_int / pop_int.sum()
            seed_by_age = rng.multinomial(n_seed_eff, probs)
            seed_by_age = np.minimum(seed_by_age, S)
            S = S - seed_by_age
            E1 = E1 + seed_by_age

        for day in range(int(time_horizon) + 1):
            # record daily totals (at start of day)
            for a in range(n_age):
                rows.append(
                    {
                        "day": day,
                        "age": a,
                        "replicate": rep,
                        "S": int(S[a]),
                        "E1": int(E1[a]),
                        "E2": int(E2[a]),
                        "Iasy": int(Iasy[a]),
                        "Imild": int(Imild[a]),
                        "Icase": int(Icase[a]),
                        "R": int(R[a]),
                        "D": int(D[a]),
                    }
                )

            if day == int(time_horizon):
                break

            for _ in range(steps_per_day):
                I_tot = Iasy + Imild + Icase
                with np.errstate(divide="ignore", invalid="ignore"):
                    I_frac = np.where(pop_int > 0, I_tot / pop_int, 0.0)

                lambda_age = beta * (contact_matrix @ I_frac)
                p_inf = 1.0 - np.exp(-lambda_age * dt)
                p_inf = np.clip(p_inf, 0.0, 1.0)

                n_SE1 = rng.binomial(S, p_inf)
                n_E1E2 = rng.binomial(E1, p_E)
                n_E2_leave = rng.binomial(E2, p_E)

                # E2 -> I categories
                P_Icase = np.clip(prob_hosp, 0.0, 1.0)
                P_Iasym = (1.0 - P_Icase) * np.clip(prob_asymp, 0.0, 1.0)
                P_Imild = (1.0 - P_Icase) * \
                    (1.0 - np.clip(prob_asymp, 0.0, 1.0))

                n_to_Iasy = np.zeros(n_age, dtype=int)
                n_to_Imild = np.zeros(n_age, dtype=int)
                n_to_Icase = np.zeros(n_age, dtype=int)
                for a in range(n_age):
                    if n_E2_leave[a] == 0:
                        continue
                    split = rng.multinomial(
                        n_E2_leave[a], [P_Iasym[a], P_Imild[a], P_Icase[a]]
                    )
                    n_to_Iasy[a], n_to_Imild[a], n_to_Icase[a] = split

                n_IasyR = rng.binomial(Iasy, p_Iasym)
                n_ImildR = rng.binomial(Imild, p_Imild)
                n_Icase_leave = rng.binomial(Icase, p_Icase)

                n_IcaseD = np.zeros(n_age, dtype=int)
                n_IcaseR = np.zeros(n_age, dtype=int)
                for a in range(n_age):
                    if n_Icase_leave[a] == 0:
                        continue
                    split = rng.multinomial(
                        n_Icase_leave[a], [prob_die[a], 1.0 - prob_die[a]])
                    n_IcaseD[a], n_IcaseR[a] = split

                # apply updates
                S = S - n_SE1
                E1 = E1 + n_SE1 - n_E1E2
                E2 = E2 + n_E1E2 - n_E2_leave

                Iasy = Iasy + n_to_Iasy - n_IasyR
                Imild = Imild + n_to_Imild - n_ImildR
                Icase = Icase + n_to_Icase - n_Icase_leave

                R = R + n_IasyR + n_ImildR + n_IcaseR
                D = D + n_IcaseD

    df = pd.DataFrame(rows)
    df["E"] = df["E1"] + df["E2"]
    df["I"] = df["Iasy"] + df["Imild"] + df["Icase"]
    return df


def run_safir(
    *,
    population: np.ndarray,
    contact_matrix: np.ndarray,
    R0: float = 2.0,
    time_horizon: int = 200,
    dt: float = 0.1,
    seed: int | None = 0,
    n_seed: int = 10,
    **kwargs,
) -> pd.DataFrame:
    df = simulate_safir(
        population=population,
        contact_matrix=contact_matrix,
        R0=R0,
        time_horizon=time_horizon,
        dt=dt,
        seed=seed,
        n_seed=n_seed,
        n_replicates=1,
        **kwargs,
    )
    g = df.groupby(["day", "replicate"], as_index=False)[
        ["S", "E", "I", "R", "D"]
    ].sum()
    return g.drop(columns=["replicate"]).reset_index(drop=True)
