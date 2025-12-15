from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffConfig:
    tau: float = 0.5
    hard: bool = True


def _prepare_beta_t_sequence(*, beta: float, gamma: float, R_t, ts):
    _require_jax()
    import jax.numpy as jnp

    if R_t is None:
        return jnp.ones_like(ts, dtype=jnp.float32) * jnp.asarray(beta, dtype=jnp.float32)
    if callable(R_t):
        raise TypeError(
            "For differentiable simulations, `R_t` must be an array/sequence (not a Python callable)."
        )
    R_t_arr = jnp.asarray(R_t, dtype=jnp.float32)
    if R_t_arr.shape[0] != ts.shape[0]:
        raise ValueError(
            "For differentiable simulations, `R_t` must have the same length as `t` (T/dt + 1)."
        )
    return R_t_arr * jnp.asarray(gamma, dtype=jnp.float32)


def _require_jax():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Differentiable models require 'jax' to be installed. "
            "Install with `pip install emidm[jax]` (once configured) or `pip install jax jaxlib`."
        ) from e


# Note: _gumbel_softmax_bernoulli_jit and _gumbel_softmax_categorical_jit
# are the JIT-compatible versions used in the core simulation functions.
# They avoid _require_jax() calls and use jax.lax.cond instead of Python if.


def _init_onehot_state_sir(*, N: int, I0: int, key=None):
    _require_jax()
    import jax
    import jax.numpy as jnp

    N = int(N)
    I0 = int(I0)
    I0 = max(0, min(I0, N))

    if key is None:
        infected_idx = jnp.arange(I0)
    else:
        infected_idx = jax.random.choice(
            key, N, shape=(I0,), replace=False)

    infected_mask = jnp.zeros(
        (N,), dtype=jnp.float32).at[infected_idx].set(1.0)
    susceptible_mask = 1.0 - infected_mask
    recovered_mask = jnp.zeros((N,), dtype=jnp.float32)
    return jnp.stack([susceptible_mask, infected_mask, recovered_mask], axis=-1)


def _run_diff_sir_core(beta_seq, state0, key, N, gamma, dt, tau, hard):
    """JIT-compilable core of the differentiable SIR simulation."""
    import jax
    import jax.numpy as jnp

    def step(carry, idx):
        state, key = carry
        S = state[:, 0]
        I = state[:, 1]
        R = state[:, 2]

        I_total = jnp.sum(I)
        beta_t = beta_seq[idx]

        p_infect = 1.0 - jnp.exp(-beta_t * I_total / N * dt)
        p_recover = 1.0 - jnp.exp(-gamma * dt)

        key, k_inf, k_rec = jax.random.split(key, 3)

        infect = _gumbel_softmax_bernoulli_jit(
            k_inf, p_infect, tau, hard, sample_shape=(N,))
        recover = _gumbel_softmax_bernoulli_jit(
            k_rec, p_recover, tau, hard, sample_shape=(N,))

        new_I = infect * S
        new_R = recover * I

        S_next = S - new_I
        I_next = I + new_I - new_R
        R_next = R + new_R

        state_next = jnp.stack([S_next, I_next, R_next], axis=-1)
        totals = jnp.array([jnp.sum(S_next), jnp.sum(I_next), jnp.sum(R_next)])
        return (state_next, key), totals

    n_steps = beta_seq.shape[0] - 1
    idxs = jnp.arange(1, n_steps + 1)
    (stateT, _), totals = jax.lax.scan(step, (state0, key), idxs)

    totals0 = jnp.array(
        [jnp.sum(state0[:, 0]), jnp.sum(state0[:, 1]), jnp.sum(state0[:, 2])]
    )
    totals = jnp.vstack([totals0[None, :], totals])

    return totals


def _gumbel_softmax_bernoulli_jit(key, p, tau, hard, sample_shape=None):
    """JIT-friendly version without _require_jax calls."""
    import jax
    import jax.numpy as jnp

    eps = 1e-6
    p = jnp.clip(jnp.asarray(p, dtype=jnp.float32), eps, 1.0 - eps)
    if sample_shape is None:
        target_shape = p.shape
    else:
        target_shape = tuple(sample_shape)
        p = jnp.broadcast_to(p, target_shape)

    logits = jnp.stack([jnp.log(p), jnp.log1p(-p)], axis=-1)
    u = jax.random.uniform(key, logits.shape, minval=eps, maxval=1.0 - eps)
    g = -jnp.log(-jnp.log(u))
    y = jax.nn.softmax((logits + g) / tau, axis=-1)

    y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), 2)
    y = jax.lax.cond(hard, lambda: jax.lax.stop_gradient(
        y_hard - y) + y, lambda: y)

    return y[..., 0]


def run_diff_sir_simulation(
    *,
    N: int = 200,
    I0: int = 5,
    beta: float = 0.2,
    gamma: float = 0.1,
    T: int = 50,
    dt: int = 1,
    R_t=None,
    key,
    reps: int = 1,
    config: DiffConfig = DiffConfig(),
):
    """Run a differentiable SIR simulation using straight-through Gumbel-Softmax.

    This model uses the Gumbel-Softmax reparameterization trick to make discrete
    stochastic transitions differentiable, enabling gradient-based optimization
    of model parameters.

    When reps > 1, uses JAX's vmap for efficient parallel execution across replicates.

    Parameters
    ----------
    N : int, default 200
        Total population size (number of agents).
    I0 : int, default 5
        Initial number of infected individuals.
    beta : float, default 0.2
        Transmission rate (used if R_t is None).
    gamma : float, default 0.1
        Recovery rate.
    T : int, default 50
        Total simulation time.
    dt : int, default 1
        Time step size.
    R_t : array-like, optional
        Time-varying reproduction number. If provided, must have length T/dt + 1.
        Beta is computed as R_t * gamma at each time step.
    key : jax.random.PRNGKey
        JAX random key for stochastic simulation. Required.
    reps : int, default 1
        Number of replicates to run. When reps > 1, uses vmap for parallel execution.
    config : DiffConfig, default DiffConfig()
        Configuration for Gumbel-Softmax (tau, hard).

    Returns
    -------
    dict
        Dictionary with keys 't', 'S', 'I', 'R' as JAX arrays.
        - If reps == 1: shapes are (T+1,) for each compartment
        - If reps > 1: shapes are (reps, T+1) for each compartment

    Examples
    --------
    >>> from emidm import run_diff_sir_simulation, DiffConfig
    >>>
    >>> import jax
    >>> # Single run
    >>> key = jax.random.PRNGKey(42)
    >>> result = run_diff_sir_simulation(
    ...     N=100,
    ...     I0=5,
    ...     beta=0.3,
    ...     gamma=0.1,
    ...     T=50,
    ...     config=DiffConfig(tau=0.5, hard=True),
    ...     key=key,
    ... )
    >>> print(result["I"].max())  # Peak infections
    >>>
    >>> # Multiple replicates
    >>> key = jax.random.PRNGKey(0)
    >>> results = run_diff_sir_simulation(
    ...     N=100, I0=5, beta=0.3, gamma=0.1, T=50, reps=10,
    ...     config=DiffConfig(tau=0.5, hard=True),
    ...     key=key,
    ... )
    >>> print(results["I"].shape)  # (10, 51)
    """
    _require_jax()
    import jax
    import jax.numpy as jnp

    N = int(N)
    T = int(T)
    dt = int(dt)
    reps = int(reps)

    ts = jnp.arange(0, T + 1, dt)
    beta_seq = _prepare_beta_t_sequence(beta=beta, gamma=gamma, R_t=R_t, ts=ts)

    if reps == 1:
        # Single replicate - original behavior
        key, k_init = jax.random.split(key)
        state0 = _init_onehot_state_sir(N=N, I0=I0, key=k_init)

        totals = _run_diff_sir_core(
            beta_seq, state0, key, N, gamma, dt, config.tau, config.hard
        )

        return {
            "t": ts,
            "S": totals[:, 0],
            "I": totals[:, 1],
            "R": totals[:, 2],
        }
    else:
        # Multiple replicates - use vmap for parallel execution
        # reps for init, reps for run
        keys = jax.random.split(key, reps * 2)
        init_keys = keys[:reps]
        run_keys = keys[reps:]

        # Initialize states for all replicates using vmap
        states0 = jax.vmap(
            lambda k: _init_onehot_state_sir(N=N, I0=I0, key=k)
        )(init_keys)

        # Run all replicates in parallel using vmap
        def run_one(state0, run_key):
            return _run_diff_sir_core(
                beta_seq, state0, run_key, N, gamma, dt, config.tau, config.hard
            )

        all_totals = jax.vmap(run_one)(states0, run_keys)

        return {
            "t": ts,
            "S": all_totals[:, :, 0],
            "I": all_totals[:, :, 1],
            "R": all_totals[:, :, 2],
        }


def make_diff_sir_model(
    *,
    N: int = 200,
    I0: int = 5,
    T: int = 50,
    dt: int = 1,
    key,
    reps: int = 1,
    config: DiffConfig = DiffConfig(),
    beta: float = 0.2,
    gamma: float = 0.1,
):
    """Create a pre-compiled SIR model function for gradient-based inference.

    This factory function pre-computes all static values (population structure,
    initial state, etc.) and returns a JIT-compiled function that takes an
    optional params dict to override default parameter values.

    Parameters
    ----------
    N : int, default 200
        Total population size (number of agents). Static - affects array shapes.
    I0 : int, default 5
        Initial number of infected individuals. Static - affects initial state.
    T : int, default 50
        Total simulation time. Static - affects output array length.
    dt : int, default 1
        Time step size. Static - affects number of steps.
    key : jax.random.PRNGKey
        JAX random key for stochastic simulation. Required.
    reps : int, default 1
        Number of replicates to run. Static - affects array shapes.
    config : DiffConfig, default DiffConfig()
        Configuration for Gumbel-Softmax (tau, hard). Static.
    beta : float, default 0.2
        Default transmission rate. Can be overridden at call time.
    gamma : float, default 0.1
        Default recovery rate. Can be overridden at call time.

    Returns
    -------
    callable
        A JIT-compiled function `model(params=None)` where `params` is an
        optional dict that can override: 'beta', 'gamma', 'R_t'.
        Returns dict with keys 't', 'S', 'I', 'R'.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from emidm import make_diff_sir_model
    >>>
    >>> # Create model once - pre-computes static values
    >>> key = jax.random.PRNGKey(0)
    >>> model = make_diff_sir_model(N=1000, I0=10, T=50, key=key)
    >>>
    >>> # Use defaults
    >>> result = model()
    >>>
    >>> # Override beta for inference
    >>> result = model({'beta': 0.35})
    >>>
    >>> # Override multiple params
    >>> result = model({'beta': 0.35, 'gamma': 0.12})
    >>>
    >>> # Use R_t instead of beta
    >>> R_t = jnp.ones(51) * 2.5
    >>> result = model({'R_t': R_t, 'gamma': 0.1})
    >>>
    >>> # Use in loss function for inference
    >>> observed_I = jnp.array([...])
    >>> @jax.jit
    ... def loss(beta):
    ...     pred = model({'beta': beta})
    ...     return jnp.mean((pred['I'] - observed_I)**2)
    >>> grad_fn = jax.grad(loss)
    """
    _require_jax()
    import jax
    import jax.numpy as jnp

    # Convert to static Python values
    N = int(N)
    T = int(T)
    dt = int(dt)
    reps = int(reps)

    # Pre-compute time array (static)
    ts = jnp.arange(0, T + 1, dt)
    n_steps = ts.shape[0]

    # Pre-compute initial state (static)
    if reps == 1:
        key, k_init = jax.random.split(key)
        state0 = _init_onehot_state_sir(N=N, I0=I0, key=k_init)
    else:
        keys = jax.random.split(key, reps * 2)
        init_keys = keys[:reps]
        run_keys = keys[reps:]
        states0 = jax.vmap(
            lambda k: _init_onehot_state_sir(N=N, I0=I0, key=k)
        )(init_keys)

    # Capture defaults in closure
    default_beta = float(beta)
    default_gamma = float(gamma)
    tau = config.tau
    hard = config.hard

    if reps == 1:
        @jax.jit
        def _model(params=None):
            # Merge params with defaults
            p_beta = default_beta
            p_gamma = default_gamma
            p_R_t = None

            if params is not None:
                p_beta = params.get('beta', p_beta)
                p_gamma = params.get('gamma', p_gamma)
                p_R_t = params.get('R_t', None)

            # Compute beta sequence
            if p_R_t is not None:
                R_t_arr = jnp.asarray(p_R_t, dtype=jnp.float32)
                beta_seq = R_t_arr * jnp.asarray(p_gamma, dtype=jnp.float32)
            else:
                beta_seq = jnp.ones(n_steps, dtype=jnp.float32) * \
                    jnp.asarray(p_beta, dtype=jnp.float32)

            totals = _run_diff_sir_core(
                beta_seq, state0, key, N, p_gamma, dt, tau, hard
            )

            return {
                "t": ts,
                "S": totals[:, 0],
                "I": totals[:, 1],
                "R": totals[:, 2],
            }
    else:
        @jax.jit
        def _model(params=None):
            # Merge params with defaults
            p_beta = default_beta
            p_gamma = default_gamma
            p_R_t = None

            if params is not None:
                p_beta = params.get('beta', p_beta)
                p_gamma = params.get('gamma', p_gamma)
                p_R_t = params.get('R_t', None)

            # Compute beta sequence
            if p_R_t is not None:
                R_t_arr = jnp.asarray(p_R_t, dtype=jnp.float32)
                beta_seq = R_t_arr * jnp.asarray(p_gamma, dtype=jnp.float32)
            else:
                beta_seq = jnp.ones(n_steps, dtype=jnp.float32) * \
                    jnp.asarray(p_beta, dtype=jnp.float32)

            def run_one(state0, run_key):
                return _run_diff_sir_core(
                    beta_seq, state0, run_key, N, p_gamma, dt, tau, hard
                )

            all_totals = jax.vmap(run_one)(states0, run_keys)

            return {
                "t": ts,
                "S": all_totals[:, :, 0],
                "I": all_totals[:, :, 1],
                "R": all_totals[:, :, 2],
            }

    return _model


def _gumbel_softmax_categorical_jit(key, probs, tau, hard):
    """JIT-friendly version of Gumbel-Softmax categorical without _require_jax calls."""
    import jax
    import jax.numpy as jnp

    eps = 1e-12
    probs = jnp.clip(probs, eps, 1.0)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)

    logits = jnp.log(probs)
    u = jax.random.uniform(key, logits.shape, minval=eps, maxval=1.0 - eps)
    g = -jnp.log(-jnp.log(u))
    y = jax.nn.softmax((logits + g) / tau, axis=-1)

    y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), y.shape[-1])
    y = jax.lax.cond(hard, lambda: jax.lax.stop_gradient(
        y_hard - y) + y, lambda: y)

    return y


def _run_diff_safir_core(
    beta_t_daily,
    state0,
    key,
    population,
    contact_matrix,
    age_index,
    prob_die_by_agent,
    P_Icase,
    P_Iasym,
    P_Imild,
    p_E,
    p_Iasym,
    p_Imild,
    p_Icase,
    dt,
    tau,
    hard,
    steps_per_day,
    n_age,
    N,
):
    """JIT-compilable core of the differentiable SAFIR simulation.

    All arguments must be JAX arrays or Python scalars that are static.
    This function contains no int() or float() calls on traced values.
    """
    import jax
    import jax.numpy as jnp

    # State indexing (constants)
    S, E1, E2, Iasy, Imild, Icase, R, D = 0, 1, 2, 3, 4, 5, 6, 7
    num_states = 8

    def step_sub(carry, _):
        state, key, beta_t = carry

        infectious = state[:, Iasy] + state[:, Imild] + state[:, Icase]
        inf_by_age = jnp.zeros(
            (n_age,), dtype=jnp.float32).at[age_index].add(infectious)
        I_frac = jnp.where(population > 0, inf_by_age / population, 0.0)
        lambda_age = beta_t * (contact_matrix @ I_frac)
        p_inf = 1.0 - jnp.exp(-lambda_age[age_index] * dt)
        p_inf = jnp.clip(p_inf, 0.0, 1.0)

        s = state[:, S]
        e1 = state[:, E1]
        e2 = state[:, E2]
        iasy = state[:, Iasy]
        imild = state[:, Imild]
        icase = state[:, Icase]
        r = state[:, R]
        d = state[:, D]

        probs_next = jnp.zeros((N, num_states), dtype=jnp.float32)

        probs_next = probs_next.at[:, S].add(s * (1.0 - p_inf))
        probs_next = probs_next.at[:, E1].add(s * p_inf)

        probs_next = probs_next.at[:, E1].add(e1 * (1.0 - p_E))
        probs_next = probs_next.at[:, E2].add(e1 * p_E)

        probs_next = probs_next.at[:, E2].add(e2 * (1.0 - p_E))
        probs_next = probs_next.at[:, Iasy].add(e2 * (p_E * P_Iasym))
        probs_next = probs_next.at[:, Imild].add(e2 * (p_E * P_Imild))
        probs_next = probs_next.at[:, Icase].add(e2 * (p_E * P_Icase))

        probs_next = probs_next.at[:, Iasy].add(iasy * (1.0 - p_Iasym))
        probs_next = probs_next.at[:, R].add(iasy * p_Iasym)

        probs_next = probs_next.at[:, Imild].add(imild * (1.0 - p_Imild))
        probs_next = probs_next.at[:, R].add(imild * p_Imild)

        probs_next = probs_next.at[:, Icase].add(icase * (1.0 - p_Icase))
        probs_next = probs_next.at[:, R].add(
            icase * (p_Icase * (1.0 - prob_die_by_agent)))
        probs_next = probs_next.at[:, D].add(
            icase * (p_Icase * prob_die_by_agent))

        probs_next = probs_next.at[:, R].add(r)
        probs_next = probs_next.at[:, D].add(d)

        probs_next = probs_next / jnp.sum(probs_next, axis=1, keepdims=True)

        key, sub = jax.random.split(key)
        next_state = _gumbel_softmax_categorical_jit(
            sub, probs_next, tau, hard)
        return (next_state, key, beta_t), None

    def one_day(carry, day_idx):
        state, key = carry
        beta_t = beta_t_daily[day_idx]
        (state, key, _), _ = jax.lax.scan(
            step_sub, (state, key, beta_t), None, length=steps_per_day
        )

        S_count = jnp.sum(state[:, S])
        E_count = jnp.sum(state[:, E1] + state[:, E2])
        I_count = jnp.sum(state[:, Iasy] + state[:, Imild] + state[:, Icase])
        R_count = jnp.sum(state[:, R])
        D_count = jnp.sum(state[:, D])
        daily = jnp.stack([S_count, E_count, I_count, R_count, D_count])
        return (state, key), daily

    days = beta_t_daily.shape[0] - 1
    day_indices = jnp.arange(days + 1)
    (stateT, keyT), daily_series = jax.lax.scan(
        one_day, (state0, key), day_indices)

    return daily_series


def run_diff_safir_simulation(
    *,
    population,
    contact_matrix,
    R0: float = 2.0,
    R_t=None,
    T: int = 200,
    dt: float = 0.1,
    seed: int = 0,
    reps: int = 1,
    config: DiffConfig = DiffConfig(),
    I0: int = 10,
    prob_hosp=None,
    prob_asymp: float = 0.3,
    prob_non_sev_death=None,
    prob_sev_death=None,
    frac_ICU: float = 0.3,
    dur_E: float = 4.6,
    dur_IMild: float = 2.1,
    dur_ICase: float = 4.5,
):
    """Age-structured differentiable SAFIR/SEIR model with contact matrices.

    This model implements an age-structured SEIR model with hospitalization and death,
    using Gumbel-Softmax for differentiable stochastic transitions.

    Compartments: S -> E1 -> E2 -> (Iasy | Imild | Icase) -> R or D

    When reps > 1, runs multiple replicates sequentially (SAFIR is too complex for vmap).

    Parameters
    ----------
    population : array-like
        Population size per age group.
    contact_matrix : array-like
        Contact matrix (n_age x n_age).
    R0 : float, default 2.0
        Basic reproduction number (used if R_t is None).
    R_t : array-like, optional
        Time-varying reproduction number. If provided, must have length T + 1.
        Uses constant interpolation per day.
    T : int, default 200
        Number of days to simulate.
    dt : float, default 0.1
        Sub-daily time step (must evenly divide 1 day).
    seed : int, default 0
        Random seed for reproducibility.
    reps : int, default 1
        Number of replicates to run. Replicates are run sequentially.
    config : DiffConfig, default DiffConfig()
        Configuration for Gumbel-Softmax (tau, hard).
    I0 : int, default 10
        Initial number of infected individuals.
    prob_hosp : array-like, optional
        Age-specific probability of hospitalization given infection.
    prob_asymp : float, default 0.3
        Probability of asymptomatic infection (given non-hospitalized).
    prob_non_sev_death : array-like, optional
        Age-specific death probability for non-severe hospitalized cases.
    prob_sev_death : array-like, optional
        Age-specific death probability for severe (ICU) cases.
    frac_ICU : float, default 0.3
        Fraction of hospitalized cases requiring ICU.
    dur_E : float, default 4.6
        Mean duration of exposed period (days).
    dur_IMild : float, default 2.1
        Mean duration of mild/asymptomatic infectious period (days).
    dur_ICase : float, default 4.5
        Mean duration from symptom onset to hospitalization (days).

    Returns
    -------
    dict
        Dictionary with keys 't', 'S', 'E', 'I', 'R', 'D' as JAX arrays.
        - If reps == 1: shapes are (T+1,) for each compartment
        - If reps > 1: shapes are (reps, T+1) for each compartment

    Examples
    --------
    >>> import numpy as np
    >>> from emidm import run_diff_safir_simulation
    >>>
    >>> population = np.array([1000, 2000, 1500])  # 3 age groups
    >>> contact_matrix = np.array([[3, 1, 0.5], [1, 2, 1], [0.5, 1, 1.5]])
    >>> result = run_diff_safir_simulation(
    ...     population=population,
    ...     contact_matrix=contact_matrix,
    ...     R0=2.5,
    ...     T=100,
    ... )
    >>>
    >>> # Multiple replicates
    >>> results = run_diff_safir_simulation(
    ...     population=population,
    ...     contact_matrix=contact_matrix,
    ...     R0=2.5,
    ...     T=100,
    ...     reps=10,
    ... )
    >>> results["I"].shape  # (10, 101)
    """
    _require_jax()
    import jax
    import jax.numpy as jnp

    # Convert to static Python ints for shapes (these are not traced)
    time_horizon = int(T)
    dt_float = float(dt)
    steps_per_day = int(round(1.0 / dt_float))
    if abs(steps_per_day * dt_float - 1.0) > 1e-6:
        raise ValueError("dt must evenly divide 1 day")

    population = jnp.asarray(population, dtype=jnp.float32)
    contact_matrix = jnp.asarray(contact_matrix, dtype=jnp.float32)
    n_age = int(population.shape[0])

    # Default age-dependent vectors
    from .safir import (
        _DEFAULT_PROB_HOSP,
        _DEFAULT_PROB_NON_SEV_DEATH,
        _DEFAULT_PROB_SEV_DEATH,
    )

    def _interp(arr):
        arr = jnp.asarray(arr, dtype=jnp.float32)
        if arr.shape[0] == n_age:
            return arr
        x_old = jnp.linspace(0.0, 1.0, arr.shape[0])
        x_new = jnp.linspace(0.0, 1.0, n_age)
        return jnp.interp(x_new, x_old, arr)

    prob_hosp_arr = _interp(
        _DEFAULT_PROB_HOSP if prob_hosp is None else prob_hosp)
    prob_non_sev_death_arr = _interp(
        _DEFAULT_PROB_NON_SEV_DEATH if prob_non_sev_death is None else prob_non_sev_death
    )
    prob_sev_death_arr = _interp(
        _DEFAULT_PROB_SEV_DEATH if prob_sev_death is None else prob_sev_death
    )
    prob_asymp_val = jnp.asarray(prob_asymp, dtype=jnp.float32)
    frac_ICU_val = jnp.asarray(frac_ICU, dtype=jnp.float32)

    prob_die = frac_ICU_val * prob_sev_death_arr + \
        (1.0 - frac_ICU_val) * prob_non_sev_death_arr
    prob_die = jnp.clip(prob_die, 0.0, 1.0)

    # Compute beta_base from R0 and contact matrix
    rel_inf_period = (1.0 - prob_hosp_arr) * dur_IMild + \
        prob_hosp_arr * dur_ICase
    M = contact_matrix * rel_inf_period[None, :]
    # Non-symmetric eigvals only works on CPU in JAX
    M_cpu = jax.device_put(M, jax.devices("cpu")[0])
    eigvals = jnp.linalg.eigvals(M_cpu)
    beta_base = float(R0) / float(jnp.max(jnp.real(eigvals)))

    # Prepare time-varying beta_t sequence
    days = time_horizon
    if R_t is not None:
        R_t_arr = jnp.asarray(R_t, dtype=jnp.float32)
        if R_t_arr.shape[0] != days + 1:
            raise ValueError(
                f"R_t must have length T + 1 = {days + 1}, got {R_t_arr.shape[0]}"
            )
        beta_t_daily = jnp.asarray(
            beta_base, dtype=jnp.float32) * R_t_arr / jnp.asarray(R0, dtype=jnp.float32)
    else:
        beta_t_daily = jnp.full(days + 1, beta_base, dtype=jnp.float32)

    # Compute population structure
    pop_int = jnp.floor(population).astype(jnp.int32)
    N = int(jnp.sum(pop_int))
    if N == 0:
        zeros = jnp.zeros((days + 1,), dtype=jnp.float32)
        return {"t": jnp.arange(days + 1), "S": zeros, "E": zeros, "I": zeros, "R": zeros, "D": zeros}

    age_index = jnp.repeat(jnp.arange(n_age, dtype=jnp.int32), pop_int)

    # Initialize random key and state
    key = jax.random.PRNGKey(int(seed))
    num_states = 8
    state0 = jax.nn.one_hot(
        jnp.zeros((N,), dtype=jnp.int32), num_states).astype(jnp.float32)
    I0_eff = min(int(I0), N)
    key, k_seed = jax.random.split(key)
    seed_idx = jax.random.choice(k_seed, N, shape=(I0_eff,), replace=False)
    state0 = state0.at[seed_idx, 0].set(
        0.0).at[seed_idx, 1].set(1.0)  # S=0, E1=1

    # Precompute transition probabilities
    p_E = 1.0 - jnp.exp(-dt_float / (dur_E / 2.0))
    p_Iasym = 1.0 - jnp.exp(-dt_float / dur_IMild)
    p_Imild = 1.0 - jnp.exp(-dt_float / dur_IMild)
    p_Icase = 1.0 - jnp.exp(-dt_float / dur_ICase)

    # Per-agent probabilities based on age
    P_Icase = jnp.clip(prob_hosp_arr[age_index], 0.0, 1.0)
    P_Iasym = (1.0 - P_Icase) * jnp.clip(prob_asymp_val, 0.0, 1.0)
    P_Imild = (1.0 - P_Icase) * (1.0 - jnp.clip(prob_asymp_val, 0.0, 1.0))
    prob_die_by_agent = prob_die[age_index]

    # Run the core simulation
    daily_series = _run_diff_safir_core(
        beta_t_daily=beta_t_daily,
        state0=state0,
        key=key,
        population=population,
        contact_matrix=contact_matrix,
        age_index=age_index,
        prob_die_by_agent=prob_die_by_agent,
        P_Icase=P_Icase,
        P_Iasym=P_Iasym,
        P_Imild=P_Imild,
        p_E=p_E,
        p_Iasym=p_Iasym,
        p_Imild=p_Imild,
        p_Icase=p_Icase,
        dt=dt_float,
        tau=config.tau,
        hard=config.hard,
        steps_per_day=steps_per_day,
        n_age=n_age,
        N=N,
    )

    t = jnp.arange(days + 1)

    if reps == 1:
        return {
            "t": t,
            "S": daily_series[:, 0],
            "E": daily_series[:, 1],
            "I": daily_series[:, 2],
            "R": daily_series[:, 3],
            "D": daily_series[:, 4],
        }
    else:
        # Multiple replicates - run sequentially (SAFIR is too complex for vmap)
        # We already ran the first replicate above, now run the rest
        all_results = [daily_series]
        for rep in range(1, reps):
            rep_seed = seed + rep
            rep_key = jax.random.PRNGKey(int(rep_seed))
            rep_state0 = jax.nn.one_hot(
                jnp.zeros((N,), dtype=jnp.int32), num_states).astype(jnp.float32)
            rep_key, rep_k_seed = jax.random.split(rep_key)
            rep_seed_idx = jax.random.choice(
                rep_k_seed, N, shape=(I0_eff,), replace=False)
            rep_state0 = rep_state0.at[rep_seed_idx, 0].set(
                0.0).at[rep_seed_idx, 1].set(1.0)

            rep_daily_series = _run_diff_safir_core(
                beta_t_daily=beta_t_daily,
                state0=rep_state0,
                key=rep_key,
                population=population,
                contact_matrix=contact_matrix,
                age_index=age_index,
                prob_die_by_agent=prob_die_by_agent,
                P_Icase=P_Icase,
                P_Iasym=P_Iasym,
                P_Imild=P_Imild,
                p_E=p_E,
                p_Iasym=p_Iasym,
                p_Imild=p_Imild,
                p_Icase=p_Icase,
                dt=dt_float,
                tau=config.tau,
                hard=config.hard,
                steps_per_day=steps_per_day,
                n_age=n_age,
                N=N,
            )
            all_results.append(rep_daily_series)

        stacked = jnp.stack(all_results)
        return {
            "t": t,
            "S": stacked[:, :, 0],
            "E": stacked[:, :, 1],
            "I": stacked[:, :, 2],
            "R": stacked[:, :, 3],
            "D": stacked[:, :, 4],
        }


def make_diff_safir_model(
    *,
    population,
    contact_matrix,
    R0: float = 2.0,
    T: int = 200,
    dt: float = 0.1,
    seed: int = 0,
    config: DiffConfig = DiffConfig(),
    I0: int = 10,
    prob_hosp=None,
    prob_asymp: float = 0.3,
    prob_non_sev_death=None,
    prob_sev_death=None,
    frac_ICU: float = 0.3,
    dur_E: float = 4.6,
    dur_IMild: float = 2.1,
    dur_ICase: float = 4.5,
):
    """Create a pre-compiled SAFIR model function for gradient-based inference.

    This factory function pre-computes all static values (population structure,
    transition probabilities, etc.) and returns a JIT-compiled function that
    only takes R_t as input. This is much faster for repeated calls during
    optimization or MCMC sampling.

    Parameters
    ----------
    population : array-like
        Population size per age group.
    contact_matrix : array-like
        Contact matrix (n_age x n_age).
    R0 : float, default 2.0
        Basic reproduction number (used for beta calibration).
    T : int, default 200
        Number of days to simulate.
    dt : float, default 0.1
        Sub-daily time step (must evenly divide 1 day).
    seed : int, default 0
        Random seed for reproducibility.
    config : DiffConfig, default DiffConfig()
        Configuration for Gumbel-Softmax (tau, hard).
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

    Returns
    -------
    callable
        A JIT-compiled function `model(params=None)` where `params` is an
        optional dict that can override: 'R_t', 'R0', 'prob_asymp', 'frac_ICU'.
        Returns dict with keys 't', 'S', 'E', 'I', 'R', 'D'.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from emidm import make_diff_safir_model
    >>>
    >>> # Create the model once
    >>> model = make_diff_safir_model(
    ...     population=jnp.array([1000, 2000, 1500]),
    ...     contact_matrix=jnp.eye(3) * 2,
    ...     T=50,
    ... )
    >>>
    >>> # Use defaults
    >>> result = model()
    >>>
    >>> # Override R_t for inference (most common case)
    >>> R_t = jnp.ones(51) * 2.0
    >>> result = model({'R_t': R_t})
    >>>
    >>> # Override multiple params
    >>> result = model({'R_t': R_t, 'prob_asymp': 0.4})
    >>>
    >>> # Compute gradients
    >>> @jax.jit
    ... def loss(R_t):
    ...     return model({'R_t': R_t})['D'][-1]
    >>> grad_fn = jax.grad(loss)
    >>> grads = grad_fn(R_t)
    """
    _require_jax()
    import jax
    import jax.numpy as jnp

    # Convert to static Python ints for shapes
    time_horizon = int(T)
    dt_float = float(dt)
    steps_per_day = int(round(1.0 / dt_float))
    if abs(steps_per_day * dt_float - 1.0) > 1e-6:
        raise ValueError("dt must evenly divide 1 day")

    population = jnp.asarray(population, dtype=jnp.float32)
    contact_matrix = jnp.asarray(contact_matrix, dtype=jnp.float32)
    n_age = int(population.shape[0])

    # Default age-dependent vectors
    from .safir import (
        _DEFAULT_PROB_HOSP,
        _DEFAULT_PROB_NON_SEV_DEATH,
        _DEFAULT_PROB_SEV_DEATH,
    )

    def _interp(arr):
        arr = jnp.asarray(arr, dtype=jnp.float32)
        if arr.shape[0] == n_age:
            return arr
        x_old = jnp.linspace(0.0, 1.0, arr.shape[0])
        x_new = jnp.linspace(0.0, 1.0, n_age)
        return jnp.interp(x_new, x_old, arr)

    prob_hosp_arr = _interp(
        _DEFAULT_PROB_HOSP if prob_hosp is None else prob_hosp)
    prob_non_sev_death_arr = _interp(
        _DEFAULT_PROB_NON_SEV_DEATH if prob_non_sev_death is None else prob_non_sev_death
    )
    prob_sev_death_arr = _interp(
        _DEFAULT_PROB_SEV_DEATH if prob_sev_death is None else prob_sev_death
    )
    prob_asymp_val = jnp.asarray(prob_asymp, dtype=jnp.float32)
    frac_ICU_val = jnp.asarray(frac_ICU, dtype=jnp.float32)

    prob_die = frac_ICU_val * prob_sev_death_arr + \
        (1.0 - frac_ICU_val) * prob_non_sev_death_arr
    prob_die = jnp.clip(prob_die, 0.0, 1.0)

    # Compute beta_base from R0 and contact matrix
    rel_inf_period = (1.0 - prob_hosp_arr) * dur_IMild + \
        prob_hosp_arr * dur_ICase
    M = contact_matrix * rel_inf_period[None, :]
    M_cpu = jax.device_put(M, jax.devices("cpu")[0])
    eigvals = jnp.linalg.eigvals(M_cpu)
    beta_base = float(R0) / float(jnp.max(jnp.real(eigvals)))
    R0_float = float(R0)

    # Compute population structure
    pop_int = jnp.floor(population).astype(jnp.int32)
    N = int(jnp.sum(pop_int))
    days = time_horizon

    if N == 0:
        def _empty_model(R_t):
            zeros = jnp.zeros((days + 1,), dtype=jnp.float32)
            return {"t": jnp.arange(days + 1), "S": zeros, "E": zeros, "I": zeros, "R": zeros, "D": zeros}
        return _empty_model

    age_index = jnp.repeat(jnp.arange(n_age, dtype=jnp.int32), pop_int)

    # Initialize random key and state
    key = jax.random.PRNGKey(int(seed))
    num_states = 8
    state0 = jax.nn.one_hot(
        jnp.zeros((N,), dtype=jnp.int32), num_states).astype(jnp.float32)
    I0_eff = min(int(I0), N)
    key, k_seed = jax.random.split(key)
    seed_idx = jax.random.choice(k_seed, N, shape=(I0_eff,), replace=False)
    state0 = state0.at[seed_idx, 0].set(0.0).at[seed_idx, 1].set(1.0)

    # Precompute transition probabilities
    p_E = 1.0 - jnp.exp(-dt_float / (dur_E / 2.0))
    p_Iasym = 1.0 - jnp.exp(-dt_float / dur_IMild)
    p_Imild = 1.0 - jnp.exp(-dt_float / dur_IMild)
    p_Icase = 1.0 - jnp.exp(-dt_float / dur_ICase)

    # Per-agent probabilities based on age
    P_Icase = jnp.clip(prob_hosp_arr[age_index], 0.0, 1.0)
    P_Iasym = (1.0 - P_Icase) * jnp.clip(prob_asymp_val, 0.0, 1.0)
    P_Imild = (1.0 - P_Icase) * (1.0 - jnp.clip(prob_asymp_val, 0.0, 1.0))
    prob_die_by_agent = prob_die[age_index]

    # Capture defaults in closure for inferrable params
    default_R0 = R0_float
    default_prob_asymp = float(prob_asymp)
    default_frac_ICU = float(frac_ICU)

    # Create the model function that takes optional params dict
    # JIT-compile for fast repeated calls during inference
    @jax.jit
    def _model(params=None):
        # Get param values (from dict or defaults)
        p_R_t = None
        p_R0 = default_R0
        p_prob_asymp = default_prob_asymp
        p_frac_ICU = default_frac_ICU

        if params is not None:
            p_R_t = params.get('R_t', None)
            p_R0 = params.get('R0', p_R0)
            p_prob_asymp = params.get('prob_asymp', p_prob_asymp)
            p_frac_ICU = params.get('frac_ICU', p_frac_ICU)

        # Compute beta_t_daily based on R_t or R0
        if p_R_t is not None:
            R_t_arr = jnp.asarray(p_R_t, dtype=jnp.float32)
            beta_t_daily = jnp.asarray(
                beta_base, dtype=jnp.float32) * R_t_arr / jnp.asarray(default_R0, dtype=jnp.float32)
        else:
            # Use constant R0 (possibly overridden)
            beta_t_daily = jnp.full(
                days + 1,
                beta_base * p_R0 / default_R0,
                dtype=jnp.float32
            )

        # Recompute prob_asymp-dependent values if changed
        p_prob_asymp_val = jnp.asarray(p_prob_asymp, dtype=jnp.float32)
        p_P_Iasym = (1.0 - P_Icase) * jnp.clip(p_prob_asymp_val, 0.0, 1.0)
        p_P_Imild = (1.0 - P_Icase) * \
            (1.0 - jnp.clip(p_prob_asymp_val, 0.0, 1.0))

        # Recompute frac_ICU-dependent values if changed
        p_frac_ICU_val = jnp.asarray(p_frac_ICU, dtype=jnp.float32)
        p_prob_die = p_frac_ICU_val * prob_sev_death_arr + \
            (1.0 - p_frac_ICU_val) * prob_non_sev_death_arr
        p_prob_die = jnp.clip(p_prob_die, 0.0, 1.0)
        p_prob_die_by_agent = p_prob_die[age_index]

        daily_series = _run_diff_safir_core(
            beta_t_daily=beta_t_daily,
            state0=state0,
            key=key,
            population=population,
            contact_matrix=contact_matrix,
            age_index=age_index,
            prob_die_by_agent=p_prob_die_by_agent,
            P_Icase=P_Icase,
            P_Iasym=p_P_Iasym,
            P_Imild=p_P_Imild,
            p_E=p_E,
            p_Iasym=p_Iasym,
            p_Imild=p_Imild,
            p_Icase=p_Icase,
            dt=dt_float,
            tau=config.tau,
            hard=config.hard,
            steps_per_day=steps_per_day,
            n_age=n_age,
            N=N,
        )

        t = jnp.arange(days + 1)
        return {
            "t": t,
            "S": daily_series[:, 0],
            "E": daily_series[:, 1],
            "I": daily_series[:, 2],
            "R": daily_series[:, 3],
            "D": daily_series[:, 4],
        }

    return _model
