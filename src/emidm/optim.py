"""Optimization utilities for gradient-based parameter calibration.

This module provides loss functions and optimization routines for fitting
epidemiological model parameters using JAX and Optax.
"""
from __future__ import annotations


def _require_jax():
    """Check that JAX and Optax are available."""
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        import optax  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Optimization utilities require 'jax' and 'optax'. "
            "Install them with `pip install jax optax`."
        ) from e


def mse_loss(predicted, observed):
    """Mean squared error loss function.

    Parameters
    ----------
    predicted : array-like
        Predicted values from model.
    observed : array-like
        Observed/target values.

    Returns
    -------
    float
        Mean squared error.
    """
    _require_jax()
    import jax.numpy as jnp

    predicted = jnp.asarray(predicted)
    observed = jnp.asarray(observed)
    return jnp.mean((predicted - observed) ** 2)


def poisson_nll(predicted, observed, eps: float = 1e-8):
    """Poisson negative log-likelihood loss.

    Useful for count data (e.g., number of infections).

    Parameters
    ----------
    predicted : array-like
        Predicted rates/counts (must be positive).
    observed : array-like
        Observed counts.
    eps : float, default 1e-8
        Small constant to avoid log(0).

    Returns
    -------
    float
        Negative log-likelihood.
    """
    _require_jax()
    import jax.numpy as jnp

    predicted = jnp.asarray(predicted)
    observed = jnp.asarray(observed)
    # NLL = sum(predicted - observed * log(predicted))
    return jnp.mean(predicted - observed * jnp.log(predicted + eps))


def gaussian_nll(predicted, observed, sigma: float = 1.0):
    """Gaussian negative log-likelihood loss.

    Parameters
    ----------
    predicted : array-like
        Predicted values.
    observed : array-like
        Observed values.
    sigma : float, default 1.0
        Standard deviation of the Gaussian.

    Returns
    -------
    float
        Negative log-likelihood (up to a constant).
    """
    _require_jax()
    import jax.numpy as jnp

    predicted = jnp.asarray(predicted)
    observed = jnp.asarray(observed)
    return jnp.mean(0.5 * ((predicted - observed) / sigma) ** 2)


def make_sir_loss(
    observed_I,
    *,
    model_fn,
    model_kwargs: dict | None = None,
    loss_type: str = "mse",
):
    """Create a loss function for SIR model calibration.

    Parameters
    ----------
    observed_I : array-like
        Observed infection counts over time.
    model_fn : callable
        Model function (e.g., run_diff_sir) that takes `beta` as first positional arg.
    model_kwargs : dict, optional
        Additional keyword arguments to pass to model_fn.
    loss_type : str, default "mse"
        Loss type: "mse", "poisson", or "gaussian".

    Returns
    -------
    callable
        Loss function that takes params and returns scalar loss.
    """
    _require_jax()
    import jax.numpy as jnp

    observed_I = jnp.asarray(observed_I)
    model_kwargs = model_kwargs or {}

    if loss_type == "mse":
        loss_fn = mse_loss
    elif loss_type == "poisson":
        loss_fn = poisson_nll
    elif loss_type == "gaussian":
        loss_fn = gaussian_nll
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    def _loss(params):
        result = model_fn(beta=params, **model_kwargs)
        predicted_I = result["I"]
        return loss_fn(predicted_I, observed_I)

    return _loss


def optimize_params(*, loss_fn, init_params, n_steps: int = 200, learning_rate: float = 1e-2):
    """Gradient-based parameter optimization using Optax (Adam).

    Parameters
    ----------
    loss_fn : callable
        Differentiable loss function that takes params and returns scalar loss.
    init_params : array-like
        Initial parameter values.
    n_steps : int, default 200
        Number of optimization steps.
    learning_rate : float, default 0.01
        Learning rate for Adam optimizer.

    Returns
    -------
    tuple[array, dict]
        Final optimized parameters and history dict with 'loss' and 'params' arrays.

    Examples
    --------
    >>> from emidm import optimize_params, mse_loss, run_diff_sir
    >>> import jax.numpy as jnp
    >>>
    >>> # Define loss function
    >>> def loss_fn(beta):
    ...     result = run_diff_sir(N=100, I0=5, beta=beta, T=30)
    ...     return mse_loss(result["I"], observed_data)
    >>>
    >>> # Optimize
    >>> beta_hat, history = optimize_params(
    ...     loss_fn=loss_fn,
    ...     init_params=jnp.array(0.15),
    ...     n_steps=100
    ... )
    """
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "optimize_params requires 'jax' and 'optax'. Install them first (e.g. `pip install optax`)."
        ) from e

    params = init_params

    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    params_history = []
    for _ in range(int(n_steps)):
        params, opt_state, loss = step(params, opt_state)
        losses.append(loss)
        params_history.append(params)

    history = {"loss": jnp.stack(losses), "params": jnp.stack(params_history)}
    return params, history
