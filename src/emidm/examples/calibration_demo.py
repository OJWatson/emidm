from emidm.diff import DiffConfig, run_diff_sir
from emidm.optim import optimize_params


def _require_jax():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "This example requires 'jax'. Install with `pip install emidm[jax]` (or install jax/jaxlib)."
        ) from e


def make_synthetic_data(*, beta_true=0.35, gamma=0.2, T=30, N=200, seed=0):
    _require_jax()
    out = run_diff_sir(
        N=N,
        I0=5,
        beta=beta_true,
        gamma=gamma,
        T=T,
        config=DiffConfig(tau=0.8, hard=True),
        seed=seed,
    )
    return out


def fit_beta_to_data(observed_I, *, gamma=0.2, T=30, N=200):
    _require_jax()
    import jax.numpy as jnp

    def loss_fn(beta):
        pred = run_diff_sir(
            N=N,
            I0=5,
            beta=beta,
            gamma=gamma,
            T=T,
            config=DiffConfig(tau=0.8, hard=True),
            seed=0,
        )
        return jnp.mean((pred["I"] - observed_I) ** 2)

    beta0 = jnp.array(0.15)
    beta_hat, hist = optimize_params(
        loss_fn=loss_fn, init_params=beta0, n_steps=150, learning_rate=0.15)
    return beta_hat, hist


def run_demo():
    _require_jax()
    data = make_synthetic_data()
    beta_hat, hist = fit_beta_to_data(data["I"])
    return {"beta_hat": beta_hat, "history": hist}
