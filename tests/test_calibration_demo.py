import pytest


jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")


def test_calibration_demo_recovers_beta_reasonably():
    from emidm.examples.calibration_demo import make_synthetic_data, fit_beta_to_data

    data = make_synthetic_data(beta_true=0.33, gamma=0.2, T=20, N=120)
    beta_hat, hist = fit_beta_to_data(data["I"], gamma=0.2, T=20, N=120)

    assert hist["loss"][0] > hist["loss"][-1]
    assert abs(float(beta_hat) - 0.33) < 0.15
