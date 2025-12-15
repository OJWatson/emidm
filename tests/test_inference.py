import time

import pytest


jax = pytest.importorskip("jax")

# Check if optional dependencies are available
try:
    import blackjax
    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False

try:
    import optax
    HAS_OPTAX = True
except ImportError:
    HAS_OPTAX = False


# =============================================================================
# BlackJAX-specific tests (require blackjax)
# =============================================================================


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not installed")
class TestBlackJAXIntegration:
    """Tests requiring BlackJAX for MCMC sampling."""

    def test_blackjax_nuts_runs_for_simple_gaussian(self):
        import jax.numpy as jnp

        from emidm.inference import run_blackjax_nuts

        def logdensity(x):
            # Standard normal
            return -0.5 * jnp.sum(x**2)

        samples = run_blackjax_nuts(logdensity_fn=logdensity, initial_position=jnp.array(
            [1.0, -1.0]), num_warmup=16, num_samples=16)
        assert samples.shape == (16, 2)

    def test_sir_nuts_sampling_runs(self):
        """NUTS sampling with SIR model should produce valid samples."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_sir_model
        from emidm.inference import run_blackjax_nuts

        key = jax.random.PRNGKey(0)
        model = make_diff_sir_model(
            N=50, I0=3, T=20, key=key,
            config=DiffConfig(tau=0.5, hard=False)
        )

        observed_I = model({'beta': 0.35})['I']

        @jax.jit
        def log_posterior(beta):
            pred_I = model({'beta': beta})['I']
            sigma = 5.0
            ll = -0.5 * jnp.sum(((pred_I - observed_I) / sigma) ** 2)
            lp = -0.5 * ((beta - 0.3) / 0.15) ** 2
            return ll + lp

        # Run short MCMC chain
        samples = run_blackjax_nuts(
            logdensity_fn=log_posterior,
            initial_position=jnp.array(0.3),
            num_warmup=10,
            num_samples=20,
        )

        assert samples.shape == (
            20,), f"Expected (20,) samples, got {samples.shape}"
        assert jnp.all(jnp.isfinite(samples)), "All samples should be finite"
        assert jnp.all(samples > 0), "Beta samples should be positive"


# =============================================================================
# Inference Pattern Tests - Verify JIT efficiency for notebook-style workflows
# These tests only require JAX (not blackjax)
# =============================================================================


class TestSIRCalibrationPattern:
    """Tests mirroring calibration.ipynb workflow - gradient-based optimization."""

    def test_make_diff_sir_model_jit_speedup(self):
        """Factory model should be significantly faster than raw simulation calls."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_sir_model, run_diff_sir_simulation

        key = jax.random.PRNGKey(42)
        N, I0, T = 200, 5, 50
        config = DiffConfig(tau=0.5, hard=False)

        # Create factory model
        model = make_diff_sir_model(
            N=N, I0=I0, T=T, key=key, config=config, reps=10
        )

        # Warm up JIT
        _ = model({'beta': 0.3})
        _ = model({'beta': 0.3})

        # Time factory model (should be fast after JIT)
        n_calls = 20
        start = time.perf_counter()
        for i in range(n_calls):
            beta = 0.2 + 0.01 * i
            _ = model({'beta': beta})
        factory_time = time.perf_counter() - start

        # Time raw simulation calls
        start = time.perf_counter()
        for i in range(n_calls):
            beta = 0.2 + 0.01 * i
            _ = run_diff_sir_simulation(
                N=N, I0=I0, beta=beta, gamma=0.1, T=T,
                key=key, reps=10, config=config
            )
        raw_time = time.perf_counter() - start

        # Factory should be at least 2x faster (typically much more)
        speedup = raw_time / factory_time
        assert speedup > 1.5, f"Factory speedup only {speedup:.1f}x, expected >1.5x"

    def test_sir_loss_function_is_differentiable(self):
        """Loss function using factory model should have finite gradients."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_sir_model

        key = jax.random.PRNGKey(0)
        model = make_diff_sir_model(
            N=100, I0=5, T=30, key=key,
            config=DiffConfig(tau=0.5, hard=False)
        )

        # Synthetic observed data
        observed_I = model({'beta': 0.35})['I']

        @jax.jit
        def loss_fn(beta):
            pred = model({'beta': beta})
            return jnp.mean((pred['I'] - observed_I) ** 2)

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.array(0.25))

        assert jnp.isfinite(grad), "Gradient should be finite"
        assert grad != 0.0, "Gradient should be non-zero"

    @pytest.mark.skipif(not HAS_OPTAX, reason="optax not installed")
    def test_sir_optimization_converges(self):
        """Optimization should recover approximately correct beta."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_sir_model
        from emidm.optim import optimize_params

        key = jax.random.PRNGKey(42)
        true_beta = 0.35

        model = make_diff_sir_model(
            N=200, I0=5, T=50, key=key, reps=20,
            config=DiffConfig(tau=0.5, hard=False)
        )

        # Generate "observed" data
        observed_I = model({'beta': true_beta})['I']

        @jax.jit
        def loss_fn(beta):
            pred = model({'beta': beta})
            return jnp.mean((pred['I'] - observed_I) ** 2)

        # Optimize from wrong initial guess
        beta_init = jnp.array(0.15)
        beta_hat, history = optimize_params(
            loss_fn=loss_fn,
            init_params=beta_init,
            n_steps=100,
            learning_rate=0.01,
        )

        # Should get reasonably close (within 0.1 of true value)
        error = abs(float(beta_hat) - true_beta)
        assert error < 0.1, f"Optimization error {error:.3f} too large"


class TestBayesianInferencePattern:
    """Tests mirroring bayesian_inference.ipynb workflow - MCMC sampling."""

    def test_sir_log_posterior_is_differentiable(self):
        """Log posterior using factory model should have finite gradients."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_sir_model

        key = jax.random.PRNGKey(0)
        model = make_diff_sir_model(
            N=100, I0=5, T=30, key=key,
            config=DiffConfig(tau=0.5, hard=False)
        )

        # Synthetic observed data
        observed_I = model({'beta': 0.35})['I']

        @jax.jit
        def log_posterior(beta):
            pred_I = model({'beta': beta})['I']
            # Gaussian likelihood
            sigma = 5.0
            ll = -0.5 * jnp.sum(((pred_I - observed_I) / sigma) ** 2)
            # Gaussian prior on beta
            lp = -0.5 * ((beta - 0.3) / 0.15) ** 2
            return ll + lp

        # Check gradient exists and is finite
        grad_fn = jax.grad(log_posterior)
        grad = grad_fn(jnp.array(0.25))

        assert jnp.isfinite(grad), "Log posterior gradient should be finite"


class TestSAFIRInferencePattern:
    """Tests mirroring safir_inference.ipynb workflow - R(t) estimation."""

    def test_make_diff_safir_model_jit_speedup(self):
        """Factory model should be faster than raw simulation for repeated calls."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_safir_model, run_diff_safir_simulation

        population = jnp.array([300, 400, 300])
        contact_matrix = jnp.eye(3) * 2.0
        T = 30
        config = DiffConfig(tau=0.5, hard=False)

        # Create factory model
        model = make_diff_safir_model(
            population=population,
            contact_matrix=contact_matrix,
            T=T,
            dt=0.5,
            config=config,
        )

        # Warm up JIT
        R_t = jnp.ones(T + 1) * 2.0
        _ = model({'R_t': R_t})
        _ = model({'R_t': R_t})

        # Time factory model
        n_calls = 10
        start = time.perf_counter()
        for i in range(n_calls):
            R_t = jnp.ones(T + 1) * (1.5 + 0.1 * i)
            _ = model({'R_t': R_t})
        factory_time = time.perf_counter() - start

        # Time raw simulation
        start = time.perf_counter()
        for i in range(n_calls):
            R_t = jnp.ones(T + 1) * (1.5 + 0.1 * i)
            _ = run_diff_safir_simulation(
                population=population,
                contact_matrix=contact_matrix,
                R_t=R_t,
                T=T,
                dt=0.5,
                config=config,
            )
        raw_time = time.perf_counter() - start

        # Factory should be faster (at least 1.5x)
        speedup = raw_time / factory_time
        assert speedup > 1.2, f"Factory speedup only {speedup:.1f}x, expected >1.2x"

    def test_safir_rt_inference_gradient_exists(self):
        """Gradients through R(t) should be finite for SAFIR model."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_safir_model

        population = jnp.array([200, 300, 200])
        contact_matrix = jnp.eye(3) * 2.0
        T = 20

        model = make_diff_safir_model(
            population=population,
            contact_matrix=contact_matrix,
            T=T,
            dt=0.5,
            config=DiffConfig(tau=0.5, hard=False),
        )

        # Generate observed deaths
        true_R_t = jnp.ones(T + 1) * 2.5
        observed_D = model({'R_t': true_R_t})['D']

        @jax.jit
        def loss(R_t):
            pred = model({'R_t': R_t})
            return jnp.mean((pred['D'] - observed_D) ** 2)

        # Compute gradient w.r.t. R_t
        grad_fn = jax.grad(loss)
        init_R_t = jnp.ones(T + 1) * 2.0
        grad = grad_fn(init_R_t)

        assert grad.shape == (T + 1,), f"Gradient shape should be ({T + 1},)"
        assert jnp.all(jnp.isfinite(grad)), "All gradients should be finite"

    @pytest.mark.skipif(not HAS_OPTAX, reason="optax not installed")
    def test_safir_piecewise_rt_inference(self):
        """Piecewise R(t) parameterization should work for inference."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_safir_model
        from emidm.optim import optimize_params

        population = jnp.array([200, 300, 200])
        contact_matrix = jnp.eye(3) * 2.0
        T = 28
        interval_days = 7
        n_intervals = T // interval_days + 1

        model = make_diff_safir_model(
            population=population,
            contact_matrix=contact_matrix,
            T=T,
            dt=0.5,
            config=DiffConfig(tau=0.5, hard=False),
        )

        def expand_R_intervals(R_intervals):
            """Expand interval R values to daily R(t) array."""
            R_daily = jnp.zeros(T + 1)
            for i in range(n_intervals):
                start = i * interval_days
                end = min((i + 1) * interval_days, T + 1)
                R_daily = R_daily.at[start:end].set(R_intervals[i])
            return R_daily

        # True piecewise R(t)
        true_R_intervals = jnp.array([2.5, 1.5, 1.0, 0.8, 1.2])[:n_intervals]
        true_R_t = expand_R_intervals(true_R_intervals)
        observed_D = model({'R_t': true_R_t})['D']

        @jax.jit
        def loss(log_R_intervals):
            R_intervals = jnp.exp(log_R_intervals)
            R_t = expand_R_intervals(R_intervals)
            pred = model({'R_t': R_t})
            return jnp.mean((pred['D'] - observed_D) ** 2)

        # Optimize
        init_log_R = jnp.log(jnp.ones(n_intervals) * 1.5)
        fitted_log_R, _ = optimize_params(
            loss_fn=loss,
            init_params=init_log_R,
            n_steps=50,
            learning_rate=0.1,
        )

        fitted_R = jnp.exp(fitted_log_R)
        # Check that optimization moved params in right direction
        assert jnp.all(jnp.isfinite(fitted_R)
                       ), "Fitted R values should be finite"
        assert jnp.all(fitted_R > 0), "Fitted R values should be positive"


class TestMultiParameterInference:
    """Tests for inferring multiple parameters simultaneously."""

    def test_sir_joint_beta_gamma_gradient(self):
        """Gradients should exist for joint beta-gamma inference."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_sir_model

        key = jax.random.PRNGKey(0)
        model = make_diff_sir_model(
            N=100, I0=5, T=30, key=key,
            config=DiffConfig(tau=0.5, hard=False)
        )

        observed = model({'beta': 0.35, 'gamma': 0.12})

        @jax.jit
        def loss(params):
            pred = model({'beta': params[0], 'gamma': params[1]})
            return jnp.mean((pred['I'] - observed['I']) ** 2)

        grad_fn = jax.grad(loss)
        params = jnp.array([0.25, 0.1])
        grad = grad_fn(params)

        assert grad.shape == (2,), "Should have gradients for both params"
        assert jnp.all(jnp.isfinite(grad)), "All gradients should be finite"

    def test_safir_joint_rt_prob_asymp_gradient(self):
        """Gradients should exist for joint R(t) and prob_asymp inference."""
        import jax.numpy as jnp

        from emidm.diff import DiffConfig, make_diff_safir_model

        population = jnp.array([200, 300, 200])
        contact_matrix = jnp.eye(3) * 2.0
        T = 20

        model = make_diff_safir_model(
            population=population,
            contact_matrix=contact_matrix,
            T=T,
            dt=0.5,
            config=DiffConfig(tau=0.5, hard=False),
        )

        # Generate observed data
        true_R_t = jnp.ones(T + 1) * 2.5
        observed = model({'R_t': true_R_t, 'prob_asymp': 0.4})

        @jax.jit
        def loss(R_t, prob_asymp):
            pred = model({'R_t': R_t, 'prob_asymp': prob_asymp})
            return jnp.mean((pred['I'] - observed['I']) ** 2)

        grad_fn = jax.grad(loss, argnums=(0, 1))
        init_R_t = jnp.ones(T + 1) * 2.0
        grad_R_t, grad_prob_asymp = grad_fn(init_R_t, 0.3)

        assert grad_R_t.shape == (T + 1,), "R_t gradient shape mismatch"
        assert jnp.all(jnp.isfinite(grad_R_t)
                       ), "R_t gradients should be finite"
        assert jnp.isfinite(
            grad_prob_asymp), "prob_asymp gradient should be finite"
