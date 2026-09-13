from __future__ import annotations

import importlib.util
from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "benchmarking"
    / "scripts"
    / "01_python"
)


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_script("benchmark_cv_scaling", "benchmark_cv_scaling.py")
MODULE = _load_script(
    "benchmark_bayesian_rsf_hmc_vectorized",
    "benchmark_bayesian_rsf_hmc_vectorized.py",
)


def test_hmc_stats_do_not_require_nuts_tree_fields():
    state = SimpleNamespace(logdensity=-3.5)
    info = SimpleNamespace(
        is_divergent=False,
        energy=4.25,
        num_integration_steps=7,
        acceptance_rate=0.91,
    )

    stats = MODULE._hmc_info_stats(state, info)

    assert stats == {
        "diverging": False,
        "energy": 4.25,
        "n_steps": 7,
        "acceptance_rate": 0.91,
        "lp": -3.5,
    }
    assert "tree_depth" not in stats


def test_vectorized_static_hmc_uses_fixed_integration_steps():
    pytest.importorskip("blackjax")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def logp(position):
        return -0.5 * jnp.sum(position**2)

    chains = 2
    draws = 4
    integration_steps = 3
    keys = jax.random.split(jax.random.PRNGKey(123), chains)
    initial_positions = jnp.asarray([[0.1, -0.1], [0.25, 0.2]])

    run_chain = partial(
        MODULE._blackjax_hmc_inference_loop,
        logp_fn=logp,
        draws=draws,
        tune=25,
        target_accept=0.8,
        integration_steps=integration_steps,
    )
    positions, stats, step_sizes = jax.vmap(run_chain)(keys, initial_positions)
    positions = np.asarray(positions)
    n_steps = np.asarray(stats["n_steps"])
    acceptance = np.asarray(stats["acceptance_rate"])
    step_sizes = np.asarray(step_sizes)

    assert positions.shape == (chains, draws, 2)
    assert n_steps.shape == (chains, draws)
    assert np.all(n_steps == integration_steps)
    assert np.all(np.isfinite(acceptance))
    assert np.all((acceptance >= 0.0) & (acceptance <= 1.0))
    assert step_sizes.shape == (chains,)
    assert np.all(np.isfinite(step_sizes))
    assert np.all(step_sizes > 0.0)
