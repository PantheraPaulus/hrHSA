from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarking"
        / "scripts"
        / "01_python"
        / "benchmark_ssf_inference.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_ssf_inference_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_ssf_has_one_choice_per_stratum():
    module = _load_module()
    frame, names = module._synthetic_choices(
        n_strata=100,
        n_choices=5,
        predictors=4,
        individuals=10,
        analysis="ssf",
        seed=42,
    )
    assert len(names) == 4
    assert len(frame) == 500
    chosen = frame.groupby("stratum_id", sort=False)["used"].sum().to_numpy()
    assert np.array_equal(chosen, np.ones(100, dtype=chosen.dtype))


def test_synthetic_issf_includes_movement_predictors():
    module = _load_module()
    frame, names = module._synthetic_choices(
        n_strata=50,
        n_choices=4,
        predictors=5,
        individuals=5,
        analysis="issf",
        seed=42,
    )
    assert "log_sl" in names
    assert "cos_ta" in names
    assert frame["cos_ta"].between(-1, 1).all()


def test_per_id_benchmark_helper_fits_one_individual():
    module = _load_module()
    frame, names = module._synthetic_choices(
        n_strata=80,
        n_choices=5,
        predictors=4,
        individuals=4,
        analysis="ssf",
        seed=43,
    )
    group = next(iter(frame.groupby("id", sort=False)))[1].copy()
    converged, llf, n_rows = module._fit_individual(
        (group, names, "lbfgs", 200)
    )

    assert converged
    assert np.isfinite(llf)
    assert n_rows == len(group)
