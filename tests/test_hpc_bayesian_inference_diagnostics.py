from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "benchmarking" / "scripts" / "01_python"


def _load(name: str, filename: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    inserted = str(SCRIPTS) not in sys.path
    if inserted:
        sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(str(SCRIPTS))
    return module


def test_physical_cpu_representatives_are_allowed_and_unique():
    module = _load(
        "benchmark_bayesian_rsf_contention",
        "benchmark_bayesian_rsf_contention.py",
    )
    cpus = module._physical_cpu_representatives()
    assert cpus
    assert len(cpus) == len(set(cpus))

    if hasattr(module.os, "sched_getaffinity"):
        allowed = set(module.os.sched_getaffinity(0))
        assert set(cpus).issubset(allowed)


def test_synthetic_issf_has_one_choice_and_centered_offset():
    module = _load(
        "benchmark_bayesian_ssf_completion",
        "benchmark_bayesian_ssf_completion.py",
    )
    frame, names, offset_col = module._synthetic_choices(
        n_strata=20,
        n_choices=5,
        predictors=6,
        individuals=4,
        analysis="issf",
        seed=123,
    )

    assert names[-2:] == ["log_sl", "cos_ta"]
    assert offset_col == "proposal_offset"
    grouped = frame.groupby(["id", "stratum_id"], sort=False)
    assert grouped["used"].sum().eq(1).all()
    assert grouped.size().eq(5).all()
    np.testing.assert_allclose(
        grouped[offset_col].mean().to_numpy(),
        0.0,
        atol=1e-6,
    )


def test_chain_method_sequence_medians_separate_cold_and_warm_folds():
    module = _load(
        "benchmark_bayesian_rsf_chain_method",
        "benchmark_bayesian_rsf_chain_method.py",
    )
    rows = [
        {
            "status": "success",
            "worker_fold_sequence": 1,
            "completed_sampling_seconds": 10.0,
        },
        {
            "status": "success",
            "worker_fold_sequence": 1,
            "completed_sampling_seconds": 14.0,
        },
        {
            "status": "success",
            "worker_fold_sequence": 2,
            "completed_sampling_seconds": 4.0,
        },
        {
            "status": "failed",
            "worker_fold_sequence": 2,
            "completed_sampling_seconds": 999.0,
        },
    ]

    assert module._sequence_medians(rows) == {
        "sequence_1": 12.0,
        "sequence_2": 4.0,
    }


def test_chain_method_failure_reporting_handles_all_failed_folds():
    module = _load(
        "benchmark_bayesian_rsf_chain_method_failure",
        "benchmark_bayesian_rsf_chain_method.py",
    )
    rows = [
        {
            "status": "failed",
            "stage": "sample_dispatch",
            "error": "ValueError: first failure",
        },
        {
            "status": "failed",
            "stage": "sample_dispatch",
            "error": "ValueError: first failure",
        },
        {
            "status": "failed",
            "stage": "materialize",
            "error": "RuntimeError: second failure",
        },
    ]

    assert module._median(rows, "wall_seconds") is None
    assert module._format_seconds(None) == "NA"
    assert module._format_seconds(np.nan) == "NA"
    assert module._format_seconds(1.23456) == "1.235s"
    assert module._distinct_failure_examples(rows) == [
        "sample_dispatch: ValueError: first failure",
        "materialize: RuntimeError: second failure",
    ]


def test_chain_method_default_reproduces_historical_sampler_kwargs():
    module = _load(
        "benchmark_bayesian_rsf_chain_method_control",
        "benchmark_bayesian_rsf_chain_method.py",
    )

    class FakePM:
        __version__ = "6.3.1"

        @staticmethod
        def sample(
            *,
            nuts_sampler=None,
            compute_convergence_checks=True,
            blas_cores="auto",
            **kwargs,
        ):
            return None

    common = dict(
        sampler="blackjax",
        draws=250,
        tune=250,
        chains=4,
        cores=1,
        target_accept=0.9,
        seed=43,
    )
    historical, historical_mode = module._configure_sampling(
        FakePM,
        chain_method="default",
        **common,
    )
    explicit, explicit_mode = module._configure_sampling(
        FakePM,
        chain_method="vectorized",
        **common,
    )

    assert historical_mode == "historical_default"
    assert historical["nuts_sampler"] == "blackjax"
    assert "nuts" not in historical
    assert "nuts_sampler_kwargs" not in historical

    assert explicit_mode == "explicit_chain_method"
    assert explicit["nuts_sampler"] == "blackjax"
    assert explicit["nuts"] == {"chain_method": "vectorized"}
