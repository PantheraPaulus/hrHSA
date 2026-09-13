from __future__ import annotations

import importlib.util
from pathlib import Path

from hsa import FeatureSpec


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarking"
    / "scripts"
    / "01_python"
    / "benchmark_cv_scaling.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_cv_scaling", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cv_scaling_cache_and_frequentist_fold(tmp_path):
    prepared = MODULE._prepare_synthetic_cache(
        tmp_path / "cache",
        folds=3,
        rows_per_individual=80,
        predictors=2,
        seed=42,
    )

    assert len(prepared.individuals) == 3
    assert int(prepared.manifest["n_rows"].sum()) == 240

    result = MODULE._frequentist_fold(
        prepared_root=str(prepared.root),
        heldout_id=prepared.individuals[0],
        spec=FeatureSpec(
            linear=["x0", "x1"],
            quadratic=["x0"],
            interactions=[("x0", "x1")],
            add_const=True,
        ),
        method="lbfgs",
        maxiter=100,
        blas_threads=1,
    )

    assert result["status"] == "success"
    assert result["n_train"] == 160
    assert result["fit_seconds"] >= 0
    assert result["load_seconds"] >= 0
