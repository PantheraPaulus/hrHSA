from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarking"
    / "scripts"
    / "01_python"
    / "build_documentation_benchmark_report.py"
)
SPEC = importlib.util.spec_from_file_location("build_documentation_benchmark_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_raster_summary_and_recommended_configuration():
    frame = pd.DataFrame(
        {
            "benchmark": [MODULE.SURFACE_BENCHMARK] * 6,
            "workers": [1, 1, 2, 2, 4, 4],
            "threads_per_worker": [1] * 6,
            "wall_seconds": [10.0, 10.2, 5.2, 5.0, 4.9, 4.8],
            "throughput_rows_s": [10e6, 9.8e6, 19.2e6, 20e6, 20.4e6, 20.8e6],
            "rows": [100_000_000] * 6,
            "bytes_processed": [2_400_000_000] * 6,
            "metadata.worker_geometry": ["1x1", "1x1", "2x1", "2x1", "4x1", "4x1"],
        }
    )

    summary = MODULE._raster_summary(frame)
    two = summary.loc[summary["geometry"].eq("2x1")].iloc[0]
    four = summary.loc[summary["geometry"].eq("4x1")].iloc[0]

    assert two["speedup_vs_smallest"] == pytest.approx(10.1 / 5.1)
    assert four["relative_to_fastest"] == pytest.approx(1.0)

    # 2x1 is just outside the 5% band here, so the fastest 4x1 row is selected.
    selected = MODULE._recommended_row(summary, MODULE.SURFACE_BENCHMARK)
    assert selected["geometry"] == "4x1"

    within_five = summary.copy()
    within_five.loc[within_five["geometry"].eq("2x1"), "median_seconds"] = 4.95
    selected = MODULE._recommended_row(within_five, MODULE.SURFACE_BENCHMARK)
    assert selected["geometry"] == "2x1"


def test_mcmc_summary_uses_quality_adjusted_throughput_and_guardrails():
    frame = pd.DataFrame(
        {
            "sampler": ["pymc", "pymc", "blackjax", "blackjax"],
            "status": ["success"] * 4,
            "completed_sampling_seconds": [100.0, 102.0, 60.0, 62.0],
            "median_ess_bulk_per_completed_second": [12.0, 11.0, 18.0, 17.0],
            "min_ess_bulk_per_completed_second": [7.0, 6.5, 9.0, 8.5],
            "max_rhat": [1.005, 1.006, 1.008, 1.009],
            "n_divergences": [0, 0, 0, 0],
            "operation_peak_process_tree_rss_mb": [1000, 1100, 1500, 1550],
            "posterior_mib": [50, 50, 52, 52],
        }
    )

    summary = MODULE._mcmc_summary(frame)
    blackjax = summary.loc[summary["sampler"].eq("blackjax")].iloc[0]
    pymc = summary.loc[summary["sampler"].eq("pymc")].iloc[0]

    assert blackjax["median_bulk_ess_s"] == pytest.approx(17.5)
    assert blackjax["median_seconds"] == pytest.approx(61.0)
    assert bool(blackjax["quality_guardrail_pass"])
    assert bool(pymc["quality_guardrail_pass"])

    frame.loc[frame["sampler"].eq("blackjax"), "n_divergences"] = 1
    summary = MODULE._mcmc_summary(frame)
    blackjax = summary.loc[summary["sampler"].eq("blackjax")].iloc[0]
    assert not bool(blackjax["quality_guardrail_pass"])
