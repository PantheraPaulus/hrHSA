"""Validate Bayesian sampler backends in fresh Python processes.

A very large apparent backend speedup should not be promoted to an hrHSA default
from one warm-process benchmark. This launcher executes one sampler/replicate per
fresh Python process, keeps the synthetic statistical problem fixed, varies the
sampling seed, saves compact posterior summaries, and compares posterior location
and scale against a nominated reference sampler.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "benchmark_inference.py"


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_record(rows: list[dict[str, Any]], sampler: str) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("benchmark") == "inference_bayesian_nuts"
            and row.get("metadata", {}).get("nuts_sampler") == sampler
        ):
            return row
    return None


def _failure_record(rows: list[dict[str, Any]], sampler: str) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("benchmark") == "inference_bayesian_failure"
            and row.get("metadata", {}).get("nuts_sampler") == sampler
        ):
            return row
    return None


def _summary_path(rows: list[dict[str, Any]], sampler: str) -> Path | None:
    record = _sample_record(rows, sampler)
    if record is None:
        return None
    value = record.get("metadata", {}).get("posterior_summary")
    return Path(value) if value else None


def _flatten_run_record(
    *,
    sampler: str,
    repeat: int,
    process_seconds: float,
    returncode: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sample = _sample_record(rows, sampler)
    failure = _failure_record(rows, sampler)
    if sample is not None:
        meta = sample.get("metadata", {})
        return {
            "sampler": sampler,
            "repeat": repeat,
            "status": "success",
            "process_seconds": process_seconds,
            "sample_seconds": sample.get("wall_seconds"),
            "returncode": returncode,
            "median_ess_bulk": meta.get("median_ess_bulk"),
            "min_ess_bulk": meta.get("min_ess_bulk"),
            "median_ess_bulk_per_second": meta.get("median_ess_bulk_per_second"),
            "min_ess_bulk_per_second": meta.get("min_ess_bulk_per_second"),
            "min_ess_tail": meta.get("min_ess_tail"),
            "max_rhat": meta.get("max_rhat"),
            "n_divergences": meta.get("n_divergences"),
            "mean_n_steps": meta.get("mean_n_steps"),
            "sampling_seed": meta.get("sampling_seed"),
            "posterior_mib": meta.get("posterior_mib"),
            "posterior_summary": meta.get("posterior_summary"),
            "exception_type": None,
            "exception_message": None,
        }
    meta = failure.get("metadata", {}) if failure is not None else {}
    return {
        "sampler": sampler,
        "repeat": repeat,
        "status": "failed",
        "process_seconds": process_seconds,
        "sample_seconds": failure.get("wall_seconds") if failure is not None else np.nan,
        "returncode": returncode,
        "median_ess_bulk": np.nan,
        "min_ess_bulk": np.nan,
        "median_ess_bulk_per_second": np.nan,
        "min_ess_bulk_per_second": np.nan,
        "min_ess_tail": np.nan,
        "max_rhat": np.nan,
        "n_divergences": np.nan,
        "mean_n_steps": np.nan,
        "sampling_seed": meta.get("sampling_seed"),
        "posterior_mib": np.nan,
        "posterior_summary": None,
        "exception_type": meta.get("exception_type", "SubprocessFailure"),
        "exception_message": meta.get(
            "exception_message",
            f"child process returned {returncode} without a benchmark failure record",
        ),
    }


def _posterior_agreement(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, float | int]:
    reference = pd.read_csv(reference_path, index_col="parameter")
    candidate = pd.read_csv(candidate_path, index_col="parameter")
    common = reference.index.intersection(candidate.index)
    if common.empty:
        raise ValueError("Posterior summaries share no parameter rows")

    ref = reference.loc[common]
    cand = candidate.loc[common]
    mean_diff = (cand["mean"] - ref["mean"]).abs()
    pooled_sd = np.sqrt((ref["sd"] ** 2 + cand["sd"] ** 2) / 2.0)
    standardized = mean_diff / pooled_sd.replace(0.0, np.nan)
    relative_sd = (cand["sd"] - ref["sd"]).abs() / ref["sd"].abs().replace(0.0, np.nan)

    return {
        "common_parameters": int(len(common)),
        "max_abs_mean_diff": float(mean_diff.max()),
        "median_abs_mean_diff": float(mean_diff.median()),
        "max_mean_diff_pooled_sd": float(standardized.max()),
        "median_mean_diff_pooled_sd": float(standardized.median()),
        "max_relative_sd_diff": float(relative_sd.max()),
        "median_relative_sd_diff": float(relative_sd.median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate NUTS backends in fresh processes with posterior agreement checks."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samplers", default="pymc,blackjax")
    parser.add_argument("--reference-sampler", default="pymc")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--predictors", type=int, default=3)
    parser.add_argument("--individuals", type=int, default=20)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--random-slopes", type=int, default=1)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--blas-cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=10_000)
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    samplers = _csv_strings(args.samplers)
    if args.reference_sampler not in samplers:
        parser.error("--reference-sampler must be included in --samplers")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    summary_paths: dict[tuple[int, str], Path] = {}

    from time import perf_counter

    for repeat in range(1, args.repeats + 1):
        paired_sampling_seed = args.sampling_seed + repeat
        for sampler in samplers:
            child_dir = output_dir / f"repeat-{repeat:02d}" / sampler
            child_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(BENCHMARK),
                "bayesian",
                "--output-dir",
                str(child_dir),
                "--rows",
                str(args.rows),
                "--predictors",
                str(args.predictors),
                "--individuals",
                str(args.individuals),
                "--bin-width",
                str(args.bin_width),
                "--random-slopes",
                str(args.random_slopes),
                "--samplers",
                sampler,
                "--draws",
                str(args.draws),
                "--tune",
                str(args.tune),
                "--chains",
                str(args.chains),
                "--cores",
                str(args.cores),
                "--target-accept",
                str(args.target_accept),
                "--eta-storage",
                "off",
                "--repeats",
                "1",
                "--seed",
                str(args.data_seed),
                "--sampling-seed",
                str(paired_sampling_seed - 1),
                "--save-posterior-summary",
            ]
            if args.blas_cores is not None:
                command.extend(["--blas-cores", str(args.blas_cores)])

            env = os.environ.copy()
            # Each child is a fresh interpreter. If a user has enabled a JAX
            # persistent compilation cache explicitly, isolate it by replicate
            # rather than allowing one sampler run to warm the next one.
            if "JAX_COMPILATION_CACHE_DIR" in env:
                env["JAX_COMPILATION_CACHE_DIR"] = str(
                    child_dir / "jax-compilation-cache"
                )

            started = perf_counter()
            completed = subprocess.run(
                command,
                cwd=HERE.parents[1],
                env=env,
                text=True,
                stdout=(child_dir / "stdout.log").open("w", encoding="utf-8"),
                stderr=(child_dir / "stderr.log").open("w", encoding="utf-8"),
                check=False,
            )
            process_seconds = perf_counter() - started
            records = _read_jsonl(child_dir / "bayesian_inference.jsonl")
            run_rows.append(
                _flatten_run_record(
                    sampler=sampler,
                    repeat=repeat,
                    process_seconds=process_seconds,
                    returncode=completed.returncode,
                    rows=records,
                )
            )
            summary_path = _summary_path(records, sampler)
            if summary_path is not None and summary_path.exists():
                summary_paths[(repeat, sampler)] = summary_path

    runs = pd.DataFrame(run_rows)
    runs.to_csv(output_dir / "sampler_validation_runs.csv", index=False)

    comparisons: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        reference_path = summary_paths.get((repeat, args.reference_sampler))
        reference_run = runs.loc[
            (runs["repeat"] == repeat)
            & (runs["sampler"] == args.reference_sampler)
            & (runs["status"] == "success")
        ]
        if reference_path is None or reference_run.empty:
            continue
        reference_seconds = float(reference_run.iloc[0]["sample_seconds"])
        reference_ess_s = float(reference_run.iloc[0]["median_ess_bulk_per_second"])

        for sampler in samplers:
            if sampler == args.reference_sampler:
                continue
            candidate_path = summary_paths.get((repeat, sampler))
            candidate_run = runs.loc[
                (runs["repeat"] == repeat)
                & (runs["sampler"] == sampler)
                & (runs["status"] == "success")
            ]
            if candidate_path is None or candidate_run.empty:
                continue
            candidate_seconds = float(candidate_run.iloc[0]["sample_seconds"])
            candidate_ess_s = float(candidate_run.iloc[0]["median_ess_bulk_per_second"])
            comparisons.append(
                {
                    "repeat": repeat,
                    "reference_sampler": args.reference_sampler,
                    "sampler": sampler,
                    "wall_speedup_vs_reference": reference_seconds / candidate_seconds,
                    "median_ess_s_ratio_vs_reference": candidate_ess_s / reference_ess_s,
                    **_posterior_agreement(reference_path, candidate_path),
                }
            )

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output_dir / "posterior_agreement.csv", index=False)

    print("\nFresh-process sampler runs")
    print(runs.to_string(index=False))
    if not comparison_df.empty:
        print("\nPosterior agreement against reference")
        print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
