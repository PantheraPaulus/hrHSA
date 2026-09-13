"""Fresh-process sampler validation using completed, synchronized inference time.

Each child run explicitly materializes posterior/sample_stats to NumPy before the
sampling timer is considered complete. Posterior agreement is checked against a
reference sampler, and operation-local process-tree memory is retained for
publication-quality resource comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from validate_bayesian_samplers import _csv_strings, _posterior_agreement


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "benchmark_bayesian_completion.py"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return "<not created>"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _report_child_failure(
    *,
    sampler: str,
    repeat: int,
    returncode: int,
    result: dict[str, Any],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    print(
        f"\n[FAILED] sampler={sampler!r}, repeat={repeat}, "
        f"returncode={returncode}, result_status={result.get('status')!r}"
    )
    print("--- child stdout (tail) ---")
    print(_tail(stdout_path))
    print("--- child stderr (tail) ---")
    print(_tail(stderr_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Bayesian sampler backends with explicit completion synchronization."
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
    parser.add_argument("--memory-sample-interval", type=float, default=0.25)
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    samplers = _csv_strings(args.samplers)
    if args.reference_sampler not in samplers:
        parser.error("--reference-sampler must be included in --samplers")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary_paths: dict[tuple[int, str], Path] = {}

    for repeat in range(1, args.repeats + 1):
        sampling_seed = args.sampling_seed + repeat
        for sampler in samplers:
            child_dir = output_dir / f"repeat-{repeat:02d}" / sampler
            child_dir.mkdir(parents=True, exist_ok=True)
            result_path = child_dir / "completion.json"
            summary_path = child_dir / "posterior_summary.csv"
            stdout_path = child_dir / "stdout.log"
            stderr_path = child_dir / "stderr.log"
            command = [
                sys.executable,
                str(BENCHMARK),
                "--sampler",
                sampler,
                "--output",
                str(result_path),
                "--posterior-summary",
                str(summary_path),
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
                "--data-seed",
                str(args.data_seed),
                "--sampling-seed",
                str(sampling_seed),
                "--memory-sample-interval",
                str(args.memory_sample_interval),
            ]
            if args.blas_cores is not None:
                command.extend(["--blas-cores", str(args.blas_cores)])

            env = os.environ.copy()
            if "JAX_COMPILATION_CACHE_DIR" in env:
                env["JAX_COMPILATION_CACHE_DIR"] = str(
                    child_dir / "jax-compilation-cache"
                )

            started = perf_counter()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(
                    command,
                    cwd=HERE.parents[1],
                    env=env,
                    text=True,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            process_seconds = perf_counter() - started
            result = _read_json(result_path)

            succeeded = completed.returncode == 0 and result.get("status") == "success"
            if succeeded:
                rows.append(
                    {
                        "sampler": sampler,
                        "repeat": repeat,
                        "status": "success",
                        "process_seconds": process_seconds,
                        "dispatch_seconds": result.get("dispatch_seconds"),
                        "materialize_seconds": result.get("materialize_seconds"),
                        "completed_sampling_seconds": result.get(
                            "completed_sampling_seconds"
                        ),
                        "diagnostics_seconds": result.get("diagnostics_seconds"),
                        "median_ess_bulk": result.get("median_ess_bulk"),
                        "min_ess_bulk": result.get("min_ess_bulk"),
                        "median_ess_bulk_per_completed_second": result.get(
                            "median_ess_bulk_per_completed_second"
                        ),
                        "min_ess_bulk_per_completed_second": result.get(
                            "min_ess_bulk_per_completed_second"
                        ),
                        "min_ess_tail": result.get("min_ess_tail"),
                        "max_rhat": result.get("max_rhat"),
                        "n_divergences": result.get("n_divergences"),
                        "mean_n_steps": result.get("mean_n_steps"),
                        "operation_peak_rss_mb": result.get("operation_peak_rss_mb"),
                        "operation_peak_process_tree_rss_mb": result.get(
                            "operation_peak_process_tree_rss_mb"
                        ),
                        "memory_samples": result.get("memory_samples"),
                        "sampling_seed": result.get("sampling_seed"),
                        "posterior_mib": result.get("posterior_mib"),
                        "posterior_summary": result.get("posterior_summary"),
                        "returncode": completed.returncode,
                    }
                )
                if summary_path.exists():
                    summary_paths[(repeat, sampler)] = summary_path
            else:
                _report_child_failure(
                    sampler=sampler,
                    repeat=repeat,
                    returncode=completed.returncode,
                    result=result,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                rows.append(
                    {
                        "sampler": sampler,
                        "repeat": repeat,
                        "status": "failed",
                        "process_seconds": process_seconds,
                        "dispatch_seconds": np.nan,
                        "materialize_seconds": np.nan,
                        "completed_sampling_seconds": np.nan,
                        "diagnostics_seconds": np.nan,
                        "median_ess_bulk": np.nan,
                        "min_ess_bulk": np.nan,
                        "median_ess_bulk_per_completed_second": np.nan,
                        "min_ess_bulk_per_completed_second": np.nan,
                        "min_ess_tail": np.nan,
                        "max_rhat": np.nan,
                        "n_divergences": np.nan,
                        "mean_n_steps": np.nan,
                        "operation_peak_rss_mb": np.nan,
                        "operation_peak_process_tree_rss_mb": np.nan,
                        "memory_samples": np.nan,
                        "sampling_seed": sampling_seed,
                        "posterior_mib": np.nan,
                        "posterior_summary": None,
                        "returncode": completed.returncode,
                    }
                )

    runs = pd.DataFrame(rows)
    runs.to_csv(output_dir / "sampler_completion_runs.csv", index=False)

    comparisons: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        reference = runs.loc[
            (runs["repeat"] == repeat)
            & (runs["sampler"] == args.reference_sampler)
            & (runs["status"] == "success")
        ]
        reference_path = summary_paths.get((repeat, args.reference_sampler))
        if reference.empty or reference_path is None:
            continue
        reference_row = reference.iloc[0]
        reference_seconds = float(reference_row["completed_sampling_seconds"])
        reference_ess_s = float(
            reference_row["median_ess_bulk_per_completed_second"]
        )

        for sampler in samplers:
            if sampler == args.reference_sampler:
                continue
            candidate = runs.loc[
                (runs["repeat"] == repeat)
                & (runs["sampler"] == sampler)
                & (runs["status"] == "success")
            ]
            candidate_path = summary_paths.get((repeat, sampler))
            if candidate.empty or candidate_path is None:
                continue
            candidate_row = candidate.iloc[0]
            candidate_seconds = float(candidate_row["completed_sampling_seconds"])
            candidate_ess_s = float(
                candidate_row["median_ess_bulk_per_completed_second"]
            )
            comparisons.append(
                {
                    "repeat": repeat,
                    "reference_sampler": args.reference_sampler,
                    "sampler": sampler,
                    "completed_speedup_vs_reference": reference_seconds
                    / candidate_seconds,
                    "completed_ess_s_ratio_vs_reference": candidate_ess_s
                    / reference_ess_s,
                    "reference_completed_seconds": reference_seconds,
                    "candidate_completed_seconds": candidate_seconds,
                    **_posterior_agreement(reference_path, candidate_path),
                }
            )

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output_dir / "posterior_completion_agreement.csv", index=False)

    print("\nCompleted fresh-process sampler runs")
    print(runs.to_string(index=False))
    if not comparison_df.empty:
        print("\nPosterior agreement and completed-time comparison")
        print(comparison_df.to_string(index=False))

    failed = runs.loc[runs["status"] != "success", ["sampler", "repeat", "returncode"]]
    if not failed.empty:
        raise RuntimeError(
            "Bayesian completion validation had failed child run(s): "
            + failed.to_dict(orient="records").__repr__()
        )


if __name__ == "__main__":
    main()
