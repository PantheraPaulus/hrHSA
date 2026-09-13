"""Fast end-to-end preflight before the production workstation/HPC campaign.

The smoke gate uses deliberately tiny workloads but exercises every execution path
that the long benchmark depends on: Zarr preparation, Dask persistence, frequentist
inference, completion-aware PyMC/BlackJAX sampling, and parallel frequentist and
Bayesian CV. It fails fast, prints the tail of the failing log, and writes SUCCESS
only after semantic output checks pass.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any

import pandas as pd

from hsa.compute import read_benchmark_records


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return "<log file was not created>"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _run(
    name: str,
    command: list[str],
    *,
    output_dir: Path,
    env: dict[str, str],
    report: list[dict[str, Any]],
) -> None:
    stage_dir = output_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / "run.log"
    print(f"\n=== smoke: {name} ===")
    print(" ".join(command))
    started = perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    seconds = perf_counter() - started
    report.append(
        {
            "stage": name,
            "returncode": completed.returncode,
            "wall_seconds": seconds,
            "command": command,
            "log": str(log_path),
        }
    )
    if completed.returncode != 0:
        print(f"\n[FAILED] {name} ({seconds:.1f} s)\n")
        print(_tail(log_path))
        raise RuntimeError(f"Smoke stage {name!r} failed; see {log_path}")
    print(f"[ok] {name}: {seconds:.1f} s")


def _preflight_dependencies() -> dict[str, str]:
    required = (
        "numpy",
        "pandas",
        "xarray",
        "zarr",
        "dask",
        "distributed",
        "statsmodels",
        "pymc",
        "arviz",
        "blackjax",
    )
    versions: dict[str, str] = {}
    for name in required:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _validate_outputs(output_dir: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    raster_path = output_dir / "01-raster" / "raster.jsonl"
    raster = read_benchmark_records(raster_path)
    if len(raster) != 4:
        raise RuntimeError(f"Expected 4 raster smoke records, found {len(raster)}")
    for column in (
        "operation_peak_process_tree_rss_mb",
        "worker_operation_peak_rss_total_mb",
        "worker_operation_peak_rss_max_mb",
    ):
        if column not in raster or raster[column].isna().any():
            raise RuntimeError(f"Raster smoke did not populate {column}")
    surface = raster.loc[raster["benchmark"] == "hpc_zarr_surface_prediction"]
    if surface.empty:
        raise RuntimeError("Raster smoke produced no surface records")
    if not (
        surface["metadata.materialization"]
        == "distributed_persist_no_final_assembly"
    ).all():
        raise RuntimeError("Raster smoke did not use distributed persistence")
    if not (surface["metadata.output_partitions"].astype(float) > 1).all():
        raise RuntimeError("Raster smoke surface collapsed to a single partition")
    checks["raster_records"] = int(len(raster))

    frequentist_summary = (
        output_dir
        / "02-frequentist"
        / "results"
        / "frequentist_inference_summary.csv"
    )
    if not frequentist_summary.exists():
        raise RuntimeError("Frequentist smoke summary was not created")
    frequentist = pd.read_csv(frequentist_summary)
    methods = set(
        frequentist.loc[
            frequentist["benchmark"] == "inference_frequentist_logit_fit",
            "variant",
        ].astype(str)
    )
    if not any(value.startswith("newton_") for value in methods) or not any(
        value.startswith("lbfgs_") for value in methods
    ):
        raise RuntimeError("Frequentist smoke did not run both Newton and L-BFGS")
    checks["frequentist_variants"] = sorted(methods)

    completion_path = (
        output_dir
        / "03-bayesian-completion"
        / "results"
        / "sampler_completion_runs.csv"
    )
    completion = pd.read_csv(completion_path)
    expected_samplers = {"pymc", "blackjax"}
    observed_samplers = set(completion.loc[completion["status"] == "success", "sampler"])
    if observed_samplers != expected_samplers:
        raise RuntimeError(
            f"Bayesian completion smoke succeeded for {observed_samplers}, "
            f"expected {expected_samplers}"
        )
    if completion["operation_peak_process_tree_rss_mb"].isna().any():
        raise RuntimeError("Bayesian completion smoke did not record peak process-tree RAM")
    checks["bayesian_samplers"] = sorted(observed_samplers)

    cv_files = {
        "frequentist": output_dir / "04-cv-frequentist" / "cv.jsonl",
        "pymc": output_dir / "05-cv-pymc" / "cv.jsonl",
        "blackjax": output_dir / "06-cv-blackjax" / "cv.jsonl",
    }
    for label, path in cv_files.items():
        frame = read_benchmark_records(path)
        if len(frame) != 1:
            raise RuntimeError(f"{label} CV smoke expected 1 record, found {len(frame)}")
        row = frame.iloc[0]
        if int(row["metadata.folds_failed"]) != 0:
            raise RuntimeError(f"{label} CV smoke had failed folds")
        if int(row["metadata.folds_completed"]) != 3:
            raise RuntimeError(f"{label} CV smoke did not complete all 3 folds")
        if pd.isna(row.get("worker_operation_peak_rss_total_mb")):
            raise RuntimeError(f"{label} CV smoke did not record operation-local worker RAM")
        checks[f"cv_{label}_wall_seconds"] = float(row["wall_seconds"])

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fast end-to-end smoke gate before the full hrHSA benchmark suite."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "production-smoke"
    )
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SUCCESS").unlink(missing_ok=True)

    versions = _preflight_dependencies()
    print("Dependency preflight OK")
    for name, version in versions.items():
        print(f"  {name}: {version}")

    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"

    report: list[dict[str, Any]] = []
    data_root = output_dir / "data"

    try:
        _run(
            "00-prepare",
            [
                sys.executable,
                str(HERE / "prepare_data.py"),
                "--root",
                str(data_root),
                "--size",
                "2048",
                "--bands",
                "3",
                "--storage-chunk",
                "512",
                "--points",
                "20000",
                "--workers",
                "2",
                "--force",
            ],
            output_dir=output_dir,
            env=env,
            report=report,
        )

        _run(
            "01-raster",
            [
                sys.executable,
                str(HERE / "run_scaling.py"),
                "--root",
                str(data_root),
                "--output",
                str(output_dir / "01-raster" / "raster.jsonl"),
                "--workers",
                "1,2",
                "--benchmarks",
                "sampling,surface",
                "--repeats",
                "1",
                "--spatial-chunk",
                "512",
                "--chunk-mb",
                "64",
                "--allow-low-task-density",
                "--backend",
                "local",
            ],
            output_dir=output_dir,
            env=env,
            report=report,
        )

        _run(
            "02-frequentist",
            [
                sys.executable,
                str(HERE / "benchmark_inference.py"),
                "frequentist",
                "--output-dir",
                str(output_dir / "02-frequentist" / "results"),
                "--rows",
                "50000",
                "--predictors",
                "3",
                "--individuals",
                "6",
                "--methods",
                "newton,lbfgs",
                "--blas-threads",
                "1",
                "--maxiter",
                "100",
                "--repeats",
                "1",
            ],
            output_dir=output_dir,
            env=env,
            report=report,
        )

        _run(
            "03-bayesian-completion",
            [
                sys.executable,
                str(HERE / "validate_bayesian_completion.py"),
                "--output-dir",
                str(output_dir / "03-bayesian-completion" / "results"),
                "--samplers",
                "pymc,blackjax",
                "--reference-sampler",
                "pymc",
                "--repeats",
                "1",
                "--rows",
                "5000",
                "--predictors",
                "2",
                "--individuals",
                "5",
                "--bin-width",
                "0.5",
                "--random-slopes",
                "1",
                "--draws",
                "50",
                "--tune",
                "50",
                "--chains",
                "2",
                "--cores",
                "1",
            ],
            output_dir=output_dir,
            env=env,
            report=report,
        )

        _run(
            "04-cv-frequentist",
            [
                sys.executable,
                str(HERE / "benchmark_cv_scaling.py"),
                "frequentist",
                "--output",
                str(output_dir / "04-cv-frequentist" / "cv.jsonl"),
                "--work-dir",
                str(output_dir / "cv-cache"),
                "--workers",
                "2",
                "--folds",
                "3",
                "--rows-per-individual",
                "200",
                "--predictors",
                "2",
                "--repeats",
                "1",
                "--method",
                "lbfgs",
                "--blas-threads",
                "1",
            ],
            output_dir=output_dir,
            env=env,
            report=report,
        )

        for stage, sampler in (("05-cv-pymc", "pymc"), ("06-cv-blackjax", "blackjax")):
            _run(
                stage,
                [
                    sys.executable,
                    str(HERE / "benchmark_cv_scaling.py"),
                    "bayesian",
                    "--output",
                    str(output_dir / stage / "cv.jsonl"),
                    "--work-dir",
                    str(output_dir / "cv-cache"),
                    "--workers",
                    "2",
                    "--folds",
                    "3",
                    "--rows-per-individual",
                    "150",
                    "--predictors",
                    "2",
                    "--repeats",
                    "1",
                    "--sampler",
                    sampler,
                    "--bin-width",
                    "0.5",
                    "--random-slopes",
                    "1",
                    "--draws",
                    "30",
                    "--tune",
                    "30",
                    "--chains",
                    "2",
                    "--cores",
                    "1",
                    "--target-accept",
                    "0.9",
                ],
                output_dir=output_dir,
                env=env,
                report=report,
            )

        checks = _validate_outputs(output_dir)
    except Exception as exc:
        (output_dir / "smoke_report.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "versions": versions,
                    "stages": report,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise

    summary = {
        "status": "success",
        "versions": versions,
        "checks": checks,
        "stages": report,
        "total_wall_seconds": float(sum(stage["wall_seconds"] for stage in report)),
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "SUCCESS").write_text("ok\n", encoding="utf-8")
    print("\nSMOKE GATE PASSED")
    print(f"Report: {output_dir / 'smoke_report.json'}")
    print(f"Total subprocess time: {summary['total_wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
