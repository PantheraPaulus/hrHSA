"""Run the complete pre-HPC hrHSA benchmark suite on a production workstation.

The suite is intended as the publication-quality single-machine reference before
moving to an HPC allocation. It validates every performance layer:

1. Zarr-backed raster strong scaling and process/thread geometry;
2. frequentist design-matrix/optimizer/BLAS behavior;
3. completion-aware PyMC versus BlackJAX timing and posterior agreement;
4. Bayesian inner-core sensitivity for each backend;
5. outer-fold CV scaling for frequentist, PyMC and BlackJAX execution.

Every stage runs in its own subprocess and writes a log plus a small suite manifest.
Existing prepared raster data can be reused with ``--reuse-raster-data``. Individual
stages can be skipped or resumed without changing the benchmark definitions. Run
``run_production_smoke.py`` first to catch obvious environment/execution failures.
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

import psutil


HERE = Path(__file__).resolve().parent


def _worker_counts(physical: int, *, bayesian: bool = False) -> list[int]:
    # Sample the saturation region more densely than a power-of-two sweep. On the
    # 12-core production host this yields 1,2,4,6,8,10,12.
    candidates = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32]
    if bayesian:
        candidates = [1, 2, 4, 6, 8, 10, 12]
    result = [value for value in candidates if value <= physical]
    if physical not in result:
        result.append(physical)
    return sorted(set(result))


def _run(
    *,
    name: str,
    command: list[str],
    output_dir: Path,
    env: dict[str, str],
    manifest: list[dict[str, Any]],
    resume: bool,
) -> None:
    stage_dir = output_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    marker = stage_dir / "SUCCESS"
    log_path = stage_dir / "run.log"

    if resume and marker.exists():
        print(f"[skip] {name}: SUCCESS marker exists")
        manifest.append(
            {
                "stage": name,
                "status": "skipped_existing_success",
                "command": command,
            }
        )
        return

    marker.unlink(missing_ok=True)
    print(f"\n=== {name} ===")
    print(" ".join(command))
    started = perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=HERE.parents[1],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    seconds = perf_counter() - started
    status = "success" if completed.returncode == 0 else "failed"
    manifest.append(
        {
            "stage": name,
            "status": status,
            "returncode": completed.returncode,
            "wall_seconds": seconds,
            "command": command,
            "log": str(log_path),
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage {name!r} failed with return code {completed.returncode}; "
            f"see {log_path}"
        )
    marker.write_text("ok\n", encoding="utf-8")
    print(f"[ok] {name}: {seconds:.1f} s")


def _completion_command(
    *,
    output_dir: Path,
    cores: int,
    repeats: int,
    rows: int,
    predictors: int,
    individuals: int,
    draws: int,
    tune: int,
) -> list[str]:
    return [
        sys.executable,
        str(HERE / "validate_bayesian_completion.py"),
        "--output-dir",
        str(output_dir),
        "--samplers",
        "pymc,blackjax",
        "--reference-sampler",
        "pymc",
        "--repeats",
        str(repeats),
        "--rows",
        str(rows),
        "--predictors",
        str(predictors),
        "--individuals",
        str(individuals),
        "--bin-width",
        "0.5",
        "--random-slopes",
        "1",
        "--draws",
        str(draws),
        "--tune",
        str(tune),
        "--chains",
        "4",
        "--cores",
        str(cores),
        "--target-accept",
        "0.9",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the publication-grade production-workstation benchmark before CoolMUC."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("quick", "standard", "stress"),
        default="standard",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frequentist-repeats", type=int, default=5)
    parser.add_argument("--geometry-threads", type=int, default=8)
    parser.add_argument("--reuse-raster-data", action="store_true")
    parser.add_argument("--include-smt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-raster", action="store_true")
    parser.add_argument("--skip-frequentist", action="store_true")
    parser.add_argument("--skip-bayesian", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--bayesian-repeats", type=int, default=3)
    parser.add_argument("--bayesian-draws", type=int, default=1000)
    parser.add_argument("--bayesian-tune", type=int, default=1000)
    parser.add_argument("--cv-bayesian-draws", type=int, default=250)
    parser.add_argument("--cv-bayesian-tune", type=int, default=250)
    parser.add_argument("--cv-bayesian-repeats", type=int, default=3)
    args = parser.parse_args()

    physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    logical = psutil.cpu_count(logical=True) or physical
    memory_gib = psutil.virtual_memory().total / 1024**3
    if args.geometry_threads <= 0 or args.geometry_threads > physical:
        parser.error(f"--geometry-threads must lie between 1 and {physical}")
    if min(
        args.repeats,
        args.frequentist_repeats,
        args.bayesian_repeats,
        args.cv_bayesian_repeats,
    ) <= 0:
        parser.error("repeat counts must be positive")

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"

    physical_worker_counts = _worker_counts(physical)
    bayesian_worker_counts = [
        value for value in _worker_counts(physical, bayesian=True) if value <= 12
    ]

    manifest: list[dict[str, Any]] = []
    header = {
        "suite": "publication-workstation-v1",
        "python": sys.version,
        "physical_cores": physical,
        "logical_cpus": logical,
        "memory_gib": memory_gib,
        "root": str(root),
        "output_dir": str(output_dir),
        "profile": args.profile,
        "geometry_threads": args.geometry_threads,
        "physical_worker_counts": physical_worker_counts,
        "bayesian_cv_worker_counts": bayesian_worker_counts,
        "repeats": args.repeats,
        "frequentist_repeats": args.frequentist_repeats,
        "bayesian_repeats": args.bayesian_repeats,
        "cv_bayesian_repeats": args.cv_bayesian_repeats,
        "include_smt": args.include_smt,
    }
    (output_dir / "machine.json").write_text(
        json.dumps(header, indent=2),
        encoding="utf-8",
    )

    _run(
        name="00-machine-probe",
        command=[sys.executable, str(HERE / "probe_workstation.py")],
        output_dir=output_dir,
        env=env,
        manifest=manifest,
        resume=args.resume,
    )

    if not args.skip_raster:
        raster_dir = output_dir / "01-raster"
        command = [
            sys.executable,
            str(HERE / "run_workstation.py"),
            "--root",
            str(root),
            "--output-dir",
            str(raster_dir),
            "--profile",
            args.profile,
            "--repeats",
            str(args.repeats),
            "--geometry-threads",
            str(args.geometry_threads),
            "--surface-compute-dtype",
            "float32",
        ]
        if args.reuse_raster_data:
            command.append("--skip-prepare")
        _run(
            name="01-raster",
            command=command,
            output_dir=output_dir,
            env=env,
            manifest=manifest,
            resume=args.resume,
        )

        if physical != args.geometry_threads:
            _run(
                name=f"01b-raster-geometry-{physical}t",
                command=[
                    sys.executable,
                    str(HERE / "run_workstation.py"),
                    "--root",
                    str(root),
                    "--output-dir",
                    str(output_dir / f"01b-raster-geometry-{physical}t" / "results"),
                    "--profile",
                    args.profile,
                    "--skip-prepare",
                    "--skip-strong-scaling",
                    "--geometry-threads",
                    str(physical),
                    "--repeats",
                    str(args.repeats),
                    "--surface-compute-dtype",
                    "float32",
                ],
                output_dir=output_dir,
                env=env,
                manifest=manifest,
                resume=args.resume,
            )

        if args.include_smt and logical > physical:
            _run(
                name="01c-raster-smt",
                command=[
                    sys.executable,
                    str(HERE / "run_workstation.py"),
                    "--root",
                    str(root),
                    "--output-dir",
                    str(output_dir / "01c-raster-smt" / "results"),
                    "--profile",
                    args.profile,
                    "--skip-prepare",
                    "--skip-strong-scaling",
                    "--skip-geometry",
                    "--include-smt",
                    "--repeats",
                    str(args.repeats),
                    "--surface-compute-dtype",
                    "float32",
                ],
                output_dir=output_dir,
                env=env,
                manifest=manifest,
                resume=args.resume,
            )

    if not args.skip_frequentist:
        _run(
            name="02-frequentist-inference",
            command=[
                sys.executable,
                str(HERE / "benchmark_inference.py"),
                "frequentist",
                "--output-dir",
                str(output_dir / "02-frequentist-inference" / "results"),
                "--rows",
                "2000000",
                "--predictors",
                "6",
                "--individuals",
                "20",
                "--methods",
                "newton,lbfgs",
                "--blas-threads",
                "1,2,4",
                "--maxiter",
                "200",
                "--repeats",
                str(args.frequentist_repeats),
            ],
            output_dir=output_dir,
            env=env,
            manifest=manifest,
            resume=args.resume,
        )

    if not args.skip_bayesian:
        _run(
            name="03-bayesian-completion-reference",
            command=_completion_command(
                output_dir=output_dir
                / "03-bayesian-completion-reference"
                / "results",
                cores=min(4, physical),
                repeats=args.bayesian_repeats,
                rows=100_000,
                predictors=3,
                individuals=20,
                draws=args.bayesian_draws,
                tune=args.bayesian_tune,
            ),
            output_dir=output_dir,
            env=env,
            manifest=manifest,
            resume=args.resume,
        )

        # These are sensitivity points, not the primary sampler comparison; one
        # paired repeat is enough to determine whether extra inner cores merit a
        # dedicated follow-up campaign.
        for cores in [value for value in (1, 2) if value <= physical]:
            _run(
                name=f"04-bayesian-inner-{cores}c",
                command=_completion_command(
                    output_dir=output_dir / f"04-bayesian-inner-{cores}c" / "results",
                    cores=cores,
                    repeats=1,
                    rows=100_000,
                    predictors=3,
                    individuals=20,
                    draws=args.bayesian_draws,
                    tune=args.bayesian_tune,
                ),
                output_dir=output_dir,
                env=env,
                manifest=manifest,
                resume=args.resume,
            )

    if not args.skip_cv:
        cv_work = output_dir / "cv-cache"
        frequentist_output = output_dir / "05-cv-frequentist" / "cv.jsonl"
        for workers in physical_worker_counts:
            _run(
                name=f"05-cv-frequentist-{workers:02d}w",
                command=[
                    sys.executable,
                    str(HERE / "benchmark_cv_scaling.py"),
                    "frequentist",
                    "--output",
                    str(frequentist_output),
                    "--work-dir",
                    str(cv_work),
                    "--workers",
                    str(workers),
                    "--folds",
                    "24",
                    "--rows-per-individual",
                    "5000",
                    "--predictors",
                    "6",
                    "--repeats",
                    str(args.repeats),
                    "--method",
                    "lbfgs",
                    "--maxiter",
                    "200",
                    "--blas-threads",
                    "1",
                ],
                output_dir=output_dir,
                env=env,
                manifest=manifest,
                resume=args.resume,
            )

        _run(
            name="05-cv-frequentist-summary",
            command=[
                sys.executable,
                str(HERE / "summarize_cv_scaling.py"),
                str(frequentist_output),
            ],
            output_dir=output_dir,
            env=env,
            manifest=manifest,
            resume=args.resume,
        )

        for sampler in ("pymc", "blackjax"):
            sampler_output = output_dir / f"06-cv-{sampler}" / "cv.jsonl"
            for workers in bayesian_worker_counts:
                if workers > 12:
                    continue
                _run(
                    name=f"06-cv-{sampler}-{workers:02d}w",
                    command=[
                        sys.executable,
                        str(HERE / "benchmark_cv_scaling.py"),
                        "bayesian",
                        "--output",
                        str(sampler_output),
                        "--work-dir",
                        str(cv_work),
                        "--workers",
                        str(workers),
                        "--folds",
                        "12",
                        "--rows-per-individual",
                        "500",
                        "--predictors",
                        "3",
                        "--repeats",
                        str(args.cv_bayesian_repeats),
                        "--sampler",
                        sampler,
                        "--bin-width",
                        "0.5",
                        "--random-slopes",
                        "1",
                        "--draws",
                        str(args.cv_bayesian_draws),
                        "--tune",
                        str(args.cv_bayesian_tune),
                        "--chains",
                        "4",
                        "--cores",
                        "1",
                        "--target-accept",
                        "0.9",
                    ],
                    output_dir=output_dir,
                    env=env,
                    manifest=manifest,
                    resume=args.resume,
                )

            _run(
                name=f"06-cv-{sampler}-summary",
                command=[
                    sys.executable,
                    str(HERE / "summarize_cv_scaling.py"),
                    str(sampler_output),
                ],
                output_dir=output_dir,
                env=env,
                manifest=manifest,
                resume=args.resume,
            )

    (output_dir / "suite_manifest.json").write_text(
        json.dumps({"machine": header, "stages": manifest}, indent=2),
        encoding="utf-8",
    )
    print(f"\nComplete production benchmark suite written to {output_dir}")


if __name__ == "__main__":
    main()
