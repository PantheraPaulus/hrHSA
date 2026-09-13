"""Run the workstation benchmark used for hrHSA documentation.

The documentation campaign is intentionally narrower than the complete production
validation suite. It measures the operations a prospective user most often needs
to budget for:

* raster point sampling;
* full-raster RSF surface prediction;
* hierarchical Bayesian NUTS/MCMC.

The canonical ``reference`` raster is deliberately large enough that steady-state
parallel work dominates workstation startup/scheduler noise: 49,152 x 49,152 cells,
six float32 covariates, and ten million sampled points. Smaller profiles remain
available for development and smoke-style runs.

Raster execution includes physical-core strong scaling plus a fixed-physical-core
worker/thread geometry sweep. MCMC is run in fresh processes and timed through
explicit posterior materialisation. The final stage turns raw measurements into
documentation-ready CSV, Markdown, PNG and SVG artifacts.
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

from hsa.compute import discover_runtime_topology, software_versions


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

# Documentation profiles are distinct from run_workstation.py's preparation
# templates. The documentation runner prepares/validates the dataset itself and
# then calls run_workstation.py with --skip-prepare, so the measured kernels always
# see the exact workload recorded here.
RASTER_PROFILES = {
    "quick": {"size": 12_288, "bands": 6, "points": 1_000_000},
    "standard": {"size": 24_576, "bands": 6, "points": 2_000_000},
    "reference": {"size": 49_152, "bands": 6, "points": 10_000_000},
}


def _git(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *command],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _run_stage(
    *,
    name: str,
    command: list[str],
    suite_dir: Path,
    env: dict[str, str],
    manifest: list[dict[str, Any]],
    resume: bool,
) -> None:
    stage_dir = suite_dir / name
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
                "log": str(log_path),
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
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    seconds = perf_counter() - started
    entry = {
        "stage": name,
        "status": "success" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "wall_seconds": seconds,
        "command": command,
        "log": str(log_path),
    }
    manifest.append(entry)
    if completed.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:]
        )
        raise RuntimeError(
            f"Stage {name!r} failed with return code {completed.returncode}.\n"
            f"Last log lines:\n{tail}"
        )
    marker.write_text("ok\n", encoding="utf-8")
    print(f"[ok] {name}: {seconds:.1f} s")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _expected_raster_manifest(
    profile: dict[str, int], *, storage_chunk: int
) -> dict[str, int]:
    size = int(profile["size"])
    return {
        "size": size,
        "bands": int(profile["bands"]),
        "storage_chunk": int(storage_chunk),
        "points": int(profile["points"]),
        "spatial_chunks": (size // storage_chunk) ** 2
        if size % storage_chunk == 0
        else ((size + storage_chunk - 1) // storage_chunk) ** 2,
    }


def _manifest_matches(actual: dict[str, Any], expected: dict[str, int]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _prepare_raster_command(
    *,
    root: Path,
    profile: dict[str, int],
    storage_chunk: int,
    physical_cores: int,
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "prepare_data.py"),
        "--root",
        str(root / "data"),
        "--size",
        str(profile["size"]),
        "--bands",
        str(profile["bands"]),
        "--storage-chunk",
        str(storage_chunk),
        "--points",
        str(profile["points"]),
        "--workers",
        str(min(physical_cores, 8)),
    ]
    if force:
        command.append("--force")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the documentation-facing hrHSA workstation benchmark."
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Fast local benchmark storage"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(RASTER_PROFILES),
        default="reference",
        help="Raster workload; 'reference' is the documentation-quality default.",
    )
    parser.add_argument("--raster-repeats", type=int, default=3)
    parser.add_argument("--storage-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument(
        "--reuse-raster-data",
        action="store_true",
        help="Require and reuse a previously prepared dataset matching the profile.",
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--raster-only",
        action="store_true",
        help="Stop after the raster campaign; rerun later with --resume for MCMC/report.",
    )

    parser.add_argument("--mcmc-samplers", default="pymc,blackjax")
    parser.add_argument("--mcmc-repeats", type=int, default=3)
    parser.add_argument("--mcmc-rows", type=int, default=100_000)
    parser.add_argument("--mcmc-predictors", type=int, default=3)
    parser.add_argument("--mcmc-individuals", type=int, default=20)
    parser.add_argument("--mcmc-random-slopes", type=int, default=1)
    parser.add_argument("--mcmc-draws", type=int, default=1000)
    parser.add_argument("--mcmc-tune", type=int, default=1000)
    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument("--mcmc-cores", type=int, default=None)
    parser.add_argument("--mcmc-target-accept", type=float, default=0.9)
    args = parser.parse_args()

    if args.raster_repeats <= 0 or args.mcmc_repeats <= 0:
        parser.error("repeat counts must be positive")
    if args.storage_chunk <= 0 or args.chunk_mb <= 0:
        parser.error("chunk sizes must be positive")
    if args.reuse_raster_data and args.force_prepare:
        parser.error("--reuse-raster-data and --force-prepare are mutually exclusive")
    if min(
        args.mcmc_rows,
        args.mcmc_predictors,
        args.mcmc_individuals,
        args.mcmc_draws,
        args.mcmc_tune,
        args.mcmc_chains,
    ) <= 0:
        parser.error("MCMC workload sizes must be positive")

    samplers = [item.strip() for item in args.mcmc_samplers.split(",") if item.strip()]
    if "pymc" not in samplers:
        parser.error("--mcmc-samplers must include pymc as the posterior reference")

    physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    logical = psutil.cpu_count(logical=True) or physical
    memory_gib = psutil.virtual_memory().total / 1024**3
    mcmc_cores = (
        min(args.mcmc_chains, physical)
        if args.mcmc_cores is None
        else args.mcmc_cores
    )
    if mcmc_cores <= 0 or mcmc_cores > physical:
        parser.error(f"--mcmc-cores must lie between 1 and {physical}")

    root = args.root.expanduser().resolve()
    suite_dir = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    suite_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"

    profile = RASTER_PROFILES[args.profile]
    expected_manifest = _expected_raster_manifest(
        profile, storage_chunk=args.storage_chunk
    )
    raster_bytes = profile["size"] ** 2 * profile["bands"] * 4

    topology = discover_runtime_topology().as_dict()
    git_status = _git(["status", "--porcelain"])
    machine = {
        "suite": "documentation-workstation-v2",
        "profile": args.profile,
        "physical_cores": physical,
        "logical_cpus": logical,
        "memory_gib": memory_gib,
        "topology": topology,
        "software_versions": software_versions(
            packages=(
                "hsa",
                "numpy",
                "pandas",
                "xarray",
                "dask",
                "distributed",
                "zarr",
                "scipy",
                "scikit-learn",
                "statsmodels",
                "pymc",
                "blackjax",
                "arviz",
                "psutil",
            )
        ),
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(git_status),
            "status_porcelain": git_status or "",
        },
        "raster": {
            **expected_manifest,
            "uncompressed_bytes": raster_bytes,
            "uncompressed_gib": raster_bytes / 1024**3,
            "chunk_mb": args.chunk_mb,
            "repeats": args.raster_repeats,
            "compute_dtype": "float32",
            "geometry_threads": physical,
        },
        "mcmc": {
            "samplers": samplers,
            "repeats": args.mcmc_repeats,
            "rows": args.mcmc_rows,
            "predictors": args.mcmc_predictors,
            "individuals": args.mcmc_individuals,
            "random_slopes": args.mcmc_random_slopes,
            "draws": args.mcmc_draws,
            "tune": args.mcmc_tune,
            "chains": args.mcmc_chains,
            "cores": mcmc_cores,
            "target_accept": args.mcmc_target_accept,
            "bin_width": 0.5,
        },
    }
    (suite_dir / "machine.json").write_text(
        json.dumps(machine, indent=2, default=str), encoding="utf-8"
    )

    manifest: list[dict[str, Any]] = []
    manifest_path = suite_dir / "suite_manifest.json"

    try:
        _run_stage(
            name="00-machine-probe",
            command=[sys.executable, str(HERE / "probe_workstation.py")],
            suite_dir=suite_dir,
            env=env,
            manifest=manifest,
            resume=args.resume,
        )

        data_manifest_path = root / "data" / "manifest.json"
        actual_manifest = _load_json(data_manifest_path)
        data_paths_exist = (root / "data" / "environment.zarr").exists() and (
            root / "data" / "points.parquet"
        ).exists()
        matches = data_paths_exist and _manifest_matches(
            actual_manifest, expected_manifest
        )

        if args.reuse_raster_data:
            if not matches:
                raise RuntimeError(
                    "--reuse-raster-data was requested, but the prepared dataset does "
                    f"not match profile {args.profile!r}. Expected {expected_manifest}; "
                    f"found {actual_manifest or '<no manifest>'}."
                )
        else:
            rebuild = args.force_prepare or (data_paths_exist and not matches)
            if data_paths_exist and not matches and not args.force_prepare:
                print(
                    "Prepared raster does not match the requested profile; rebuilding "
                    "the benchmark data root with --force."
                )
            _run_stage(
                name="01-raster-prepare",
                command=_prepare_raster_command(
                    root=root,
                    profile=profile,
                    storage_chunk=args.storage_chunk,
                    physical_cores=physical,
                    force=rebuild,
                ),
                suite_dir=suite_dir,
                env=env,
                manifest=manifest,
                resume=args.resume and matches,
            )

        # run_workstation's profile is only a preparation template when
        # --skip-prepare is absent. We prepare and validate above, then exercise the
        # production kernels against those exact files.
        raster_command = [
            sys.executable,
            str(HERE / "run_workstation.py"),
            "--root",
            str(root),
            "--output-dir",
            str(suite_dir / "01-raster"),
            "--profile",
            "standard",
            "--skip-prepare",
            "--repeats",
            str(args.raster_repeats),
            "--storage-chunk",
            str(args.storage_chunk),
            "--chunk-mb",
            str(args.chunk_mb),
            "--geometry-threads",
            str(physical),
            "--surface-compute-dtype",
            "float32",
        ]
        _run_stage(
            name="01-raster",
            command=raster_command,
            suite_dir=suite_dir,
            env=env,
            manifest=manifest,
            resume=args.resume,
        )

        if args.raster_only:
            print("\nRaster-only benchmark complete")
            print(f"  raster results: {suite_dir / '01-raster'}")
            print("  rerun without --raster-only and with --resume to continue")
            return

        _run_stage(
            name="02-mcmc",
            command=[
                sys.executable,
                str(HERE / "validate_bayesian_completion.py"),
                "--output-dir",
                str(suite_dir / "02-mcmc"),
                "--samplers",
                args.mcmc_samplers,
                "--reference-sampler",
                "pymc",
                "--repeats",
                str(args.mcmc_repeats),
                "--rows",
                str(args.mcmc_rows),
                "--predictors",
                str(args.mcmc_predictors),
                "--individuals",
                str(args.mcmc_individuals),
                "--bin-width",
                "0.5",
                "--random-slopes",
                str(args.mcmc_random_slopes),
                "--draws",
                str(args.mcmc_draws),
                "--tune",
                str(args.mcmc_tune),
                "--chains",
                str(args.mcmc_chains),
                "--cores",
                str(mcmc_cores),
                "--target-accept",
                str(args.mcmc_target_accept),
            ],
            suite_dir=suite_dir,
            env=env,
            manifest=manifest,
            resume=args.resume,
        )

        _run_stage(
            name="03-report",
            command=[
                sys.executable,
                str(HERE / "build_documentation_benchmark_report.py"),
                "--suite-dir",
                str(suite_dir),
                "--output-dir",
                str(suite_dir / "report"),
            ],
            suite_dir=suite_dir,
            env=env,
            manifest=manifest,
            resume=args.resume,
        )
    finally:
        manifest_path.write_text(
            json.dumps(
                {"suite": machine["suite"], "git": machine["git"], "stages": manifest},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    print("\nDocumentation benchmark complete")
    print(f"  suite:  {suite_dir}")
    print(f"  report: {suite_dir / 'report' / 'documentation_benchmark.md'}")
    print(f"  table:  {suite_dir / 'report' / 'reference_operations.csv'}")
    print(f"  plots:  {suite_dir / 'report' / 'figures'}")


if __name__ == "__main__":
    main()
