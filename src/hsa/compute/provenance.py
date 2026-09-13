"""Reproducible run manifests for local and HPC execution."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from hsa.compute.benchmark import current_git_commit
from hsa.compute.profiles import (
    COOLMUC4_HARDWARE,
    COOLMUC4_POLICY,
    benchmark_profile_for,
)
from hsa.compute.topology import (
    collect_worker_topology,
    discover_runtime_topology,
    validate_worker_topology,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


def software_versions(
    packages: tuple[str, ...] = (
        "hsa",
        "numpy",
        "pandas",
        "xarray",
        "dask",
        "distributed",
        "zarr",
        "scipy",
        "scikit-learn",
        "pymc",
    ),
) -> dict[str, str | None]:
    """Collect versions without importing heavy optional dependencies."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def selected_environment() -> dict[str, str]:
    """Return reproducibility-relevant non-secret runtime environment variables."""
    names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TMPDIR",
        "SCRATCH_DSS",
        "HRHSA_GIT_COMMIT",
        "HRHSA_DASK_INTERFACE",
        "HRHSA_DATA_ROOT",
        "HRHSA_LARGE_DATA_ROOT",
        "HRHSA_RESULT_ROOT",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NUM_NODES",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


@dataclass
class RunManifest:
    """Machine-readable provenance for one hrHSA execution."""

    run_id: str
    status: str
    created_utc: str
    finished_utc: str | None
    command: list[str]
    git_commit: str | None
    git_dirty: bool | None
    python: str
    platform: str
    software: dict[str, str | None]
    environment: dict[str, str]
    execution_plan: dict[str, Any]
    site_policy: dict[str, Any] | None
    hardware_profile: dict[str, Any] | None
    benchmark_profile: dict[str, Any] | None
    slurm_allocation: dict[str, Any] | None
    driver_topology: dict[str, Any]
    worker_topology: dict[str, dict[str, Any]] = field(default_factory=dict)
    topology_validation: dict[str, Any] | None = None
    dataset: dict[str, Any] = field(default_factory=dict)
    slurm_accounting: list[dict[str, str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _allocation_dict() -> dict[str, Any] | None:
    from hsa.compute.slurm import current_slurm_allocation

    allocation = current_slurm_allocation()
    return None if allocation is None else asdict(allocation)


def build_run_manifest(
    plan,
    *,
    client=None,
    dataset: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    strict_affinity: bool = False,
) -> RunManifest:
    """Build a manifest from the requested plan and observed runtime state."""
    profile = benchmark_profile_for(
        site=plan.site,
        workload=plan.workload,
        cores_per_node=plan.cores_per_node,
    )
    workers = collect_worker_topology(client) if client is not None else {}
    validation = None
    if client is not None:
        validation = validate_worker_topology(
            client,
            plan,
            strict_affinity=strict_affinity,
        ).as_dict()

    site_policy = COOLMUC4_POLICY.as_dict() if plan.site == "coolmuc4" else None
    hardware = COOLMUC4_HARDWARE.as_dict() if plan.site == "coolmuc4" else None

    return RunManifest(
        run_id=str(uuid4()),
        status="running",
        created_utc=_utc_now(),
        finished_utc=None,
        command=[str(value) for value in sys.argv],
        git_commit=current_git_commit(),
        git_dirty=_git_dirty(),
        python=platform.python_version(),
        platform=platform.platform(),
        software=software_versions(),
        environment=selected_environment(),
        execution_plan=plan.as_dict(),
        site_policy=site_policy,
        hardware_profile=hardware,
        benchmark_profile=None if profile is None else profile.as_dict(),
        slurm_allocation=_allocation_dict(),
        driver_topology=discover_runtime_topology().as_dict(),
        worker_topology=workers,
        topology_validation=validation,
        dataset=dict(dataset or {}),
        extra=dict(extra or {}),
    )


def write_run_manifest(manifest: RunManifest | dict[str, Any], path: str | Path) -> Path:
    """Atomically write one JSON manifest and return its final path."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict() if isinstance(manifest, RunManifest) else dict(manifest)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def collect_slurm_accounting(
    job_id: str | int,
    *,
    cluster: str | None = None,
) -> list[dict[str, str]]:
    """Take one low-frequency ``sacct`` snapshot for a job.

    This function performs a single scheduler query. It is intended for explicit
    post-run collection, not a polling loop.
    """
    sacct = shutil.which("sacct")
    if sacct is None:
        return []
    fields = (
        "JobID",
        "NNodes",
        "NTasks",
        "Start",
        "Elapsed",
        "TotalCPU",
        "CPUTimeRAW",
        "MaxRSS",
        "State",
        "Reason",
        "ExitCode",
        "NodeList",
    )
    command = [sacct]
    if cluster:
        command.extend(["-M", str(cluster)])
    command.extend(
        [
            "-n",
            "-P",
            "-j",
            str(job_id),
            "--format=" + ",".join(fields),
        ]
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    records: list[dict[str, str]] = []
    for raw in result.stdout.splitlines():
        values = raw.rstrip("|").split("|")
        if len(values) != len(fields):
            continue
        records.append(dict(zip(fields, values)))
    return records


def finalize_run_manifest(
    manifest: RunManifest,
    path: str | Path,
    *,
    status: str,
    collect_accounting: bool = False,
) -> Path:
    """Mark a run complete/failed and atomically persist the final manifest."""
    manifest.status = str(status)
    manifest.finished_utc = _utc_now()
    if collect_accounting and manifest.slurm_allocation:
        manifest.slurm_accounting = collect_slurm_accounting(
            manifest.slurm_allocation["job_id"],
            cluster=manifest.slurm_allocation.get("cluster"),
        )
    return write_run_manifest(manifest, path)
