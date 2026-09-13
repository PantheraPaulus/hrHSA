"""Slurm submission and allocation-aware Dask helpers.

Notebook submission remains a thin wrapper around version-controlled batch files.
Inside an allocation, Dask workers are launched as explicit Slurm job steps and
validated against the requested topology.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from hsa.compute.profiles import COOLMUC4_POLICY
from hsa.compute.sites import ExecutionPlan
from hsa.compute.topology import validate_worker_topology


@dataclass(frozen=True)
class SlurmJob:
    """A job submitted through :func:`submit_slurm_job`."""

    job_id: str
    script: Path
    submission_output: str
    cluster: str | None = None

    def status(self) -> str:
        return slurm_job_status(self.job_id, cluster=self.cluster)

    def wait(
        self,
        *,
        poll_seconds: float = COOLMUC4_POLICY.scheduler_poll_seconds,
        timeout: float | None = None,
    ) -> str:
        return wait_for_slurm_job(
            self.job_id,
            cluster=self.cluster,
            poll_seconds=poll_seconds,
            timeout=timeout,
        )

    def cancel(self) -> None:
        cancel_slurm_job(self.job_id, cluster=self.cluster)

    def accounting(self) -> list[dict[str, str]]:
        """Take one post-run ``sacct`` snapshot for this job."""
        from hsa.compute.provenance import collect_slurm_accounting

        return collect_slurm_accounting(self.job_id, cluster=self.cluster)


@dataclass(frozen=True)
class SlurmAllocation:
    """Relevant resource facts exposed by the current Slurm allocation."""

    job_id: str
    cluster: str | None
    partition: str | None
    nodes: int
    cpus_per_node: int | None
    memory_per_node_gib: float | None
    node_list: str | None = None
    tasks_per_node: int | None = None
    cpus_per_task: int | None = None


def _require_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"{name!r} is not available on PATH; this operation must run on a "
            "host with the Slurm client commands installed."
        )
    return path


def slurm_available() -> bool:
    """Return whether basic Slurm submission/status commands are available."""
    return shutil.which("sbatch") is not None and shutil.which("squeue") is not None


def _parse_submission(text: str) -> tuple[str, str | None]:
    """Parse ``sbatch --parsable`` output as ``(job_id, cluster)``."""
    value = text.strip().splitlines()[0] if text.strip() else ""
    job_text, separator, cluster_text = value.partition(";")
    job_id = job_text.strip()
    if not job_id or not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
        raise RuntimeError(f"Could not parse Slurm job id from: {text!r}")
    cluster = cluster_text.strip() if separator and cluster_text.strip() else None
    return job_id, cluster


def _cluster_args(cluster: str | None) -> list[str]:
    return [] if not cluster else ["-M", cluster]


def submit_slurm_job(
    script: str | Path,
    *,
    args: Iterable[str | int | float] = (),
    chdir: str | Path | None = None,
) -> SlurmJob:
    """Submit a version-controlled batch script and return its Slurm job handle.

    Resource flags are deliberately not accepted here. CPU, memory, partition and
    walltime stay visible in reviewed ``.sbatch`` files.
    """
    sbatch = _require_command("sbatch")
    script_path = Path(script).expanduser()
    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    command = [sbatch, "--parsable", str(script_path), *[str(x) for x in args]]
    result = subprocess.run(
        command,
        cwd=None if chdir is None else str(Path(chdir).expanduser()),
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    job_id, cluster = _parse_submission(output)
    return SlurmJob(
        job_id=job_id,
        script=script_path,
        submission_output=output,
        cluster=cluster,
    )


def slurm_job_status(
    job_id: str | int,
    *,
    cluster: str | None = None,
) -> str:
    """Return one current Slurm state snapshot."""
    job_id = str(job_id)
    squeue = _require_command("squeue")
    queued = subprocess.run(
        [squeue, *_cluster_args(cluster), "-h", "-j", job_id, "-o", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    state = queued.stdout.strip().splitlines()
    if state:
        return state[0].strip()

    sacct = shutil.which("sacct")
    if sacct is None:
        return "UNKNOWN"
    completed = subprocess.run(
        [
            sacct,
            *_cluster_args(cluster),
            "-n",
            "-X",
            "-j",
            job_id,
            "--format=State",
            "-P",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    states = [line.strip().split("|", 1)[0] for line in completed.stdout.splitlines()]
    states = [state for state in states if state]
    return states[0] if states else "UNKNOWN"


def wait_for_slurm_job(
    job_id: str | int,
    *,
    cluster: str | None = None,
    poll_seconds: float = COOLMUC4_POLICY.scheduler_poll_seconds,
    timeout: float | None = None,
) -> str:
    """Wait for a job using an LRZ-friendly low-frequency polling interval."""
    if poll_seconds < 60:
        raise ValueError("poll_seconds must be at least 60 seconds")
    active = {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "RESIZING",
        "SUSPENDED",
    }
    started = time.monotonic()
    while True:
        state = slurm_job_status(job_id, cluster=cluster).split("+", 1)[0]
        if state not in active:
            return state
        if timeout is not None and time.monotonic() - started > timeout:
            raise TimeoutError(f"Timed out waiting for Slurm job {job_id}")
        time.sleep(poll_seconds)


def cancel_slurm_job(
    job_id: str | int,
    *,
    cluster: str | None = None,
) -> None:
    """Cancel one Slurm job."""
    scancel = _require_command("scancel")
    subprocess.run([scancel, *_cluster_args(cluster), str(job_id)], check=True)


def _first_cpu_count(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"\s*([0-9]+)", value)
    return None if match is None else int(match.group(1))


def _positive_memory_gib(value_mb: str | None) -> float | None:
    if not value_mb:
        return None
    value = float(value_mb)
    # Slurm commonly represents an unconstrained/full-node memory allocation as
    # zero. Treat that as unknown/whole-node rather than literally zero GiB.
    return None if value <= 0 else value / 1024.0


def current_slurm_allocation() -> SlurmAllocation | None:
    """Read the current Slurm allocation from environment variables."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None

    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    cpus_per_node = _first_cpu_count(
        os.environ.get("SLURM_JOB_CPUS_PER_NODE")
        or os.environ.get("SLURM_CPUS_ON_NODE")
    )
    memory_per_node_gib = _positive_memory_gib(os.environ.get("SLURM_MEM_PER_NODE"))
    if memory_per_node_gib is None:
        mem_per_cpu_gib = _positive_memory_gib(os.environ.get("SLURM_MEM_PER_CPU"))
        if mem_per_cpu_gib is not None and cpus_per_node:
            memory_per_node_gib = mem_per_cpu_gib * cpus_per_node

    return SlurmAllocation(
        job_id=job_id,
        cluster=os.environ.get("SLURM_CLUSTER_NAME"),
        partition=os.environ.get("SLURM_JOB_PARTITION"),
        nodes=nodes,
        cpus_per_node=cpus_per_node,
        memory_per_node_gib=memory_per_node_gib,
        node_list=os.environ.get("SLURM_JOB_NODELIST"),
        tasks_per_node=_first_cpu_count(os.environ.get("SLURM_TASKS_PER_NODE")),
        cpus_per_task=_first_cpu_count(os.environ.get("SLURM_CPUS_PER_TASK")),
    )


def _validate_plan_against_allocation(
    plan: ExecutionPlan,
    allocation: SlurmAllocation,
) -> None:
    if allocation.nodes < plan.nodes:
        raise RuntimeError(
            f"Execution plan needs {plan.nodes} node(s), allocation has {allocation.nodes}."
        )
    if allocation.cpus_per_node is not None and allocation.cpus_per_node < plan.cores_per_node:
        raise RuntimeError(
            f"Execution plan needs {plan.cores_per_node} CPUs/node, allocation "
            f"exposes {allocation.cpus_per_node}."
        )
    if (
        allocation.memory_per_node_gib is not None
        and allocation.memory_per_node_gib + 1e-9 < plan.memory_per_node_gib
    ):
        raise RuntimeError(
            f"Execution plan assumes {plan.memory_per_node_gib:g} GiB/node, "
            f"allocation exposes {allocation.memory_per_node_gib:g} GiB/node."
        )


@contextmanager
def slurm_allocation_client(
    plan: ExecutionPlan,
    *,
    scheduler_file: str | Path | None = None,
    interface: str | None = None,
    worker_startup_timeout: float = 600.0,
    validate_topology: bool = True,
    strict_affinity: bool = False,
):
    """Start Dask workers as explicitly placed Slurm job steps.

    One Slurm task is created per Dask worker. ``--ntasks-per-node`` makes the
    per-node decomposition explicit, ``--cpus-per-task`` carries the Dask thread
    count into Slurm, and core binding makes the worker geometry physically
    testable. The observed placement is attached to the returned client as
    ``_hrhsa_topology_validation``.
    """
    allocation = current_slurm_allocation()
    if allocation is None:
        raise RuntimeError("slurm_allocation_client() must run inside an existing Slurm allocation")
    _validate_plan_against_allocation(plan, allocation)

    dask = _require_command("dask")
    srun = _require_command("srun")

    if scheduler_file is None:
        scheduler_path = Path.cwd() / f".hrhsa-dask-scheduler-{allocation.job_id}.json"
    else:
        scheduler_path = Path(scheduler_file).expanduser()
    scheduler_path = scheduler_path.resolve()
    scheduler_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler_path.unlink(missing_ok=True)

    scheduler_command = [
        dask,
        "scheduler",
        "--scheduler-file",
        str(scheduler_path),
        "--dashboard-address",
        ":0",
    ]
    if interface:
        scheduler_command.extend(["--interface", interface])

    scheduler = subprocess.Popen(scheduler_command)
    workers = None
    client = None
    try:
        deadline = time.monotonic() + worker_startup_timeout
        while not scheduler_path.is_file():
            if scheduler.poll() is not None:
                raise RuntimeError("Dask scheduler exited before publishing its file")
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for Dask scheduler file")
            time.sleep(1.0)

        total_workers = plan.total_workers
        memory_limit = f"{plan.managed_memory_per_worker_gib:.3f}GiB"
        local_directory = "${TMPDIR:-/tmp}/dask-${SLURM_PROCID}"
        worker_shell = " ".join(
            [
                "exec",
                shlex.quote(dask),
                "worker",
                "--scheduler-file",
                shlex.quote(str(scheduler_path)),
                "--nthreads",
                str(plan.geometry.threads_per_worker),
                "--memory-limit",
                shlex.quote(memory_limit),
                "--local-directory",
                f'"{local_directory}"',
            ]
        )
        if interface:
            worker_shell += f" --interface {shlex.quote(interface)}"

        worker_command = [
            srun,
            "--exact",
            f"--nodes={plan.nodes}",
            f"--ntasks={total_workers}",
            f"--ntasks-per-node={plan.geometry.workers}",
            f"--cpus-per-task={plan.geometry.threads_per_worker}",
            "--distribution=block:block",
            "--cpu-bind=cores",
            "--kill-on-bad-exit=1",
            "bash",
            "-lc",
            worker_shell,
        ]
        workers = subprocess.Popen(worker_command)

        from dask.distributed import Client

        client = Client(scheduler_file=str(scheduler_path))
        client.wait_for_workers(total_workers, timeout=worker_startup_timeout)

        if validate_topology:
            validation = validate_worker_topology(
                client,
                plan,
                strict_affinity=strict_affinity,
            )
            setattr(client, "_hrhsa_topology_validation", validation.as_dict())
            if not validation.ok:
                raise RuntimeError("Invalid Dask/Slurm placement:\n" + validation.explain())

        yield client
    finally:
        if client is not None:
            client.close()
        if workers is not None:
            workers.terminate()
            try:
                workers.wait(timeout=10)
            except subprocess.TimeoutExpired:
                workers.kill()
        scheduler.terminate()
        try:
            scheduler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            scheduler.kill()
        scheduler_path.unlink(missing_ok=True)
