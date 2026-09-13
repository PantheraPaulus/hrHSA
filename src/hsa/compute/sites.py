"""Site-aware and benchmark-informed execution planning for HPC systems."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from hsa.compute.planning import recommend_worker_geometry
from hsa.compute.profiles import (
    COOLMUC4_HARDWARE,
    COOLMUC4_POLICY,
    HardwareTopology,
    SitePolicy,
    Workload,
    benchmark_profile_for,
)


@dataclass(frozen=True)
class WorkerGeometry:
    """One Dask process/thread layout for a fixed physical-core budget."""

    workers: int
    threads_per_worker: int
    cores: int
    label: str

    @property
    def total_threads(self) -> int:
        return self.workers * self.threads_per_worker


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved execution geometry, memory policy and benchmark provenance."""

    site: str
    workload: Workload
    nodes: int
    cores_per_node: int
    geometry: WorkerGeometry
    partition: str
    memory_per_node_gib: float
    managed_memory_per_worker_gib: float
    chunk_mb: int
    evidence: str
    benchmark_profile: str | None = None
    benchmark_confidence: str | None = None
    memory_reserve_per_node_gib: float = 0.0
    placement: str = "block:block/core-bound"

    @property
    def total_workers(self) -> int:
        return self.nodes * self.geometry.workers

    @property
    def total_cores(self) -> int:
        return self.nodes * self.cores_per_node

    @property
    def managed_memory_per_node_gib(self) -> float:
        return self.managed_memory_per_worker_gib * self.geometry.workers

    @property
    def workers_per_socket(self) -> float | None:
        if self.site != "coolmuc4":
            return None
        return self.geometry.workers / COOLMUC4_HARDWARE.sockets_per_node

    def as_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "workload": self.workload,
            "nodes": self.nodes,
            "cores_per_node": self.cores_per_node,
            "total_cores": self.total_cores,
            "workers_per_node": self.geometry.workers,
            "threads_per_worker": self.geometry.threads_per_worker,
            "workers_per_socket": self.workers_per_socket,
            "total_workers": self.total_workers,
            "geometry": self.geometry.label,
            "placement": self.placement,
            "partition": self.partition,
            "memory_per_node_gib": self.memory_per_node_gib,
            "memory_reserve_per_node_gib": self.memory_reserve_per_node_gib,
            "managed_memory_per_node_gib": self.managed_memory_per_node_gib,
            "managed_memory_per_worker_gib": self.managed_memory_per_worker_gib,
            "chunk_mb": self.chunk_mb,
            "benchmark_profile": self.benchmark_profile,
            "benchmark_confidence": self.benchmark_confidence,
            "evidence": self.evidence,
        }

    def explain(self) -> str:
        rows = self.as_dict()
        width = max(len(key) for key in rows)
        return "\n".join(f"{key:<{width}} : {value}" for key, value in rows.items())


@dataclass(frozen=True)
class HPCSiteProfile:
    """Composition of site policy, hardware facts and conservative defaults.

    Compatibility properties keep the pre-profile public API available while the
    underlying concepts remain independently versionable.
    """

    name: str
    policy: SitePolicy
    hardware: HardwareTopology
    conservative_job_memory_gib: float = 224.0

    @property
    def physical_cores_per_node(self) -> int:
        return self.hardware.physical_cores_per_node

    @property
    def sockets_per_node(self) -> int:
        return self.hardware.sockets_per_node

    @property
    def cores_per_socket(self) -> int:
        return self.hardware.cores_per_socket

    @property
    def memory_gib_per_node(self) -> float:
        return self.hardware.memory_gib_per_node

    @property
    def scratch_env(self) -> str:
        return self.hardware.scratch_env

    @property
    def local_scratch_env(self) -> str:
        return self.hardware.local_scratch_env

    @property
    def interconnect(self) -> str | None:
        return self.hardware.interconnect

    @property
    def shared_partition(self) -> str:
        return self.policy.shared_partition

    @property
    def shared_qos(self) -> str:
        return self.policy.shared_qos

    @property
    def full_node_partition(self) -> str:
        return self.policy.full_node_partition

    @property
    def full_node_qos(self) -> str:
        return self.policy.full_node_qos

    @property
    def min_parallel_cores_per_node(self) -> int:
        return self.policy.min_parallel_cores_per_node

    @property
    def max_parallel_nodes(self) -> int:
        return self.policy.max_parallel_nodes

    @property
    def policy_memory_ceiling_gib(self) -> float:
        return self.policy.max_memory_per_node_gib

    def scratch_path(self) -> str | None:
        return os.environ.get(self.scratch_env)

    def local_scratch_path(self) -> str | None:
        return os.environ.get(self.local_scratch_env)

    def memory_per_worker_gib(
        self,
        workers_per_node: int,
        *,
        reserve_gib: float = 32.0,
        managed_fraction: float = 0.85,
        budget_gib: float | None = None,
    ) -> float:
        """Recommend managed Dask memory per worker with job-level headroom."""
        if workers_per_node <= 0:
            raise ValueError("workers_per_node must be positive")
        budget = float(
            budget_gib
            if budget_gib is not None
            else self.conservative_job_memory_gib
        )
        if reserve_gib < 0 or reserve_gib >= budget:
            raise ValueError("reserve_gib must be between 0 and the job memory budget")
        if not 0 < managed_fraction <= 1:
            raise ValueError("managed_fraction must lie in (0, 1]")
        usable = (budget - reserve_gib) * managed_fraction
        return usable / workers_per_node

    @staticmethod
    def _geometry_candidates(cores: int) -> tuple[WorkerGeometry, ...]:
        candidates: list[WorkerGeometry] = []
        for threads in (1, 2, 4, 7, 8, 14, 16, 28, 56, 112):
            if threads > cores or cores % threads:
                continue
            workers = cores // threads
            candidates.append(
                WorkerGeometry(
                    workers=workers,
                    threads_per_worker=threads,
                    cores=cores,
                    label=f"{workers}x{threads}",
                )
            )
        return tuple(candidates)

    def socket_geometry_candidates(self) -> tuple[WorkerGeometry, ...]:
        return self._geometry_candidates(self.cores_per_socket)

    def node_geometry_candidates(
        self,
        cores: int | None = None,
    ) -> tuple[WorkerGeometry, ...]:
        return self._geometry_candidates(cores or self.physical_cores_per_node)

    def recommend_geometry(
        self,
        *,
        workload: Workload = "generic",
        cores: int | None = None,
        task_count: int | None = None,
    ) -> WorkerGeometry:
        """Recommend geometry using dated evidence, then a generic fallback."""
        cores = int(cores or self.physical_cores_per_node)
        candidates = self.node_geometry_candidates(cores)
        benchmark = benchmark_profile_for(
            site=self.name,
            workload=workload,
            cores_per_node=cores,
        )
        preferred = () if benchmark is None else benchmark.preferred_geometries
        return recommend_worker_geometry(
            candidates,
            task_count=task_count,
            preferred_labels=preferred,
        )

    def recommend_chunk_mb(
        self,
        geometry: WorkerGeometry,
        *,
        working_set_multiplier: float = 4.0,
        memory_fraction: float = 0.60,
        minimum_mb: int = 64,
        maximum_mb: int = 512,
    ) -> int:
        """Bound chunk size by concurrent per-worker memory pressure."""
        if working_set_multiplier <= 0:
            raise ValueError("working_set_multiplier must be positive")
        if not 0 < memory_fraction <= 1:
            raise ValueError("memory_fraction must lie in (0, 1]")
        worker_gib = self.memory_per_worker_gib(geometry.workers)
        per_task_gib = (
            worker_gib
            * memory_fraction
            / (geometry.threads_per_worker * working_set_multiplier)
        )
        value = int(per_task_gib * 1024)
        return max(minimum_mb, min(maximum_mb, value))


COOLMUC4 = HPCSiteProfile(
    name="coolmuc4",
    policy=COOLMUC4_POLICY,
    hardware=COOLMUC4_HARDWARE,
    conservative_job_memory_gib=224.0,
)

SITE_PROFILES = {COOLMUC4.name: COOLMUC4}


def get_site_profile(name: str = "auto") -> HPCSiteProfile | None:
    """Return a known site profile, optionally detecting CoolMUC-4 from Slurm."""
    normalized = name.lower().strip()
    if normalized == "auto":
        cluster = os.environ.get("SLURM_CLUSTER_NAME", "").lower()
        partition = os.environ.get("SLURM_JOB_PARTITION", "").lower()
        if cluster == COOLMUC4_POLICY.cluster or partition.startswith("cm4_"):
            return COOLMUC4
        return None
    try:
        return SITE_PROFILES[normalized]
    except KeyError as exc:
        raise KeyError(
            f"Unknown HPC site profile {name!r}; available: {sorted(SITE_PROFILES)}"
        ) from exc


def _parse_geometry(value: str, *, cores: int) -> WorkerGeometry:
    try:
        workers_text, threads_text = value.lower().split("x", 1)
        workers = int(workers_text)
        threads = int(threads_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError("geometry must look like '8x14' or be 'auto'") from exc
    if workers <= 0 or threads <= 0 or workers * threads != cores:
        raise ValueError(
            f"geometry {value!r} must use exactly {cores} physical cores"
        )
    return WorkerGeometry(workers, threads, cores, f"{workers}x{threads}")


def coolmuc4_plan(
    *,
    workload: Workload = "surface_prediction",
    nodes: int = 1,
    cores_per_node: int = 112,
    geometry: str = "auto",
    task_count: int | None = None,
    memory_per_node_gib: float | None = None,
    chunk_mb: int | None = None,
) -> ExecutionPlan:
    """Create a policy-valid, benchmark-informed CoolMUC-4 execution plan."""
    if not 1 <= nodes <= COOLMUC4.max_parallel_nodes:
        raise ValueError(f"CoolMUC-4 plans support 1-{COOLMUC4.max_parallel_nodes} nodes")
    if cores_per_node <= 0 or cores_per_node > COOLMUC4.physical_cores_per_node:
        raise ValueError("cores_per_node must lie in 1..112")
    if nodes > 1 and cores_per_node != COOLMUC4.physical_cores_per_node:
        raise ValueError("cm4_std multi-node jobs must fill each 112-core node")

    benchmark = benchmark_profile_for(
        site=COOLMUC4.name,
        workload=workload,
        cores_per_node=cores_per_node,
    )

    if geometry == "auto":
        resolved = COOLMUC4.recommend_geometry(
            workload=workload,
            cores=cores_per_node,
            task_count=task_count,
        )
    else:
        resolved = _parse_geometry(geometry, cores=cores_per_node)

    if nodes == 1:
        if cores_per_node < COOLMUC4.min_parallel_cores_per_node:
            raise ValueError(
                f"cm4_tiny currently requires at least {COOLMUC4.min_parallel_cores_per_node} "
                "physical cores; use serial_std for 1-16 core jobs"
            )
        partition = COOLMUC4.shared_partition
    else:
        partition = COOLMUC4.full_node_partition

    if memory_per_node_gib is None:
        budget = float(COOLMUC4.conservative_job_memory_gib)
        if nodes == 1:
            fraction = cores_per_node / COOLMUC4.physical_cores_per_node
            memory_per_node_gib = max(8.0, budget * fraction)
        else:
            # This is the hrHSA working-memory budget, not a claim that cm4_std
            # exposes only 224 GiB. Full CoolMUC-4 nodes have ~488 GiB available.
            memory_per_node_gib = budget

    ceiling = COOLMUC4.policy_memory_ceiling_gib
    if memory_per_node_gib <= 0 or memory_per_node_gib > ceiling:
        raise ValueError(
            f"memory_per_node_gib must be in (0, {ceiling:g}] under current LRZ policy"
        )

    reserve_gib = min(32.0, max(1.0, memory_per_node_gib * 0.25))
    managed_memory = COOLMUC4.memory_per_worker_gib(
        resolved.workers,
        reserve_gib=reserve_gib,
        managed_fraction=1.0,
        budget_gib=memory_per_node_gib,
    )

    if chunk_mb is None:
        if benchmark is not None and benchmark.chunk_mb is not None:
            chunk_mb = benchmark.chunk_mb
        else:
            chunk_mb = COOLMUC4.recommend_chunk_mb(resolved)

    evidence = (
        benchmark.evidence
        if benchmark is not None
        else "Generic balanced process/thread heuristic; validate with benchmarks."
    )

    return ExecutionPlan(
        site=COOLMUC4.name,
        workload=workload,
        nodes=nodes,
        cores_per_node=cores_per_node,
        geometry=resolved,
        partition=partition,
        memory_per_node_gib=float(memory_per_node_gib),
        managed_memory_per_worker_gib=float(managed_memory),
        chunk_mb=int(chunk_mb),
        evidence=evidence,
        benchmark_profile=None if benchmark is None else benchmark.name,
        benchmark_confidence=None if benchmark is None else benchmark.confidence,
        memory_reserve_per_node_gib=float(reserve_gib),
    )


def coolmuc4_execution(
    *,
    workload: Workload = "surface_prediction",
    nodes: int = 1,
    workers_per_node: int | None = None,
    threads_per_worker: int | None = None,
    geometry: str = "auto",
    task_count: int | None = None,
    partition: str | None = None,
    memory_per_node_gib: float | None = None,
    walltime: str = "02:00:00",
    project: str | None = None,
    interface: str | None = None,
    chunk_mb: int | None = None,
    cache_dir: str | None = None,
    worker_startup_timeout: float = 1800.0,
):
    """Build a one-node ``dask-jobqueue`` CoolMUC-4 execution config.

    For benchmark-matched CPU placement or multi-node ``cm4_std`` allocations,
    create :func:`coolmuc4_plan` and use
    :func:`hsa.compute.slurm_allocation_client` inside the allocation.
    """
    from hsa.compute.config import ExecutionConfig

    if nodes != 1:
        raise ValueError(
            "coolmuc4_execution() supports one cm4_tiny node only; use "
            "coolmuc4_plan(nodes=...) with slurm_allocation_client() for cm4_std"
        )

    if (workers_per_node is None) != (threads_per_worker is None):
        raise ValueError("workers_per_node and threads_per_worker must be supplied together")
    if workers_per_node is not None:
        cores = workers_per_node * int(threads_per_worker)
        explicit = WorkerGeometry(
            workers=int(workers_per_node),
            threads_per_worker=int(threads_per_worker),
            cores=cores,
            label=f"{workers_per_node}x{threads_per_worker}",
        )
        if cores > COOLMUC4.physical_cores_per_node:
            raise ValueError(
                f"Requested {cores} worker cores, but CoolMUC-4 has "
                f"{COOLMUC4.physical_cores_per_node} physical cores per node."
            )
        plan = coolmuc4_plan(
            workload=workload,
            nodes=1,
            cores_per_node=cores,
            geometry=explicit.label,
            task_count=task_count,
            memory_per_node_gib=memory_per_node_gib,
            chunk_mb=chunk_mb,
        )
    else:
        plan = coolmuc4_plan(
            workload=workload,
            nodes=1,
            cores_per_node=112,
            geometry=geometry,
            task_count=task_count,
            memory_per_node_gib=memory_per_node_gib,
            chunk_mb=chunk_mb,
        )

    partition = partition or plan.partition
    if partition != COOLMUC4.shared_partition:
        raise ValueError(
            "coolmuc4_execution() supports cm4_tiny only; use "
            "coolmuc4_plan(nodes=...) with slurm_allocation_client() for cm4_std"
        )

    directives = [
        f"--clusters={COOLMUC4_POLICY.cluster}",
        f"--qos={COOLMUC4.shared_qos}",
        "--hint=nomultithread",
        "--get-user-env",
        "--export=NONE",
    ]
    prologue = [
        "module load slurm_setup",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
    ]

    slurm_options = {
        "queue": partition,
        "project": project,
        "cores": plan.geometry.cores,
        "processes": plan.geometry.workers,
        "memory": f"{int(math.floor(plan.memory_per_node_gib))}GB",
        "walltime": walltime,
        "job_extra_directives": directives,
        "env_extra": prologue,
        "scheduler_options": {"dashboard_address": None},
        "name": "hrhsa",
    }
    if interface:
        slurm_options["interface"] = interface

    return ExecutionConfig(
        backend="slurm",
        n_workers=plan.geometry.workers,
        threads_per_worker=plan.geometry.threads_per_worker,
        local_directory="$TMPDIR",
        dashboard_address=None,
        chunk_mb=plan.chunk_mb,
        cache_dir=cache_dir,
        worker_startup_timeout=worker_startup_timeout,
        slurm_options=slurm_options,
    )
