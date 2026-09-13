from __future__ import annotations

from collections.abc import Callable
from typing import Any


def configure_dask_memory(
    *,
    target: float = 0.75,
    spill: float = 0.85,
    pause: float = 0.92,
    terminate: float = 0.98,
) -> None:
    """Configure Dask distributed worker memory thresholds.

    These defaults are intentionally conservative for raster/xarray workloads,
    where spilling a little early is usually preferable to killing a worker.
    """

    import dask

    dask.config.set(
        {
            "distributed.worker.memory.target": target,
            "distributed.worker.memory.spill": spill,
            "distributed.worker.memory.pause": pause,
            "distributed.worker.memory.terminate": terminate,
        }
    )


def shutdown_default_client() -> None:
    """Shutdown the active Dask default client if one exists."""

    from dask.distributed import default_client

    try:
        default_client().shutdown()
    except Exception:
        pass


def recommend_local_worker_count(*, threads_per_worker: int = 1) -> int:
    """Recommend a local worker count without oversubscribing physical cores.

    The workstation diagnostics showed that NumPy-heavy raster kernels can lose
    substantial efficiency when concurrency spills into SMT or when many active
    threads are packed into a few processes. The default therefore treats
    *physical cores* as the execution budget and divides that budget by the
    requested threads per worker.

    Examples on a 12-physical-core workstation are 12x1, 6x2 and 4x3. This is a
    topology-aware budget rule, not a claim that any one geometry is universally
    optimal. Supplying ``n_workers`` explicitly still overrides it.
    """
    if threads_per_worker <= 0:
        raise ValueError("threads_per_worker must be positive")

    try:
        from hsa.compute.topology import discover_runtime_topology

        observed = discover_runtime_topology()
        physical_cores = observed.physical_core_count
    except Exception:
        physical_cores = None

    if physical_cores is None or physical_cores <= 0:
        # Preserve the previous conservative fallback on platforms where physical
        # topology cannot be discovered (for example, restricted containers).
        import os

        physical_cores = max(1, (os.cpu_count() or 2) // 2)

    return max(1, int(physical_cores) // int(threads_per_worker))


def make_local_dask_client(
    *,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    processes: bool = True,
    memory_limit: str | int | None = "auto",
    dashboard_address: str | None = ":8787",
    local_directory: str | None = "dask-tmp",
    memory_target: float = 0.75,
    memory_spill: float = 0.85,
    memory_pause: float = 0.92,
    memory_terminate: float = 0.98,
):
    """Create a local Dask distributed client.

    This is the notebook/workstation backend. On HPC systems, prefer
    :func:`make_slurm_cluster` if ``dask-jobqueue`` is available.

    When ``n_workers`` is omitted, hrHSA now budgets against observed physical
    cores rather than logical/SMT CPUs. The budget is divided by
    ``threads_per_worker`` so changing from 1 to 2 threads does not silently
    double execution concurrency. Pass ``n_workers`` explicitly when SMT or
    oversubscription is intentional.
    """

    from dask.distributed import Client, LocalCluster

    if threads_per_worker <= 0:
        raise ValueError("threads_per_worker must be positive")

    configure_dask_memory(
        target=memory_target,
        spill=memory_spill,
        pause=memory_pause,
        terminate=memory_terminate,
    )

    if n_workers is None:
        n_workers = recommend_local_worker_count(
            threads_per_worker=threads_per_worker,
        )
    elif n_workers <= 0:
        raise ValueError("n_workers must be positive when supplied")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
        local_directory=local_directory,
    )
    return Client(cluster)


def get_or_create_client(*, local: bool = True, **kwargs):
    """Return the active Dask client or create a local one.

    Parameters are forwarded to :func:`make_local_dask_client` when no client is
    active and ``local=True``.
    """

    from dask.distributed import default_client

    try:
        return default_client()
    except Exception:
        if not local:
            raise RuntimeError("No active Dask client found and local=False.")
        return make_local_dask_client(**kwargs)


def make_slurm_cluster(
    *,
    queue: str | None = None,
    project: str | None = None,
    cores: int = 1,
    processes: int = 1,
    memory: str = "8GB",
    walltime: str = "02:00:00",
    local_directory: str | None = "$TMPDIR",
    job_extra_directives: list[str] | None = None,
    env_extra: list[str] | None = None,
    scale_jobs: int | None = None,
    adapt: bool = False,
    minimum_jobs: int = 0,
    maximum_jobs: int = 10,
    **kwargs,
):
    """Create a SLURM-backed Dask cluster using ``dask-jobqueue``.

    The function returns the cluster object, not a client. This keeps control in
    the caller's hands:

    ``cluster = make_slurm_cluster(..., scale_jobs=10)``
    ``client = Client(cluster)``

    ``dask-jobqueue`` is intentionally not a hard dependency of HSA; install it
    in the HPC environment when needed.
    """

    try:
        from dask_jobqueue import SLURMCluster
    except ImportError as exc:
        raise ImportError(
            "make_slurm_cluster requires dask-jobqueue. Install it with "
            "`conda install -c conda-forge dask-jobqueue` or add it to your HPC environment."
        ) from exc

    cluster_kwargs: dict[str, Any] = {
        "cores": cores,
        "processes": processes,
        "memory": memory,
        "walltime": walltime,
        "local_directory": local_directory,
    }
    if queue is not None:
        cluster_kwargs["queue"] = queue
    if project is not None:
        cluster_kwargs["account"] = project
    if job_extra_directives is not None:
        cluster_kwargs["job_extra_directives"] = job_extra_directives
    if env_extra is not None:
        cluster_kwargs["job_script_prologue"] = env_extra
    cluster_kwargs.update(kwargs)

    cluster = SLURMCluster(**cluster_kwargs)
    if scale_jobs is not None:
        cluster.scale(jobs=scale_jobs)
    if adapt:
        cluster.adapt(minimum_jobs=minimum_jobs, maximum_jobs=maximum_jobs)
    return cluster


def initialize_earth_engine_on_workers(
    client,
    *,
    project: str | None = None,
) -> dict:
    """Initialize Earth Engine on every Dask worker.

    Earth Engine state is process-local. If raster predictors are constructed or
    sampled inside Dask tasks, every worker needs its own ``ee.Initialize`` call.
    """

    def _init(project: str | None = None):
        import ee

        if project is None:
            ee.Initialize()
        else:
            ee.Initialize(project=project)
            ee.data.setCloudApiUserProject(project)
        return True

    return client.run(_init, project=project)


def run_on_workers(client, func: Callable[..., Any], *args, **kwargs) -> dict:
    """Run a small initialization/check function on all workers."""

    return client.run(func, *args, **kwargs)
