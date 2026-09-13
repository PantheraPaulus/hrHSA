"""Execution configuration shared by scalable hrHSA workflows.

The public scientific API should not depend on how work is scheduled. This
module therefore keeps scheduler choices in one small immutable object which can
be passed into sampling, preparation, validation and benchmarking helpers.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ExecutionBackend = Literal["serial", "local", "distributed", "slurm"]


@dataclass(frozen=True)
class ExecutionConfig:
    """Describe how a computationally expensive hrHSA operation should run.

    Parameters
    ----------
    backend
        ``"serial"`` keeps the reference execution path. ``"local"`` creates a
        local Dask cluster, ``"distributed"`` expects an already running Dask
        client, and ``"slurm"`` creates a :class:`dask_jobqueue.SLURMCluster`.
    chunk_mb
        Target in-memory size of raster chunks used by chunk-aware operations.
    point_batch_mb
        Target size used when a workload must still be divided into row batches.
    point_batch_rows
        Optional explicit point-row batch size. When supplied, this overrides
        ``point_batch_mb`` for bounded in-memory point sampling.
    point_batches_in_flight
        Maximum number of bounded in-memory point-sampling calls that may overlap.
        This controls the legacy/client-driven GeoDataFrame path only.
    point_graph_partitions
        Optional number of spatial Parquet point partitions submitted in one
        Dask-native block-local graph window. ``None`` lets the partition-native
        sampler derive a bounded value from execution-thread capacity. Explicit
        values are useful for reproducible workstation/HPC calibration.
    cache_dir
        Optional persistent location for prepared Parquet products.
    n_workers, threads_per_worker, processes, memory_limit
        Dask worker configuration. One thread per process is the default because
        geospatial Python workloads frequently mix Python, GDAL and BLAS code and
        otherwise oversubscribe CPU cores. For SLURM, ``n_workers`` is translated
        into the required number of jobs when ``scale_jobs`` is not explicitly set.
    local_directory
        Directory used for Dask spill files. On SLURM this should normally point
        at node-local scratch (for example ``$TMPDIR``).
    dashboard_address
        Local Dask dashboard address. Set to ``None`` for non-interactive batch
        runs to avoid unnecessary dashboard servers and port collisions.
    worker_startup_timeout
        Maximum seconds to wait for requested SLURM workers before starting work.
    slurm_options
        Keyword arguments forwarded to :func:`hsa.compute.make_slurm_cluster`.
    """

    backend: ExecutionBackend = "serial"
    chunk_mb: int = 256
    point_batch_mb: int = 128
    point_batch_rows: int | None = None
    point_batches_in_flight: int = 1
    point_graph_partitions: int | None = None
    cache_dir: str | Path | None = None
    n_workers: int | None = None
    threads_per_worker: int = 1
    processes: bool = True
    memory_limit: str | int | None = "auto"
    local_directory: str | None = None
    dashboard_address: str | None = ":8787"
    worker_startup_timeout: float = 900.0
    slurm_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in {"serial", "local", "distributed", "slurm"}:
            raise ValueError(
                "backend must be 'serial', 'local', 'distributed' or 'slurm'."
            )
        if self.chunk_mb <= 0:
            raise ValueError("chunk_mb must be positive.")
        if self.point_batch_mb <= 0:
            raise ValueError("point_batch_mb must be positive.")
        if self.point_batch_rows is not None and self.point_batch_rows <= 0:
            raise ValueError("point_batch_rows must be positive when supplied.")
        if self.point_batches_in_flight <= 0:
            raise ValueError("point_batches_in_flight must be positive.")
        if self.point_graph_partitions is not None and self.point_graph_partitions <= 0:
            raise ValueError("point_graph_partitions must be positive when supplied.")
        if self.threads_per_worker <= 0:
            raise ValueError("threads_per_worker must be positive.")
        if self.n_workers is not None and self.n_workers <= 0:
            raise ValueError("n_workers must be positive when supplied.")
        if self.worker_startup_timeout <= 0:
            raise ValueError("worker_startup_timeout must be positive.")

    @property
    def distributed(self) -> bool:
        """Whether this configuration uses a distributed scheduler."""
        return self.backend in {"local", "distributed", "slurm"}

    @property
    def cache_path(self) -> Path | None:
        """Return ``cache_dir`` as a :class:`Path` when configured."""
        return None if self.cache_dir is None else Path(self.cache_dir)

    def create_client(self):
        """Create the Dask client implied by this configuration.

        ``backend='distributed'`` deliberately refuses to create a client: it is
        the mode used when a notebook or batch script already owns a scheduler.
        The caller should pass that client into the relevant hrHSA function.

        Returns
        -------
        client, cluster
            ``(None, None)`` for serial execution. Local and SLURM execution
            return both their client and cluster so callers can close scheduler
            resources deterministically.
        """
        if self.backend == "serial":
            return None, None
        if self.backend == "distributed":
            raise RuntimeError(
                "ExecutionConfig(backend='distributed') requires an existing "
                "Dask client; pass client=... to the operation."
            )
        if self.backend == "local":
            from hsa.compute.dask import make_local_dask_client

            client = make_local_dask_client(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                processes=self.processes,
                memory_limit=self.memory_limit,
                dashboard_address=self.dashboard_address,
                local_directory=self.local_directory or "dask-tmp",
            )
            if self.n_workers is not None:
                client.wait_for_workers(
                    self.n_workers,
                    timeout=self.worker_startup_timeout,
                )
            return client, getattr(client, "cluster", None)

        from dask.distributed import Client
        from hsa.compute.dask import make_slurm_cluster

        options = dict(self.slurm_options)
        options.setdefault(
            "local_directory",
            self.local_directory or "$TMPDIR",
        )

        expected_workers = self.n_workers
        if (
            expected_workers is not None
            and "scale_jobs" not in options
            and not bool(options.get("adapt", False))
        ):
            processes_per_job = max(1, int(options.get("processes", 1)))
            options["scale_jobs"] = int(
                math.ceil(expected_workers / processes_per_job)
            )

        cluster = make_slurm_cluster(**options)
        client = Client(cluster)
        if expected_workers is not None:
            client.wait_for_workers(
                expected_workers,
                timeout=self.worker_startup_timeout,
            )
        return client, cluster


def resolve_execution(
    execution: ExecutionConfig | None,
    *,
    client=None,
) -> tuple[ExecutionConfig, Any | None, Any | None, bool]:
    """Resolve execution state for a single operation.

    Returns ``(config, client, cluster, owns_client)``. The ownership flag lets
    callers close clients they created without closing clients supplied by a
    notebook or a surrounding workflow.
    """
    config = ExecutionConfig() if execution is None else execution

    if client is not None:
        return config, client, None, False

    if config.backend == "distributed":
        try:
            from dask.distributed import default_client

            return config, default_client(), None, False
        except Exception as exc:
            raise RuntimeError(
                "No active Dask client found for backend='distributed'."
            ) from exc

    created_client, cluster = config.create_client()
    return config, created_client, cluster, created_client is not None


def close_execution(client, cluster=None, *, owns_client: bool = False) -> None:
    """Close scheduler resources owned by the current operation."""
    if owns_client and client is not None:
        client.close()
    if cluster is not None:
        cluster.close()


@contextmanager
def execution_context(execution: ExecutionConfig, *, client=None):
    """Reuse one scheduler/client across a sequence of expensive operations.

    Repeatedly constructing LocalCluster or SLURMCluster instances is measurable
    overhead and can cause avoidable scheduler churn. This context manager makes
    ownership explicit while allowing sampling, fitting, validation and raster
    prediction to share one already-warmed client.

    Examples
    --------
    ``with execution_context(config) as client: ...``
    """
    config, resolved_client, cluster, owns_client = resolve_execution(
        execution,
        client=client,
    )
    try:
        yield resolved_client
    finally:
        close_execution(
            resolved_client,
            cluster,
            owns_client=owns_client,
        )
