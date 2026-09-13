"""Helpers for materializing and releasing distributed Dask collections."""

from __future__ import annotations

from typing import Any


def persist_distributed(client, collection: Any):
    """Compute a Dask collection while preserving its distributed partitioning.

    ``Client.compute`` turns a collection such as a Dask Array into one final
    concrete result. For large raster benchmarks that can introduce an
    artificial final gather on a single worker. ``Client.persist`` instead
    replaces each graph partition/chunk with a distributed Future, leaving the
    result partitioned across workers.

    The returned object has the same Dask collection type as ``collection`` and
    is fully materialized before this function returns.
    """

    if client is None:
        raise ValueError("persist_distributed requires an active Dask Client")

    from dask.distributed import wait

    persisted = client.persist(collection)
    wait(persisted)
    return persisted


def cancel_distributed(client, collection: Any) -> None:
    """Release a previously persisted distributed collection."""

    if client is None or collection is None:
        return
    try:
        client.cancel(collection, force=True)
    except TypeError:  # compatibility with Distributed versions without force=
        client.cancel(collection)


def _worker_gc() -> int:
    """Collect unreachable Python objects inside one Dask worker process."""

    import gc

    return gc.collect()


def release_distributed(
    client,
    collection: Any,
    *,
    collect_workers: bool = True,
    suppress_worker_errors: bool = True,
) -> None:
    """Release persisted data and optionally run an untimed worker GC barrier.

    Cancellation removes distributed keys synchronously from the scheduler, but
    Python objects, allocator arenas and communication buffers on workers may
    outlive those keys briefly. Repeated large raster materializations can
    therefore overlap retained process state even though the previous collection
    has been cancelled.

    This helper centralizes the cleanup protocol used by repeated hrHSA
    benchmarks: cancel the distributed collection, then force one worker-local
    ``gc.collect()`` call before the next repeat. Benchmark drivers should call
    this outside the timed section.

    Parameters
    ----------
    client
        Active Dask client.
    collection
        Previously persisted distributed collection, or ``None``.
    collect_workers
        Run ``gc.collect()`` on every active worker after cancellation.
    suppress_worker_errors
        Preserve an original benchmark exception if the scheduler is already
        unavailable during failure cleanup. Set to ``False`` when cleanup failure
        itself should be surfaced.
    """

    cancel_distributed(client, collection)

    if client is None or not collect_workers:
        return

    try:
        client.run(_worker_gc)
    except Exception:
        if not suppress_worker_errors:
            raise


__all__ = [
    "persist_distributed",
    "cancel_distributed",
    "release_distributed",
]
