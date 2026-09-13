"""Workload-aware planning helpers for chunked raster execution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np
import xarray as xr

from hsa.compute.chunking import suggest_xy_chunks


RasterWorkload = Literal["point_sampling", "surface_prediction", "generic"]


class GeometryLike(Protocol):
    """Minimal interface needed by :func:`recommend_worker_geometry`."""

    workers: int
    threads_per_worker: int
    label: str

    @property
    def total_threads(self) -> int: ...


def storage_chunk_shape(env: xr.DataArray) -> dict[str, int] | None:
    """Best-effort discovery of the physical/source chunk shape."""
    encoded = env.encoding.get("chunks")
    if encoded is not None and len(encoded) == env.ndim:
        return {
            dim: int(size)
            for dim, size in zip(env.dims, encoded)
            if isinstance(size, (int, np.integer)) and int(size) > 0
        }

    data_chunks = getattr(env.data, "chunks", None)
    if data_chunks is None:
        return None
    return {
        dim: int(chunks[0])
        for dim, chunks in zip(env.dims, data_chunks)
        if chunks
    }


def _nearest_multiple(value: int, base: int, *, maximum: int) -> int:
    if base <= 0:
        return min(value, maximum)
    multiple = max(1, int(round(value / base)))
    return min(maximum, multiple * base)


def plan_raster_chunks(
    env: xr.DataArray,
    *,
    workload: RasterWorkload = "generic",
    target_chunk_mb: int = 256,
    align_storage: bool = True,
    selected_band_count: int | None = None,
) -> dict[str, int]:
    """Plan computational chunks using memory target and source layout."""
    if workload not in {"point_sampling", "surface_prediction", "generic"}:
        raise ValueError(
            "workload must be 'point_sampling', 'surface_prediction' or 'generic'."
        )
    if target_chunk_mb <= 0:
        raise ValueError("target_chunk_mb must be positive.")
    if "x" not in env.dims or "y" not in env.dims:
        raise ValueError("plan_raster_chunks expects x and y dimensions.")

    base = suggest_xy_chunks(env, target_chunk_mb=target_chunk_mb)
    if selected_band_count is not None and "band" in env.dims:
        n_bands = max(1, int(selected_band_count))
        itemsize = np.dtype(env.dtype).itemsize
        target_bytes = target_chunk_mb * 1024**2
        side = int(math.sqrt(max(1, target_bytes // (n_bands * itemsize))))
        base["x"] = min(int(env.sizes["x"]), max(256, side))
        base["y"] = min(int(env.sizes["y"]), max(256, side))

    base["band"] = -1 if "band" in env.dims else 1
    if not align_storage:
        return base

    source = storage_chunk_shape(env)
    if not source:
        return base

    for dimension in ("x", "y"):
        storage_size = source.get(dimension)
        if storage_size:
            base[dimension] = _nearest_multiple(
                int(base[dimension]),
                int(storage_size),
                maximum=int(env.sizes[dimension]),
            )
    return base


def spatial_task_count(env: xr.DataArray, chunks: dict[str, int]) -> int:
    """Estimate the number of independent spatial chunk tasks in ``env``."""
    if "x" not in env.dims or "y" not in env.dims:
        raise ValueError("spatial_task_count expects x and y dimensions.")
    x_chunk = int(chunks.get("x", env.sizes["x"]))
    y_chunk = int(chunks.get("y", env.sizes["y"]))
    if x_chunk <= 0 or y_chunk <= 0:
        raise ValueError("x/y chunk lengths must be positive.")
    return int(
        math.ceil(int(env.sizes["x"]) / x_chunk)
        * math.ceil(int(env.sizes["y"]) / y_chunk)
    )


def recommend_worker_count(
    task_count: int,
    requested_workers: int,
    *,
    min_tasks_per_worker: int = 8,
) -> int:
    """Cap a requested worker count using a conservative task-density heuristic.

    Retained for backwards compatibility. New execution planning should prefer
    :func:`recommend_worker_geometry`.
    """
    if task_count <= 0:
        raise ValueError("task_count must be positive.")
    if requested_workers <= 0:
        raise ValueError("requested_workers must be positive.")
    if min_tasks_per_worker <= 0:
        raise ValueError("min_tasks_per_worker must be positive.")
    useful_workers = max(1, int(task_count) // int(min_tasks_per_worker))
    return min(int(requested_workers), useful_workers)


def recommend_worker_geometry(
    candidates: Sequence[GeometryLike],
    *,
    task_count: int | None = None,
    preferred_labels: Sequence[str] = (),
    min_tasks_per_worker: int = 8,
) -> GeometryLike:
    """Choose a process/thread geometry, using task density as a constraint.

    Benchmark-derived preferences are honored first. Otherwise the fallback favors
    a balanced process/thread decomposition near the square root of total CPU
    concurrency, avoiding both one process per core and one enormous worker.
    """
    if not candidates:
        raise ValueError("candidates must not be empty.")
    if min_tasks_per_worker <= 0:
        raise ValueError("min_tasks_per_worker must be positive.")

    pool = list(candidates)
    for geometry in pool:
        if geometry.workers <= 0 or geometry.threads_per_worker <= 0:
            raise ValueError("worker geometries must contain positive counts.")

    if task_count is not None:
        if task_count <= 0:
            raise ValueError("task_count must be positive when supplied.")
        useful_workers = max(1, int(task_count) // int(min_tasks_per_worker))
        eligible = [g for g in pool if g.workers <= useful_workers]
        if eligible:
            pool = eligible
        else:
            minimum_workers = min(g.workers for g in pool)
            pool = [g for g in pool if g.workers == minimum_workers]

    by_label = {g.label: g for g in pool}
    for label in preferred_labels:
        if label in by_label:
            return by_label[label]

    total_threads = max(g.total_threads for g in pool)
    target_workers = math.sqrt(total_threads)
    return min(
        pool,
        key=lambda g: (
            abs(math.log2(g.workers / target_workers)),
            g.workers,
            -g.threads_per_worker,
        ),
    )


def task_density(task_count: int, workers: int) -> float:
    """Return independent tasks per worker for benchmark diagnostics."""
    if task_count <= 0:
        raise ValueError("task_count must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    return float(task_count) / float(workers)
