"""Reusable workload-size specifications for hrHSA performance campaigns.

These helpers deliberately express raster workloads in *logical uncompressed bytes*.
Storage compression is an implementation detail and may differ between machines,
filesystems, compressors and data distributions. For reproducible scaling studies
we instead choose square, storage-chunk-aligned raster windows whose logical size is
as close as possible to a requested GiB target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


GIB = 1024**3


@dataclass(frozen=True)
class RasterWorkload:
    """One square, chunk-aligned raster workload."""

    target_gib: float
    size: int
    bands: int
    dtype: str
    storage_chunk: int
    logical_bytes: int
    logical_gib: float
    spatial_chunks: int

    @property
    def cells(self) -> int:
        return self.size * self.size

    @property
    def relative_error_fraction(self) -> float:
        if self.target_gib == 0:
            return 0.0
        return self.logical_gib / self.target_gib - 1.0

    @property
    def label(self) -> str:
        target = f"{self.target_gib:g}".replace(".", "p")
        return f"raster-{target}gib"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_raster_parameters(
    *,
    target_gib: float,
    bands: int,
    dtype: str | np.dtype,
    storage_chunk: int,
) -> np.dtype:
    if not math.isfinite(target_gib) or target_gib <= 0:
        raise ValueError("target_gib must be a finite positive number")
    if bands <= 0:
        raise ValueError("bands must be positive")
    if storage_chunk <= 0:
        raise ValueError("storage_chunk must be positive")
    return np.dtype(dtype)


def raster_workload_for_gib(
    target_gib: float,
    *,
    bands: int = 6,
    dtype: str | np.dtype = "float32",
    storage_chunk: int = 1024,
) -> RasterWorkload:
    """Return the closest square storage-chunk-aligned raster to ``target_gib``.

    The requested size is treated as logical uncompressed raster input. Alignment
    to ``storage_chunk`` makes nested benchmark windows correspond to complete Zarr
    chunks and prevents a scaling point from paying a disproportionate tail-chunk
    penalty.
    """

    dtype_obj = _validate_raster_parameters(
        target_gib=target_gib,
        bands=bands,
        dtype=dtype,
        storage_chunk=storage_chunk,
    )
    target_bytes = float(target_gib) * GIB
    bytes_per_xy_cell = bands * dtype_obj.itemsize
    ideal_side = math.sqrt(target_bytes / bytes_per_xy_cell)

    lower = max(storage_chunk, int(math.floor(ideal_side / storage_chunk)) * storage_chunk)
    upper = max(storage_chunk, int(math.ceil(ideal_side / storage_chunk)) * storage_chunk)

    def logical_bytes(side: int) -> int:
        return int(side * side * bytes_per_xy_cell)

    candidates = sorted({lower, upper})
    size = min(candidates, key=lambda side: (abs(logical_bytes(side) - target_bytes), side))
    nbytes = logical_bytes(size)
    chunks_per_axis = math.ceil(size / storage_chunk)

    return RasterWorkload(
        target_gib=float(target_gib),
        size=int(size),
        bands=int(bands),
        dtype=dtype_obj.name,
        storage_chunk=int(storage_chunk),
        logical_bytes=nbytes,
        logical_gib=nbytes / GIB,
        spatial_chunks=int(chunks_per_axis**2),
    )


def raster_workloads(
    targets_gib: Iterable[float],
    *,
    bands: int = 6,
    dtype: str | np.dtype = "float32",
    storage_chunk: int = 1024,
) -> list[RasterWorkload]:
    """Resolve and validate a monotonically increasing raster workload family."""

    workloads = [
        raster_workload_for_gib(
            float(target),
            bands=bands,
            dtype=dtype,
            storage_chunk=storage_chunk,
        )
        for target in targets_gib
    ]
    if not workloads:
        raise ValueError("at least one raster workload is required")

    targets = [workload.target_gib for workload in workloads]
    if targets != sorted(targets) or len(set(targets)) != len(targets):
        raise ValueError("raster workload targets must be unique and increasing")

    sizes = [workload.size for workload in workloads]
    if sizes != sorted(sizes) or len(set(sizes)) != len(sizes):
        raise ValueError(
            "requested targets collapse to duplicate/non-increasing chunk-aligned "
            "raster sizes; use more widely separated targets"
        )
    return workloads


def weak_raster_workloads(
    resource_counts: Iterable[int],
    *,
    gib_per_resource: float,
    bands: int = 6,
    dtype: str | np.dtype = "float32",
    storage_chunk: int = 1024,
) -> list[tuple[int, RasterWorkload]]:
    """Return weak-scaling workloads at constant logical GiB/resource."""

    counts = [int(value) for value in resource_counts]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("resource_counts must contain positive integers")
    if counts != sorted(counts) or len(set(counts)) != len(counts):
        raise ValueError("resource_counts must be unique and increasing")
    if not math.isfinite(gib_per_resource) or gib_per_resource <= 0:
        raise ValueError("gib_per_resource must be a finite positive number")

    targets = [count * float(gib_per_resource) for count in counts]
    workloads = raster_workloads(
        targets,
        bands=bands,
        dtype=dtype,
        storage_chunk=storage_chunk,
    )
    return list(zip(counts, workloads))


def parse_positive_floats(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("expected comma-separated positive finite numbers")
    return values


def parse_positive_ints(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("expected comma-separated positive integers")
    return values


def parse_geometry(text: str) -> tuple[int, int]:
    """Parse ``workers x threads`` into positive integers."""

    try:
        workers_text, threads_text = text.lower().strip().split("x", 1)
        workers = int(workers_text)
        threads = int(threads_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid geometry {text!r}; expected e.g. '6x2'") from exc
    if workers <= 0 or threads <= 0:
        raise ValueError("geometry workers and threads must be positive")
    return workers, threads


def geometry_for_core_budget(base_geometry: str, cores: int) -> tuple[int, int]:
    """Preserve worker thread width while scaling a local physical-core budget."""

    _, threads = parse_geometry(base_geometry)
    if cores <= 0:
        raise ValueError("cores must be positive")
    if cores % threads:
        raise ValueError(
            f"{cores} cores cannot preserve {threads} threads/worker from "
            f"base geometry {base_geometry}"
        )
    return cores // threads, threads
