"""Reusable point-count workload specifications for hrHSA benchmarks.

Each point-count workload is stored independently as a spatially tiled Parquet
collection.  ``partition_size`` is a target upper bound on rows per tile, not a
point-ID range.  Keeping every tile spatially local lets the chunk-aware raster
sampler preserve its core I/O property: a raster chunk is loaded only for nearby
points rather than being revisited once per arbitrary point batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class PointWorkload:
    """One exact point-count workload with balanced spatial partitions."""

    target_points: int
    target_partition_size: int
    partitions: int
    base_points_per_partition: int
    extra_partitions: int

    @property
    def max_points_per_partition(self) -> int:
        return self.base_points_per_partition + int(self.extra_partitions > 0)

    @property
    def label(self) -> str:
        return f"points-{self.target_points}"

    def points_in_partition(self, index: int) -> int:
        if index < 0 or index >= self.partitions:
            raise IndexError(index)
        return self.base_points_per_partition + int(index < self.extra_partitions)

    def start_point_id(self, index: int) -> int:
        if index < 0 or index >= self.partitions:
            raise IndexError(index)
        return (
            index * self.base_points_per_partition
            + min(index, self.extra_partitions)
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def point_workload_for_count(
    target_points: int,
    *,
    partition_size: int = 1_000_000,
) -> PointWorkload:
    """Resolve an exact point count into balanced bounded spatial partitions."""

    target = int(target_points)
    target_partition = int(partition_size)
    if target <= 0:
        raise ValueError("target_points must be positive")
    if target_partition <= 0:
        raise ValueError("partition_size must be positive")

    partitions = max(1, math.ceil(target / target_partition))
    base, extra = divmod(target, partitions)
    return PointWorkload(
        target_points=target,
        target_partition_size=target_partition,
        partitions=partitions,
        base_points_per_partition=base,
        extra_partitions=extra,
    )


def point_workloads(
    targets: Iterable[int],
    *,
    partition_size: int = 1_000_000,
) -> list[PointWorkload]:
    """Resolve a unique increasing family of exact point-count workloads."""

    values = [int(value) for value in targets]
    if not values or any(value <= 0 for value in values):
        raise ValueError("point targets must contain positive integers")
    if values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("point targets must be unique and increasing")
    return [
        point_workload_for_count(value, partition_size=partition_size)
        for value in values
    ]


def weak_point_workloads(
    resource_counts: Iterable[int],
    *,
    points_per_resource: int,
    partition_size: int = 1_000_000,
) -> list[tuple[int, PointWorkload]]:
    """Return weak-scaling point workloads at fixed points per resource."""

    counts = [int(value) for value in resource_counts]
    density = int(points_per_resource)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("resource_counts must contain positive integers")
    if counts != sorted(counts) or len(set(counts)) != len(counts):
        raise ValueError("resource_counts must be unique and increasing")
    if density <= 0:
        raise ValueError("points_per_resource must be positive")

    return [
        (
            count,
            point_workload_for_count(
                count * density,
                partition_size=partition_size,
            ),
        )
        for count in counts
    ]


__all__ = [
    "PointWorkload",
    "point_workload_for_count",
    "point_workloads",
    "weak_point_workloads",
]
