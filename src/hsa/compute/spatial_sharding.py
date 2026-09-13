"""Workload-balanced spatial decomposition for partitioned point sampling.

The partitioned point sampler benefits from running independent spatial domains,
but a regular grid can leave a long critical-path tail when point density is not
perfectly uniform or when the requested shard count does not divide the prepared
tile grid.  This module plans compact shards by recursively bisecting spatial
partitions while targeting equal point-row weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hsa.compute.partitioned import Bounds, PointPartition


@dataclass(frozen=True)
class SpatialPointShard:
    """One workload-balanced spatial group of point partitions."""

    partition_indices: tuple[int, ...]
    rows: int
    bounds: Bounds

    @property
    def partition_count(self) -> int:
        return len(self.partition_indices)


def _union_bounds(partitions: Sequence[PointPartition], indices: Sequence[int]) -> Bounds:
    xmin = min(partitions[index].bounds[0] for index in indices)
    ymin = min(partitions[index].bounds[1] for index in indices)
    xmax = max(partitions[index].bounds[2] for index in indices)
    ymax = max(partitions[index].bounds[3] for index in indices)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _footprint_cost(bounds: Bounds) -> float:
    """Bounding-box area used as a proxy for raster chunks touched by a shard."""
    xmin, ymin, xmax, ymax = bounds
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    return float(width * height)


def _centroid(partition: PointPartition) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = partition.bounds
    return 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)


def _best_bisection(
    partitions: Sequence[PointPartition],
    weights: np.ndarray,
    indices: Sequence[int],
    *,
    left_shards: int,
    right_shards: int,
) -> tuple[list[int], list[int]]:
    """Split one region near the requested workload ratio while staying compact."""
    shard_count = left_shards + right_shards
    total_weight = float(np.sum(weights[list(indices)]))
    target_left = total_weight * (left_shards / shard_count)
    parent_cost = max(
        _footprint_cost(_union_bounds(partitions, indices)),
        np.finfo(float).eps,
    )

    candidates: list[tuple[float, float, int, int, list[int]]] = []
    for axis in (0, 1):
        other = 1 - axis
        ordered = sorted(
            indices,
            key=lambda index: (
                _centroid(partitions[index])[axis],
                _centroid(partitions[index])[other],
                index,
            ),
        )
        ordered_weights = weights[ordered]
        cumulative = np.cumsum(ordered_weights, dtype=np.float64)
        min_cut = left_shards
        max_cut = len(ordered) - right_shards
        for cut in range(min_cut, max_cut + 1):
            left_weight = float(cumulative[cut - 1])
            imbalance = abs(left_weight - target_left) / max(total_weight, 1.0)
            left = ordered[:cut]
            right = ordered[cut:]
            compactness = (
                _footprint_cost(_union_bounds(partitions, left))
                + _footprint_cost(_union_bounds(partitions, right))
            ) / parent_cost
            candidates.append((imbalance, compactness, axis, cut, ordered))

    if not candidates:
        raise RuntimeError(
            "Could not bisect spatial partitions while leaving at least one "
            "partition for every requested shard."
        )

    # Workload balance is the primary objective.  Raster-footprint area breaks
    # ties (or effectively equal row-weight solutions), followed by deterministic
    # axis / cut ordering so plans are reproducible across processes and machines.
    _, _, _, cut, ordered = min(
        candidates,
        key=lambda item: (round(item[0], 14), item[1], item[2], item[3]),
    )
    return ordered[:cut], ordered[cut:]


def plan_spatial_point_shards(
    partitions: Sequence[PointPartition],
    shard_count: int,
) -> tuple[SpatialPointShard, ...]:
    """Plan approximately equal-row, spatially compact point shards.

    The planner treats each :class:`PointPartition` as an indivisible spatial
    unit.  It recursively bisects the current region and targets the fraction of
    total row weight required by the number of descendant shards on each side.
    Among equally balanced cuts, it prefers the split with the smaller combined
    bounding-box area, which is a direct proxy for the raster footprint each
    shard can touch.  The resulting groups are deterministic and may be irregular
    along their boundaries, which avoids the severe load imbalance of forcing an
    arbitrary regular grid.

    Parameters
    ----------
    partitions:
        Spatial point partitions with finite bounds and known ``rows``.
    shard_count:
        Number of independent spatial domains to create.  Must be between one
        and the number of input partitions.
    """
    partitions = tuple(partitions)
    shard_count = int(shard_count)
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if not partitions:
        raise ValueError("partitions must not be empty.")
    if shard_count > len(partitions):
        raise ValueError(
            f"shard_count={shard_count} exceeds partition count={len(partitions)}."
        )
    if any(partition.rows is None for partition in partitions):
        raise ValueError("All partitions must provide rows for workload balancing.")

    rows = np.asarray([int(partition.rows or 0) for partition in partitions], dtype=np.float64)
    if np.any(rows < 0):
        raise ValueError("partition rows must be non-negative.")
    # If every partition is empty, fall back to unit weights so the spatial plan
    # is still well defined and every requested shard receives a partition.
    weights = rows if float(np.sum(rows)) > 0 else np.ones(len(partitions), dtype=np.float64)

    def recurse(indices: list[int], requested: int) -> list[list[int]]:
        if requested == 1:
            return [indices]
        left_shards = requested // 2
        right_shards = requested - left_shards
        left, right = _best_bisection(
            partitions,
            weights,
            indices,
            left_shards=left_shards,
            right_shards=right_shards,
        )
        return recurse(left, left_shards) + recurse(right, right_shards)

    groups = recurse(list(range(len(partitions))), shard_count)
    shards = []
    for indices in groups:
        ordered = tuple(sorted(indices))
        shards.append(
            SpatialPointShard(
                partition_indices=ordered,
                rows=int(sum(int(partitions[index].rows or 0) for index in ordered)),
                bounds=_union_bounds(partitions, ordered),
            )
        )

    assigned = [index for shard in shards for index in shard.partition_indices]
    if sorted(assigned) != list(range(len(partitions))):
        raise RuntimeError("Spatial shard planner did not assign every partition exactly once.")
    return tuple(shards)
