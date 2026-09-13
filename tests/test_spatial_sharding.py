from __future__ import annotations

from pathlib import Path

import pytest

from hsa.compute import PointPartition, plan_spatial_point_shards


def _grid_partitions(rows: int, cols: int, *, weights=None):
    if weights is None:
        weights = [1] * (rows * cols)
    partitions = []
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            partitions.append(
                PointPartition(
                    path=Path(f"part-{index:06d}.parquet"),
                    bounds=(float(col), float(row), float(col + 1), float(row + 1)),
                    rows=int(weights[index]),
                    crs="EPSG:3857",
                )
            )
    return partitions


def test_uniform_25x40_grid_balances_28_shards_to_35_or_36_tiles():
    partitions = _grid_partitions(25, 40, weights=[1_000_000] * 1000)

    shards = plan_spatial_point_shards(partitions, 28)

    assert len(shards) == 28
    assigned = [index for shard in shards for index in shard.partition_indices]
    assert sorted(assigned) == list(range(1000))
    assert len(assigned) == len(set(assigned))
    assert {shard.partition_count for shard in shards} == {35, 36}
    assert {shard.rows for shard in shards} == {35_000_000, 36_000_000}

    # The exact-load solution should remain spatially compact rather than
    # degenerating into long strips across the full 25x40 domain.
    shard_areas = [
        (shard.bounds[2] - shard.bounds[0]) * (shard.bounds[3] - shard.bounds[1])
        for shard in shards
    ]
    assert max(shard_areas) <= 42.0


def test_uniform_25x40_grid_balances_56_shards_to_17_or_18_tiles():
    partitions = _grid_partitions(25, 40, weights=[1_000_000] * 1000)

    shards = plan_spatial_point_shards(partitions, 56)

    assert len(shards) == 56
    assigned = [index for shard in shards for index in shard.partition_indices]
    assert sorted(assigned) == list(range(1000))
    assert len(assigned) == len(set(assigned))
    assert {shard.partition_count for shard in shards} == {17, 18}
    assert {shard.rows for shard in shards} == {17_000_000, 18_000_000}


def test_weighted_plan_balances_uneven_partition_rows():
    weights = [1, 1, 8, 2, 3, 7, 1, 4, 2, 6, 1, 5, 3, 2, 7, 1]
    partitions = _grid_partitions(4, 4, weights=weights)

    shards = plan_spatial_point_shards(partitions, 4)

    assert len(shards) == 4
    assert sum(shard.rows for shard in shards) == sum(weights)
    assigned = [index for shard in shards for index in shard.partition_indices]
    assert sorted(assigned) == list(range(16))

    ideal = sum(weights) / 4
    max_item = max(weights)
    assert max(abs(shard.rows - ideal) for shard in shards) <= max_item


def test_spatial_plan_is_deterministic():
    partitions = _grid_partitions(5, 7)
    first = plan_spatial_point_shards(partitions, 6)
    second = plan_spatial_point_shards(partitions, 6)
    assert first == second


def test_spatial_plan_requires_known_rows():
    partitions = [
        PointPartition(
            path="part.parquet",
            bounds=(0.0, 0.0, 1.0, 1.0),
            rows=None,
            crs="EPSG:3857",
        )
    ]
    with pytest.raises(ValueError, match="must provide rows"):
        plan_spatial_point_shards(partitions, 1)


def test_spatial_plan_rejects_more_shards_than_partitions():
    partitions = _grid_partitions(2, 2)
    with pytest.raises(ValueError, match="exceeds partition count"):
        plan_spatial_point_shards(partitions, 5)
