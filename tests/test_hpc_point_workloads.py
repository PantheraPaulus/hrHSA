from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from hsa.compute.point_workloads import (
    point_workload_for_count,
    point_workloads,
    weak_point_workloads,
)


def test_point_workload_balances_bounded_spatial_partitions():
    workload = point_workload_for_count(
        2_500_001,
        partition_size=1_000_000,
    )
    assert workload.partitions == 3
    assert workload.base_points_per_partition == 833_333
    assert workload.extra_partitions == 2
    assert [workload.points_in_partition(i) for i in range(3)] == [
        833_334,
        833_334,
        833_333,
    ]
    assert [workload.start_point_id(i) for i in range(3)] == [
        0,
        833_334,
        1_666_668,
    ]
    assert workload.max_points_per_partition == 833_334
    assert workload.max_points_per_partition <= workload.target_partition_size
    assert sum(workload.points_in_partition(i) for i in range(3)) == 2_500_001


def test_point_workloads_require_unique_increasing_targets():
    with pytest.raises(ValueError):
        point_workloads([1_000, 1_000])
    with pytest.raises(ValueError):
        point_workloads([2_000, 1_000])


def test_weak_point_workloads_keep_points_per_resource_constant():
    resolved = weak_point_workloads(
        [1, 2, 4],
        points_per_resource=2_500_000,
        partition_size=1_000_000,
    )
    assert [workload.target_points for _, workload in resolved] == [
        2_500_000,
        5_000_000,
        10_000_000,
    ]
    assert [
        workload.target_points / resource
        for resource, workload in resolved
    ] == [2_500_000, 2_500_000, 2_500_000]


def _load_prepare_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarking"
        / "scripts"
        / "01_python"
        / "prepare_point_family.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_point_family_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partition_generation_is_deterministic_by_partition():
    module = _load_prepare_module()
    kwargs = {
        "start": 0,
        "count": 20,
        "seed": 42,
        "target_points": 40,
        "partition_index": 0,
        "tile_rows": 1,
        "tile_cols": 2,
    }
    first = module._partition_frame(**kwargs)
    second = module._partition_frame(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert first["point_id"].tolist() == list(range(20))
    assert first["u"].between(0.0, 0.5, inclusive="left").all()
    assert first["v"].between(0.0, 1.0, inclusive="left").all()


def test_different_partition_indices_occupy_different_spatial_tiles():
    module = _load_prepare_module()
    a = module._partition_frame(
        start=0,
        count=20,
        seed=42,
        target_points=40,
        partition_index=0,
        tile_rows=1,
        tile_cols=2,
    )
    b = module._partition_frame(
        start=20,
        count=20,
        seed=42,
        target_points=40,
        partition_index=1,
        tile_rows=1,
        tile_cols=2,
    )

    assert a["u"].between(0.0, 0.5, inclusive="left").all()
    assert b["u"].between(0.5, 1.0, inclusive="left").all()
    assert not a["u"].equals(b["u"])
    assert b["point_id"].iloc[0] == 20
