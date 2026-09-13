from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hsa.compute import (
    COOLMUC4,
    coolmuc4_execution,
    get_site_profile,
    open_raster_stack_zarr,
    plan_zarr_shards,
    write_raster_stack_zarr,
    zarr_storage_units,
)


def _env() -> xr.DataArray:
    return xr.DataArray(
        np.arange(2 * 16 * 16, dtype="float32").reshape(2, 16, 16),
        dims=("band", "y", "x"),
        coords={"band": ["a", "b"], "y": np.arange(16), "x": np.arange(16)},
        name="env",
    )


def test_coolmuc4_profile_geometry_and_memory():
    assert COOLMUC4.physical_cores_per_node == 112
    assert COOLMUC4.sockets_per_node == 2
    assert COOLMUC4.cores_per_socket == 56
    assert COOLMUC4.memory_gib_per_node == 488
    assert COOLMUC4.min_parallel_cores_per_node == 17
    assert COOLMUC4.max_parallel_nodes == 4
    assert COOLMUC4.policy_memory_ceiling_gib == 488.0

    geometries = COOLMUC4.node_geometry_candidates()
    assert geometries
    assert all(item.total_threads == 112 for item in geometries)
    assert {item.label for item in geometries} >= {"112x1", "28x4", "14x8", "8x14", "2x56"}
    assert 1.0 < COOLMUC4.memory_per_worker_gib(112) < 2.0
    assert COOLMUC4.recommend_geometry(workload="surface_prediction").label == "8x14"


def test_coolmuc4_auto_detection(monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cm4")
    assert get_site_profile("auto") is COOLMUC4


def test_coolmuc4_execution_factory():
    one_node = coolmuc4_execution()
    assert one_node.backend == "slurm"
    assert one_node.n_workers == 8
    assert one_node.threads_per_worker == 14
    assert one_node.local_directory == "$TMPDIR"
    assert one_node.slurm_options["queue"] == "cm4_tiny"
    assert one_node.slurm_options["cores"] == 112
    assert one_node.slurm_options["processes"] == 8
    directives = one_node.slurm_options["job_extra_directives"]
    assert "--clusters=cm4" in directives
    assert "--qos=cm4_tiny" in directives
    assert "--hint=nomultithread" in directives
    assert "--get-user-env" in directives
    assert "--export=NONE" in directives
    assert one_node.slurm_options["memory"] == "224GB"

    legacy_geometry = coolmuc4_execution(workers_per_node=28, threads_per_worker=4)
    assert legacy_geometry.n_workers == 28
    assert legacy_geometry.threads_per_worker == 4

    # LRZ allows an explicit shared-node memory request above the default
    # core-proportional amount, up to the 488 GiB node limit. hrHSA's 224 GiB
    # value is a conservative benchmark default, not a site-policy ceiling.
    larger_memory = coolmuc4_execution(memory_per_node_gib=300)
    assert larger_memory.slurm_options["memory"] == "300GB"


def test_coolmuc4_execution_rejects_illegal_or_unrepresentable_jobs():
    with pytest.raises(ValueError, match="at least 17"):
        coolmuc4_execution(workers_per_node=8, threads_per_worker=1)

    with pytest.raises(ValueError, match="supports cm4_tiny only"):
        coolmuc4_execution(partition="cm4_std")

    with pytest.raises(ValueError, match="supports one cm4_tiny node only"):
        coolmuc4_execution(nodes=2)

    with pytest.raises(ValueError, match="supports one cm4_tiny node only"):
        coolmuc4_execution(nodes=5)

    with pytest.raises(ValueError, match="current LRZ policy"):
        coolmuc4_execution(memory_per_node_gib=500)


def test_zarr_shard_planner_reduces_storage_units():
    env = _env()
    chunks = {"band": -1, "y": 4, "x": 4}
    shards = plan_zarr_shards(env, chunks, target_shard_mb=1)
    assert shards["band"] == 2
    assert shards["x"] % 4 == 0
    assert shards["y"] % 4 == 0
    assert zarr_storage_units(env, shards) <= zarr_storage_units(env, chunks)


def test_sharded_zarr_roundtrip(tmp_path):
    pytest.importorskip("zarr")
    env = _env()
    path = tmp_path / "sharded.zarr"
    write_raster_stack_zarr(
        env,
        path,
        chunks={"band": -1, "y": 4, "x": 4},
        shards={"band": 2, "y": 8, "x": 8},
        zarr_format=3,
    )
    reopened = open_raster_stack_zarr(path, name="env", chunks=None)
    np.testing.assert_array_equal(reopened.values, env.values)
