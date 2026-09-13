from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from hsa import FeatureSpec
from hsa.compute.dask import recommend_local_worker_count
from hsa.compute.topology import (
    RuntimeTopology,
    _last_level_cache_domains,
    spread_physical_cpus_across_llc,
)
from hsa.rsf.surface_fast import _BLOCK_WORKSPACES, _block_predict


def _surface_case():
    values = np.stack(
        [
            np.arange(64, dtype="float32").reshape(8, 8),
            np.linspace(-1.0, 1.0, 64, dtype="float32").reshape(8, 8),
            np.ones((8, 8), dtype="float32"),
        ]
    )
    env = xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={
            "band": ["a", "b", "c"],
            "y": np.arange(8),
            "x": np.arange(8),
        },
    )
    spec = FeatureSpec(
        linear=["a", "b", "c"],
        quadratic=["a"],
        interactions=[("a", "b")],
        add_const=True,
    )
    model = SimpleNamespace(
        params=pd.Series(
            {
                "const": -0.2,
                "a": 0.04,
                "b": -0.1,
                "c": 0.2,
                "a__sq": 0.01,
                "a__x__b": -0.02,
            }
        )
    )
    scaler = SimpleNamespace(
        mean_=np.array([31.5, 0.0, 1.0]),
        scale_=np.array([20.0, 0.5, 1.0]),
    )
    meta = {"categorical": {}, "columns": list(model.params.index)}
    kwargs = {
        "coefficients": {
            str(name): float(value) for name, value in model.params.items()
        },
        "means": [float(value) for value in scaler.mean_],
        "scales": [float(value) for value in scaler.scale_],
        "spec": spec,
        "meta": meta,
        "dtype": "float32",
        "compute_dtype": "float32",
    }
    return env, kwargs


def test_reused_block_workspace_matches_allocating_path_and_does_not_alias():
    env, kwargs = _surface_case()
    _BLOCK_WORKSPACES.clear_current_thread()

    reused_first = _block_predict(env, reuse_workspace=True, **kwargs)
    first_snapshot = reused_first.values.copy()
    allocating = _block_predict(env, reuse_workspace=False, **kwargs)

    np.testing.assert_allclose(reused_first.values, allocating.values, rtol=0, atol=0)

    changed = env.copy(data=(env.values + 0.5).astype("float32"))
    reused_second = _block_predict(changed, reuse_workspace=True, **kwargs)

    # The second task may reuse scratch arrays, but never the returned eta/output.
    np.testing.assert_array_equal(reused_first.values, first_snapshot)
    assert not np.shares_memory(reused_first.values, reused_second.values)


def test_linear_affine_fold_matches_standardized_formula():
    """Linear-only predictor c may be affine-folded without changing the model."""
    env, kwargs = _surface_case()
    predicted = _block_predict(env, reuse_workspace=True, **kwargs).values

    a = env.sel(band="a").values.astype("float32")
    b = env.sel(band="b").values.astype("float32")
    c = env.sel(band="c").values.astype("float32")
    za = (a - np.float32(31.5)) / np.float32(20.0)
    zb = (b - np.float32(0.0)) / np.float32(0.5)
    zc = (c - np.float32(1.0)) / np.float32(1.0)
    eta = (
        np.float32(-0.2)
        + np.float32(0.04) * za
        + np.float32(-0.1) * zb
        + np.float32(0.2) * zc
        + np.float32(0.01) * za * za
        + np.float32(-0.02) * za * zb
    )
    expected = np.exp(eta).astype("float32")

    # Algebraic folding changes floating-point operation order slightly, so use
    # the same tolerance as the fused-vs-reference surface correctness tests.
    np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-6)


def _ryzen_like_topology() -> RuntimeTopology:
    return RuntimeTopology(
        hostname="test",
        pid=1,
        cpu_affinity=tuple(range(24)),
        affinity_cpu_count=24,
        physical_core_count=12,
        socket_ids=(0,),
        numa_node_ids=(0,),
        cpu_model="test",
        memory_total_gib=128.0,
        slurm_procid=None,
        slurm_localid=None,
        slurm_nodeid=None,
        physical_cpu_representatives=tuple(range(12)),
        llc_level=3,
        llc_size_bytes=16 * 1024**2,
        llc_domains=(
            (0, 1, 2, 12, 13, 14),
            (3, 4, 5, 15, 16, 17),
            (6, 7, 8, 18, 19, 20),
            (9, 10, 11, 21, 22, 23),
        ),
    )


def test_spread_physical_cpus_round_robins_across_llc_domains():
    topology = _ryzen_like_topology()
    assert spread_physical_cpus_across_llc(topology, 4) == (0, 3, 6, 9)
    assert spread_physical_cpus_across_llc(topology, 8) == (
        0,
        3,
        6,
        9,
        1,
        4,
        7,
        10,
    )


def test_last_level_cache_domains_are_read_from_sysfs(tmp_path):
    root = tmp_path
    domains = {
        0: "0-1",
        1: "0-1",
        2: "2-3",
        3: "2-3",
    }
    for cpu, shared in domains.items():
        index = root / f"cpu{cpu}" / "cache" / "index3"
        index.mkdir(parents=True)
        (index / "level").write_text("3\n", encoding="utf-8")
        (index / "type").write_text("Unified\n", encoding="utf-8")
        (index / "shared_cpu_list").write_text(shared + "\n", encoding="utf-8")
        (index / "size").write_text("16M\n", encoding="utf-8")

    level, size, groups = _last_level_cache_domains(
        (0, 1, 2, 3),
        sysfs_root=root,
    )
    assert level == 3
    assert size == 16 * 1024**2
    assert groups == ((0, 1), (2, 3))


def test_local_worker_budget_tracks_physical_cores(monkeypatch):
    monkeypatch.setattr(
        "hsa.compute.topology.discover_runtime_topology",
        lambda: SimpleNamespace(physical_core_count=12),
    )
    assert recommend_local_worker_count(threads_per_worker=1) == 12
    assert recommend_local_worker_count(threads_per_worker=2) == 6
    assert recommend_local_worker_count(threads_per_worker=3) == 4
    assert recommend_local_worker_count(threads_per_worker=4) == 3
