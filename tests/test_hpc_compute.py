from __future__ import annotations

import threading
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers the xarray .rio accessor
import xarray as xr
from shapely.geometry import Point

from hsa import FeatureSpec
from hsa.compute import (
    ExecutionConfig,
    PreparedDataset,
    compare_sampling_engines,
    iter_sample_raster_stack_chunked,
    recommend_worker_count,
    sample_raster_stack_batched,
    sample_raster_stack_chunked,
    scaling_table,
    spatial_task_count,
    task_density,
)
from hsa.compute.raster import (
    _chunk_number,
    _extract_numpy_block_flat,
    _nearest_indices,
)
from hsa.rsf import FrequentistRSF, predict_rsf_surface, predict_rsf_surface_chunked


def _small_env() -> xr.DataArray:
    x = np.arange(0.0, 100.0, 10.0)
    y = np.arange(90.0, -10.0, -10.0)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    values = np.stack(
        [
            xx + yy,
            0.1 * xx - 0.2 * yy,
        ],
        axis=0,
    ).astype("float32")
    env = xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={"band": ["a", "b"], "y": y, "x": x},
    )
    return env.rio.write_crs("EPSG:3857")


def _small_samples() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "used": [True, False, True, False],
            "Individual_ID": ["A", "A", "B", "B"],
        },
        geometry=[
            Point(5, 85),
            Point(25, 45),
            Point(75, 15),
            Point(91, 2),
        ],
        crs="EPSG:3857",
    )


def test_regular_grid_nearest_indices_preserve_midpoint_semantics():
    ascending = np.array([0.0, 10.0, 20.0, 30.0])
    values = np.array([-2.0, 0.0, 4.9, 5.0, 5.1, 15.0, 25.0, 40.0])
    expected_ascending = np.array([0, 0, 0, 1, 1, 2, 3, 3])
    np.testing.assert_array_equal(
        _nearest_indices(ascending, values),
        expected_ascending,
    )

    descending = ascending[::-1]
    expected_descending = np.array([3, 3, 3, 2, 2, 1, 0, 0])
    np.testing.assert_array_equal(
        _nearest_indices(descending, values),
        expected_descending,
    )


def test_irregular_grid_nearest_indices_keep_general_fallback():
    coordinates = np.array([0.0, 9.0, 20.0, 33.0])
    values = np.array([4.5, 14.5, 26.5])
    np.testing.assert_array_equal(
        _nearest_indices(coordinates, values),
        np.array([1, 2, 3]),
    )


def test_regular_chunk_number_fast_path_matches_general_mapping():
    lengths = (1024, 1024, 1024, 500)
    indices = np.array([0, 1023, 1024, 2047, 2048, 3071, 3072, 3571])
    boundaries = np.cumsum(np.asarray(lengths, dtype=np.int64))
    expected = np.searchsorted(boundaries, indices, side="right")
    np.testing.assert_array_equal(_chunk_number(indices, lengths), expected)


def test_irregular_chunk_number_keeps_general_fallback():
    lengths = (500, 700, 1000, 250)
    indices = np.arange(sum(lengths), dtype=np.int64)
    boundaries = np.cumsum(np.asarray(lengths, dtype=np.int64))
    expected = np.searchsorted(boundaries, indices, side="right")
    np.testing.assert_array_equal(_chunk_number(indices, lengths), expected)


def test_flat_block_indices_match_row_col_gather():
    block = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 3])
    flat_indices = (rows * block.shape[2] + cols).astype(np.uint32)

    expected = block[:, rows, cols]
    actual = _extract_numpy_block_flat(block, flat_indices)
    np.testing.assert_array_equal(actual, expected)


def test_chunked_sampler_matches_reference():
    env = _small_env()
    samples = _small_samples()

    comparison = compare_sampling_engines(samples, env, bands=["a", "b"])
    assert comparison["equivalent"]
    assert comparison["max_abs_difference"] < 1e-6

    sampled = sample_raster_stack_chunked(
        samples,
        env,
        bands=["a"],
        id_cols="Individual_ID",
        chunks={"band": -1, "y": 3, "x": 4},
    )
    assert list(sampled["Individual_ID"]) == ["A", "A", "B", "B"]
    assert sampled["used"].tolist() == [True, False, True, False]


def test_chunked_batch_iterator_matches_single_chunked_call():
    env = _small_env()
    samples = _small_samples()
    kwargs = {
        "bands": ["a", "b"],
        "id_cols": "Individual_ID",
        "chunks": {"band": -1, "y": 3, "x": 4},
    }

    batches = list(
        iter_sample_raster_stack_chunked(
            samples,
            env,
            batch_size=2,
            batches_in_flight=2,
            **kwargs,
        )
    )
    assert [len(batch) for batch in batches] == [2, 2]

    combined = pd.concat(batches, ignore_index=True)
    direct = sample_raster_stack_chunked(samples, env, **kwargs).reset_index(drop=True)
    pd.testing.assert_frame_equal(combined, direct)

    gathered = sample_raster_stack_batched(
        samples,
        env,
        batch_size=2,
        batches_in_flight=2,
        **kwargs,
    )
    pd.testing.assert_frame_equal(gathered, direct)


def test_concurrent_batch_iterator_is_bounded_and_ordered(monkeypatch):
    samples = _small_samples()
    env = _small_env()
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_sampler(batch, env, *, client=None, **kwargs):
        nonlocal active, max_active
        start = int(batch.index[0])
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=2.0)
            return pd.DataFrame({"batch_start": [start]})
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        "hsa.compute.raster.sample_raster_stack_chunked",
        fake_sampler,
    )

    result = list(
        iter_sample_raster_stack_chunked(
            samples,
            env,
            batch_size=2,
            batches_in_flight=2,
        )
    )
    assert [int(frame.iloc[0]["batch_start"]) for frame in result] == [0, 2]
    assert max_active == 2


def test_concurrent_batch_iterator_propagates_sampling_errors(monkeypatch):
    samples = _small_samples()
    env = _small_env()

    def fake_sampler(batch, env, *, client=None, **kwargs):
        if int(batch.index[0]) == 2:
            raise RuntimeError("synthetic batch failure")
        return pd.DataFrame({"ok": [True]})

    monkeypatch.setattr(
        "hsa.compute.raster.sample_raster_stack_chunked",
        fake_sampler,
    )

    with pytest.raises(RuntimeError, match="synthetic batch failure"):
        list(
            iter_sample_raster_stack_chunked(
                samples,
                env,
                batch_size=2,
                batches_in_flight=2,
            )
        )


def test_execution_config_accepts_explicit_point_batch_policy():
    config = ExecutionConfig(
        point_batch_rows=16_000_000,
        point_batches_in_flight=2,
    )
    assert config.point_batch_rows == 16_000_000
    assert config.point_batches_in_flight == 2

    with pytest.raises(ValueError, match="point_batch_rows"):
        ExecutionConfig(point_batch_rows=0)
    with pytest.raises(ValueError, match="point_batches_in_flight"):
        ExecutionConfig(point_batches_in_flight=0)


def test_chunked_surface_matches_reference():
    env = _small_env()
    spec = FeatureSpec(
        linear=["a", "b"],
        quadratic=["a"],
        interactions=[("a", "b")],
        add_const=True,
    )
    model = SimpleNamespace(
        params=pd.Series(
            {
                "const": -0.2,
                "a": 0.7,
                "b": -0.3,
                "a__sq": 0.15,
                "a__x__b": 0.08,
            }
        )
    )
    scaler = SimpleNamespace(
        mean_=np.array([80.0, -4.0]),
        scale_=np.array([35.0, 8.0]),
    )
    meta = {"categorical": {}, "columns": list(model.params.index)}

    reference = predict_rsf_surface(env, model, scaler, spec, meta)
    accelerated = predict_rsf_surface_chunked(
        env,
        model,
        scaler,
        spec,
        meta,
        chunks={"band": -1, "y": 4, "x": 4},
    )
    np.testing.assert_allclose(
        reference.values,
        accelerated.values,
        rtol=1e-5,
        atol=1e-6,
    )


def test_scaling_table_calculates_efficiency():
    results = pd.DataFrame(
        {
            "workers": [1, 2, 4],
            "wall_seconds": [100.0, 55.0, 30.0],
        }
    )
    scaled = scaling_table(results)
    assert scaled.loc[0, "speedup"] == pytest.approx(1.0)
    assert scaled.loc[1, "speedup"] == pytest.approx(100.0 / 55.0)
    assert scaled.loc[2, "parallel_efficiency"] == pytest.approx(
        (100.0 / 30.0) / 4.0
    )


def test_task_density_helpers_cap_oversized_worker_pool():
    env = _small_env()
    chunks = {"band": -1, "y": 5, "x": 5}
    tasks = spatial_task_count(env, chunks)
    assert tasks == 4
    assert task_density(tasks, workers=2) == pytest.approx(2.0)
    assert recommend_worker_count(tasks, requested_workers=8) == 1

    large_tasks = 1024
    assert recommend_worker_count(large_tasks, requested_workers=256) == 128
    assert recommend_worker_count(large_tasks, requested_workers=64) == 64


def test_prepared_dataset_roundtrip_and_fit(tmp_path):
    pytest.importorskip("pyarrow")
    env = _small_env()
    rows = []
    y_pattern = [20, 35, 55, 75, 65, 45, 25, 50]
    for individual, offset in [("A", 0.0), ("B", 25.0)]:
        for index in range(8):
            rows.append(
                {
                    "Individual_ID": individual,
                    "Timestamp": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(hours=index),
                    "geometry": Point(
                        10 + offset + index * 2,
                        y_pattern[index],
                    ),
                }
            )
    reloc = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")
    spec = FeatureSpec(linear=["a", "b"], add_const=True)
    analysis = FrequentistRSF(reloc, env, spec=spec)

    prepared = analysis.prepare(
        tmp_path / "prepared",
        sampling_factor=2,
        engine="reference",
    )
    reopened = PreparedDataset.open(prepared.root)
    assert set(reopened.individuals) == {"A", "B"}
    assert reopened.manifest["n_rows"].sum() > len(reloc)

    fit = analysis.fit(prepared=reopened)
    assert {"a", "b"}.issubset(fit.model.params.index)