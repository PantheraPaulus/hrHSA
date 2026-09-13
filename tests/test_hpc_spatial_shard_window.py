from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from hsa.compute import PointPartition


def _load_spatial_shard_module():
    script_dir = (
        Path(__file__).resolve().parents[1]
        / "benchmarking"
        / "scripts"
        / "01_python"
    )
    path = script_dir / "profile_point_partitioned_spatial_shard.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("spatial_shard_window_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_dir))


def _env(*, descending_y: bool = True):
    x = np.arange(10, dtype=np.float64)
    y = np.arange(9, -1, -1, dtype=np.float64) if descending_y else x.copy()
    return xr.DataArray(
        np.zeros((1, len(y), len(x)), dtype=np.float32),
        dims=("band", "y", "x"),
        coords={"band": ["b0"], "y": y, "x": x},
    )


def test_covering_index_bounds_bracket_outer_half_pixel_points():
    module = _load_spatial_shard_module()
    coords = np.arange(10, dtype=np.float64)

    # Nearest-centre selection would choose 3 and 5 and therefore reject 2.6
    # under require_inside=True.  The covering bounds must include centres 2 and 6.
    assert module._covering_index_bounds(coords, 2.6, 5.4) == (2, 6)
    assert module._covering_index_bounds(coords[::-1], 2.6, 5.4) == (3, 7)


def test_aligned_raster_window_contains_full_point_bounds_for_descending_y():
    module = _load_spatial_shard_module()
    env = _env(descending_y=True)
    partition = PointPartition(
        path=Path("unused.parquet"),
        bounds=(2.6, 2.6, 5.4, 5.4),
        rows=1,
        crs=None,
    )

    cropped, window = module._aligned_raster_window(
        env,
        [partition],
        spatial_chunk=1,
    )

    assert float(cropped.x.min()) <= 2.6 <= float(cropped.x.max())
    assert float(cropped.x.min()) <= 5.4 <= float(cropped.x.max())
    assert float(cropped.y.min()) <= 2.6 <= float(cropped.y.max())
    assert float(cropped.y.min()) <= 5.4 <= float(cropped.y.max())
    assert window["x_start"] == 2
    assert window["x_stop"] == 7
    assert window["y_start"] == 3
    assert window["y_stop"] == 8


def test_aligned_raster_window_stays_chunk_aligned_while_expanding_outward():
    module = _load_spatial_shard_module()
    env = _env(descending_y=False)
    partition = PointPartition(
        path=Path("unused.parquet"),
        bounds=(2.6, 2.6, 5.4, 5.4),
        rows=1,
        crs=None,
    )

    cropped, window = module._aligned_raster_window(
        env,
        [partition],
        spatial_chunk=4,
    )

    assert window["x_start"] == 0
    assert window["x_stop"] == 8
    assert window["y_start"] == 0
    assert window["y_stop"] == 8
    assert float(cropped.x.min()) <= 2.6
    assert float(cropped.x.max()) >= 5.4
    assert float(cropped.y.min()) <= 2.6
    assert float(cropped.y.max()) >= 5.4
