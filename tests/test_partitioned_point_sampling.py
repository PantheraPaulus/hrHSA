from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers xarray .rio
import xarray as xr
from shapely.geometry import Point

from hsa.compute import (
    PointPartition,
    iter_sample_raster_stack_partitioned,
    sample_raster_stack_chunked,
    sample_raster_stack_partitioned,
)


def _env() -> xr.DataArray:
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


def _frames() -> list[pd.DataFrame]:
    return [
        pd.DataFrame(
            {
                "x_coord": [5.0, 25.0],
                "y_coord": [85.0, 45.0],
                "used": [True, False],
                "Individual_ID": ["A", "A"],
            }
        ),
        pd.DataFrame(
            {
                # Keep these coordinates inside the raster-coordinate extent.
                # The raster x centres run from 0 to 90, so x=91 would correctly
                # fail when require_inside=True in both sampling engines.
                "x_coord": [75.0, 89.0],
                "y_coord": [15.0, 2.0],
                "used": [True, False],
                "Individual_ID": ["B", "B"],
            }
        ),
    ]


def _write_partitions(tmp_path) -> tuple[list[PointPartition], list[pd.DataFrame]]:
    frames = _frames()
    partitions = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"part-{index:06d}.parquet"
        frame.to_parquet(path, index=False)
        partitions.append(
            PointPartition(
                path=path,
                bounds=(
                    float(frame["x_coord"].min()),
                    float(frame["y_coord"].min()),
                    float(frame["x_coord"].max()),
                    float(frame["y_coord"].max()),
                ),
                rows=len(frame),
                crs="EPSG:3857",
            )
        )
    return partitions, frames


def _reference(frame: pd.DataFrame, env: xr.DataArray) -> pd.DataFrame:
    samples = gpd.GeoDataFrame(
        frame[["used", "Individual_ID"]].copy(),
        geometry=[Point(x, y) for x, y in zip(frame["x_coord"], frame["y_coord"])],
        crs="EPSG:3857",
    )
    return sample_raster_stack_chunked(
        samples,
        env,
        bands=["a", "b"],
        id_cols="Individual_ID",
        chunks={"band": -1, "y": 3, "x": 4},
        require_inside=True,
    ).reset_index(drop=True)


def test_partitioned_sampler_matches_in_memory_chunked_sampler(tmp_path):
    distributed = pytest.importorskip("distributed")
    env = _env()
    partitions, frames = _write_partitions(tmp_path)
    cluster = distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        completed = list(
            iter_sample_raster_stack_partitioned(
                partitions,
                env,
                x_col="x_coord",
                y_col="y_coord",
                bands=["a", "b"],
                preserve_cols="used",
                id_cols="Individual_ID",
                chunks={"band": -1, "y": 3, "x": 4},
                graph_partitions=2,
                client=client,
                require_inside=True,
            )
        )
    finally:
        client.close()
        cluster.close()

    assert sorted(item.partition_index for item in completed) == [0, 1]
    by_index = {item.partition_index: item.frame.reset_index(drop=True) for item in completed}
    for index, frame in enumerate(frames):
        pd.testing.assert_frame_equal(by_index[index], _reference(frame, env))


def test_partitioned_collector_restores_input_partition_order(tmp_path):
    distributed = pytest.importorskip("distributed")
    env = _env()
    partitions, frames = _write_partitions(tmp_path)
    cluster = distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        sampled = sample_raster_stack_partitioned(
            partitions,
            env,
            x_col="x_coord",
            y_col="y_coord",
            bands=["a", "b"],
            preserve_cols="used",
            id_cols="Individual_ID",
            chunks={"band": -1, "y": 3, "x": 4},
            graph_partitions=2,
            client=client,
            require_inside=True,
        )
    finally:
        client.close()
        cluster.close()

    expected = pd.concat([_reference(frame, env) for frame in frames], ignore_index=True)
    pd.testing.assert_frame_equal(sampled, expected)


def test_partitioned_require_inside_matches_chunked_coordinate_extent(tmp_path):
    distributed = pytest.importorskip("distributed")
    env = _env()
    frame = pd.DataFrame(
        {
            "x_coord": [91.0],
            "y_coord": [2.0],
            "used": [False],
            "Individual_ID": ["outside"],
        }
    )
    path = tmp_path / "outside.parquet"
    frame.to_parquet(path, index=False)
    partition = PointPartition(
        path=path,
        bounds=(91.0, 2.0, 91.0, 2.0),
        rows=1,
        crs="EPSG:3857",
    )

    samples = gpd.GeoDataFrame(
        frame[["used", "Individual_ID"]].copy(),
        geometry=[Point(91.0, 2.0)],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="outside the environmental raster extent"):
        sample_raster_stack_chunked(samples, env, require_inside=True)

    cluster = distributed.LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        with pytest.raises(ValueError, match="outside the raster extent"):
            list(
                iter_sample_raster_stack_partitioned(
                    [partition],
                    env,
                    x_col="x_coord",
                    y_col="y_coord",
                    preserve_cols="used",
                    id_cols="Individual_ID",
                    client=client,
                    require_inside=True,
                )
            )
    finally:
        client.close()
        cluster.close()


def test_partitioned_sampler_requires_matching_crs_and_client(tmp_path):
    env = _env()
    partitions, _ = _write_partitions(tmp_path)

    with pytest.raises(ValueError, match="active Dask client"):
        list(iter_sample_raster_stack_partitioned(partitions, env))

    mismatched = [
        PointPartition(
            path=partitions[0].path,
            bounds=partitions[0].bounds,
            rows=partitions[0].rows,
            crs="EPSG:4326",
        )
    ]
    distributed = pytest.importorskip("distributed")
    cluster = distributed.LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        with pytest.raises(ValueError, match="does not match raster CRS"):
            list(iter_sample_raster_stack_partitioned(mismatched, env, client=client))
    finally:
        client.close()
        cluster.close()


def test_point_partition_validates_bounds_and_rows(tmp_path):
    path = tmp_path / "part.parquet"
    pd.DataFrame({"x": [], "y": []}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="xmin<=xmax"):
        PointPartition(path, (2.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="rows"):
        PointPartition(path, (0.0, 0.0, 1.0, 1.0), rows=-1)
