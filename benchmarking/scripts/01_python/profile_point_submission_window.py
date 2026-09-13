"""A/B benchmark bounded distributed submission windows for point sampling.

The production sampler currently scatters every chunk-local index payload for one
point batch before submitting any extraction tasks. This calibration companion
keeps all routing/indexing/output logic identical but, for the experimental path,
scatters and submits chunk tasks in bounded windows. Earlier windows can therefore
execute while later payloads are still being scattered.

A submission window of 0 selects the unmodified production sampler. Positive
values select the experimental windowed path. No package behavior is changed by
this script.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil

import run_point_scaling as point_base
from hsa.compute.planning import plan_raster_chunks
from hsa.compute.raster import (
    _chunk_number,
    _chunk_starts,
    _dimension_chunk_lengths,
    _extract_numpy_block_flat,
    _group_positions_by_chunk,
    _nearest_indices,
    _normalise_id_cols,
)
from hsa.compute.workloads import parse_geometry
from hsa.sampling import _as_raster_dataarray
from profile_point_mechanisms import _run_once
from run_point_scaling import (
    _load_inputs,
    _point_frame_to_gdf,
    _point_workload_entry,
)
from run_surface_scaling import _local_configuration, _window


_ORIGINAL_SAMPLER = point_base.sample_raster_stack_chunked
_SUBMISSION_WINDOW = 48


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _windowed_sample_raster_stack_chunked(
    samples: gpd.GeoDataFrame,
    env,
    bands: Sequence[str] | None = None,
    *,
    dtype: str | np.dtype = "float32",
    id_cols: str | Sequence[str] | None = None,
    chunks: dict[str, int] | None = None,
    target_chunk_mb: int = 256,
    align_storage: bool = True,
    client=None,
    require_inside: bool = False,
) -> pd.DataFrame:
    """Experimental sampler differing only in distributed submission policy."""
    if not isinstance(samples, gpd.GeoDataFrame):
        raise TypeError("samples must be a geopandas.GeoDataFrame.")
    if samples.crs is None:
        raise ValueError("samples.crs is None; set a CRS before raster sampling.")

    raster = _as_raster_dataarray(env)
    if bands is not None:
        raster = raster.sel(band=list(bands))
    raster = raster.transpose("band", "y", "x")

    try:
        env_crs = raster.rio.crs
    except Exception:
        env_crs = None

    transformed = samples
    if env_crs is not None and samples.crs != env_crs:
        transformed = samples.to_crs(env_crs)

    x_coordinates = np.asarray(raster["x"].values)
    y_coordinates = np.asarray(raster["y"].values)
    point_x = transformed.geometry.x.to_numpy(dtype=float)
    point_y = transformed.geometry.y.to_numpy(dtype=float)

    if require_inside and len(samples):
        outside = (
            (point_x < float(np.min(x_coordinates)))
            | (point_x > float(np.max(x_coordinates)))
            | (point_y < float(np.min(y_coordinates)))
            | (point_y > float(np.max(y_coordinates)))
        )
        if np.any(outside):
            raise ValueError(
                f"{int(outside.sum()):,} points fall outside the environmental raster extent."
            )

    row_index = _nearest_indices(y_coordinates, point_y)
    col_index = _nearest_indices(x_coordinates, point_x)

    if chunks is None:
        chunks = plan_raster_chunks(
            raster,
            workload="point_sampling",
            target_chunk_mb=target_chunk_mb,
            align_storage=align_storage,
            selected_band_count=int(raster.sizes["band"]),
        )
    chunks = dict(chunks)
    chunks["band"] = -1

    data_chunks = getattr(raster.data, "chunks", None)
    if data_chunks is not None:
        raster = raster.chunk(chunks)

    y_nominal = int(chunks.get("y", raster.sizes["y"]))
    x_nominal = int(chunks.get("x", raster.sizes["x"]))
    y_lengths = _dimension_chunk_lengths(raster, "y", nominal=y_nominal)
    x_lengths = _dimension_chunk_lengths(raster, "x", nominal=x_nominal)
    y_starts = _chunk_starts(y_lengths)
    x_starts = _chunk_starts(x_lengths)

    chunk_y = _chunk_number(row_index, y_lengths)
    chunk_x = _chunk_number(col_index, x_lengths)
    grouped_positions = _group_positions_by_chunk(
        chunk_y,
        chunk_x,
        n_x_chunks=len(x_lengths),
    )

    n_bands = int(raster.sizes["band"])
    values = np.empty((n_bands, len(samples)), dtype=np.dtype(dtype))

    delayed_tasks = []
    delayed_positions: list[np.ndarray] = []
    distributed_specs: list[tuple[Any, np.ndarray]] = []
    dask_backed = getattr(raster.data, "chunks", None) is not None

    for cy, cx, positions in grouped_positions:
        y_start = int(y_starts[cy])
        x_start = int(x_starts[cx])
        y_stop = y_start + int(y_lengths[cy])
        x_stop = x_start + int(x_lengths[cx])
        block_width = int(x_lengths[cx])

        local_rows = row_index[positions] - y_start
        local_cols = col_index[positions] - x_start
        max_flat_index = int(y_lengths[cy]) * block_width - 1
        index_dtype = (
            np.uint32
            if max_flat_index <= np.iinfo(np.uint32).max
            else np.uint64
        )
        flat_indices = (
            local_rows * block_width + local_cols
        ).astype(index_dtype, copy=False)
        block = raster.isel(
            y=slice(y_start, y_stop),
            x=slice(x_start, x_stop),
        ).data

        if dask_backed:
            try:
                import dask
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "A Dask-backed raster requires the optional hsa[dask] or hsa[hpc] dependencies."
                ) from exc
            if client is None:
                delayed_tasks.append(
                    dask.delayed(_extract_numpy_block_flat)(block, flat_indices)
                )
            else:
                distributed_specs.append((block, flat_indices))
            delayed_positions.append(positions)
        else:
            values[:, positions] = _extract_numpy_block_flat(
                block,
                flat_indices,
            ).astype(dtype, copy=False)

    if delayed_tasks:
        import dask

        computed = dask.compute(*delayed_tasks)
        for positions, extracted in zip(delayed_positions, computed):
            values[:, positions] = np.asarray(extracted, dtype=dtype)

    elif distributed_specs:
        import dask

        window = max(1, int(_SUBMISSION_WINDOW))
        index_futures = []
        extraction_futures = []
        try:
            # Unlike the production path, do not wait for every index payload in
            # the full point batch to be scattered before the first extraction
            # graph is submitted. A small, bounded queue is enough to expose the
            # workers while keeping graph and communication payloads compact.
            for start in range(0, len(distributed_specs), window):
                stop = min(start + window, len(distributed_specs))
                spec_window = distributed_specs[start:stop]
                payloads = [flat_indices for _, flat_indices in spec_window]
                window_index_futures = client.scatter(
                    payloads,
                    broadcast=False,
                    hash=False,
                )
                index_futures.extend(window_index_futures)
                tasks = [
                    dask.delayed(_extract_numpy_block_flat)(block, index_future)
                    for (block, _), index_future in zip(
                        spec_window,
                        window_index_futures,
                    )
                ]
                extraction_futures.extend(client.compute(tasks))

            computed = client.gather(extraction_futures)
            for positions, extracted in zip(delayed_positions, computed):
                values[:, positions] = np.asarray(extracted, dtype=dtype)
        finally:
            futures_to_cancel = [*index_futures, *extraction_futures]
            if futures_to_cancel:
                client.cancel(futures_to_cancel)

    sampled_bands = [str(value) for value in raster["band"].values]
    out = pd.DataFrame(values.T, columns=sampled_bands, index=samples.index)
    out["x"] = point_x
    out["y"] = point_y

    for column in ("used", "Timestamp"):
        if column in samples.columns:
            out[column] = samples[column].to_numpy()

    for column in _normalise_id_cols(id_cols):
        if column not in samples.columns:
            raise ValueError(f"id_cols not found in samples.columns: {column!r}")
        out[column] = samples[column].to_numpy()

    return out


def _validate_candidate(*, entry, env, client, chunks, points: int) -> None:
    path = Path(entry["directory"]) / "part-000000.parquet"
    frame = pd.read_parquet(
        path,
        columns=["point_id", "u", "v", "used"],
    )
    if len(frame) > points:
        frame = frame.iloc[:points].copy()
    gdf = _point_frame_to_gdf(frame, env)
    reference = _ORIGINAL_SAMPLER(
        gdf,
        env,
        chunks=chunks,
        client=client,
        id_cols="point_id",
        require_inside=True,
    )
    candidate = _windowed_sample_raster_stack_chunked(
        gdf,
        env,
        chunks=chunks,
        client=client,
        id_cols="point_id",
        require_inside=True,
    )
    pd.testing.assert_frame_equal(reference, candidate, check_exact=True)
    del frame, gdf, reference, candidate
    gc.collect()


def _parse_windows(text: str) -> list[int]:
    result = []
    for token in text.split(","):
        value = int(token.strip())
        if value < 0:
            raise ValueError("submission windows must be non-negative")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("at least one submission window is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare production all-at-once vs windowed point-task submission."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="6x2")
    parser.add_argument("--partitions-per-batch", type=int, default=16)
    parser.add_argument("--batches-in-flight", type=int, default=4)
    parser.add_argument(
        "--submission-windows",
        default="0,24,48,96",
        help="Comma-separated chunk-task windows; 0 is unmodified production.",
    )
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    windows = _parse_windows(args.submission_windows)
    if min(
        args.point_count,
        args.partitions_per_batch,
        args.batches_in_flight,
        args.spatial_chunk,
        args.chunk_mb,
        args.validation_points,
        args.repeats,
    ) <= 0:
        parser.error("point/chunk/batch/validation/repeat values must be positive")

    workers, threads = parse_geometry(args.geometry)
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    if workers * threads > physical:
        parser.error(
            f"Geometry {args.geometry} needs {workers * threads} physical cores; "
            f"machine reports {physical}."
        )

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    full_env, surface_manifest, point_manifest = _load_inputs(root)
    entry = _point_workload_entry(point_manifest, args.point_count)
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )
    env, raster_workload = _window(full_env, args.raster_gib, storage_chunk)
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}

    total_memory_gib = psutil.virtual_memory().total / 1024**3
    local_tmp = root / "dask-tmp-points"
    local_tmp.mkdir(parents=True, exist_ok=True)
    execution = _local_configuration(
        workers=workers,
        threads=threads,
        total_memory_gib=total_memory_gib,
        managed_memory_fraction=args.managed_memory_fraction,
        local_directory=local_tmp,
        chunk_mb=args.chunk_mb,
        worker_startup_timeout=args.worker_startup_timeout,
    )
    client, cluster = execution.create_client()
    if client is None:
        raise RuntimeError("Submission-window profile requires a Dask client.")

    global _SUBMISSION_WINDOW
    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
        try:
            client.get_task_stream()
        except Exception:
            pass

        first_positive = next((value for value in windows if value > 0), 48)
        _SUBMISSION_WINDOW = first_positive
        _validate_candidate(
            entry=entry,
            env=env,
            client=client,
            chunks=chunks,
            points=args.validation_points,
        )
        print(f"validation passed for {args.validation_points:,} points")

        for repeat in range(1, args.repeats + 1):
            # Alternate the order across repeats to reduce cache/order bias while
            # remaining deterministic and easy to reproduce.
            ordered = windows if repeat % 2 else list(reversed(windows))
            for submission_window in ordered:
                if submission_window == 0:
                    point_base.sample_raster_stack_chunked = _ORIGINAL_SAMPLER
                    strategy = "production_all_at_once"
                else:
                    _SUBMISSION_WINDOW = int(submission_window)
                    point_base.sample_raster_stack_chunked = (
                        _windowed_sample_raster_stack_chunked
                    )
                    strategy = "windowed_scatter_submit"

                summary, _ = _run_once(
                    entry=entry,
                    env=env,
                    client=client,
                    chunks=chunks,
                    partitions_per_batch=args.partitions_per_batch,
                    batches_in_flight=args.batches_in_flight,
                    workers=workers,
                    threads_per_worker=threads,
                )
                record = {
                    **summary,
                    "repeat": repeat,
                    "submission_strategy": strategy,
                    "submission_window": int(submission_window),
                    "geometry": args.geometry,
                    "workers": workers,
                    "threads_per_worker": threads,
                    "execution_threads": workers * threads,
                    "spatial_chunk": args.spatial_chunk,
                    "raster_gib": raster_workload.logical_gib,
                }
                _append_jsonl(output, record)
                print(
                    f"repeat {repeat} window={submission_window}: "
                    f"pipeline={record['pipeline_seconds']:.3f}s "
                    f"points/s={record['throughput_points_s']:,.0f} "
                    f"worker_cores={record['worker_busy_cores_pipeline']:.2f} "
                    f"task_parallelism={record.get('task_compute_parallelism')}"
                )
                gc.collect()
    finally:
        point_base.sample_raster_stack_chunked = _ORIGINAL_SAMPLER
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
