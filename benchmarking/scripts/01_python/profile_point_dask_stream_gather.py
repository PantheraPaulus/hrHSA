"""Measure a full block-local point graph with incremental result draining.

This wrapper reuses the validated block-local sampler unchanged. It runs all point
partitions as one static Dask graph, but replaces bulk ``Client.gather(list)``
with an ``as_completed`` iterator so completed ~1M-row result partitions are
transferred and discarded one at a time while the remaining graph is still busy.
No package behavior is changed.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import psutil
from distributed import as_completed

from hsa.compute.workloads import parse_geometry
from profile_point_dask_block_local import _run_once, _validate_candidate
from profile_point_dask_native_graph import _append_jsonl, _route_meta
from run_point_scaling import _load_inputs, _point_workload_entry
from run_surface_scaling import _local_configuration, _window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="6x2")
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

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
    tile_rows = int(entry["tile_rows"])
    tile_cols = int(entry["tile_cols"])
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )
    env, raster_workload = _window(full_env, args.raster_gib, storage_chunk)
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}
    meta, raster = _route_meta(env, chunks)
    raster_blocks = np.asarray(raster.data.to_delayed(), dtype=object)

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
        raise RuntimeError("Stream-gather profile requires a Dask client.")

    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)
        try:
            client.get_task_stream()
        except Exception:
            pass

        _validate_candidate(
            entry=entry,
            env=env,
            chunks=chunks,
            meta=meta,
            raster_blocks=raster_blocks,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            client=client,
            validation_points=args.validation_points,
        )
        print(f"validation passed for {args.validation_points:,} points")

        real_gather = client.gather

        def stream_gather(futures, *gather_args, **gather_kwargs):
            # _run_once passes a list of final partition futures. Yield results
            # as soon as they arrive instead of building one giant result list.
            if not isinstance(futures, (list, tuple)):
                return real_gather(futures, *gather_args, **gather_kwargs)

            def results():
                for future, result in as_completed(
                    list(futures),
                    with_results=True,
                    raise_errors=True,
                ):
                    try:
                        yield result
                    finally:
                        # Release each completed final partition immediately;
                        # unfinished futures keep shared dependencies alive.
                        try:
                            future.release()
                        except Exception:
                            pass

            return results()

        client.gather = stream_gather
        try:
            total_partitions = int(entry["partitions"])
            for repeat in range(1, args.repeats + 1):
                record = _run_once(
                    entry=entry,
                    meta=meta,
                    raster_blocks=raster_blocks,
                    client=client,
                    tile_rows=tile_rows,
                    tile_cols=tile_cols,
                    submission_group_partitions=total_partitions,
                    groups_in_flight=1,
                    workers=workers,
                    threads_per_worker=threads,
                )
                span = record.get("task_stream_span_seconds")
                record.update(
                    {
                        "repeat": repeat,
                        "geometry": args.geometry,
                        "workers": workers,
                        "threads_per_worker": threads,
                        "execution_threads": workers * threads,
                        "spatial_chunk": args.spatial_chunk,
                        "raster_gib": raster_workload.logical_gib,
                        "architecture": "dask_native_block_local_stream_gather",
                        "result_drain_strategy": "as_completed",
                        "pipeline_minus_task_span_seconds": (
                            None
                            if span is None
                            else record["pipeline_seconds"] - float(span)
                        ),
                    }
                )
                _append_jsonl(output, record)
                print(
                    f"repeat {repeat}: "
                    f"pipeline={record['pipeline_seconds']:.3f}s "
                    f"points/s={record['throughput_points_s']:,.0f} "
                    f"worker_cores={record['worker_busy_cores_pipeline']} "
                    f"task_parallelism={record['task_compute_parallelism']} "
                    f"task_span={record['task_stream_span_seconds']} "
                    f"tail={record['pipeline_minus_task_span_seconds']} "
                    f"transfer_s={record['task_transfer_seconds']} "
                    f"rss_mb={record['operation_peak_process_tree_rss_mb']}"
                )
                gc.collect()
        finally:
            client.gather = real_gather
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()
