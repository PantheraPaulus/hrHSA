"""Profile bounded overlap between spatial point-sampling batches.

This is a calibration benchmark rather than a formal scaling campaign. It uses
the exact production point-scaling workload path and varies only execution-policy
parameters such as bounded batches in flight, batch width and worker geometry.
Results can therefore be used directly to choose the policy for a frozen final
capacity/strong/weak campaign without maintaining a second concurrency
implementation in the profiler.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import psutil

from hsa.compute.scaling_telemetry import (
    distributed_node_runtime_delta,
    distributed_node_runtime_snapshot,
)
from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    distributed_worker_runtime_snapshot,
)
from hsa.compute.workloads import parse_geometry
from run_point_scaling import (
    _load_inputs,
    _point_workload_entry,
    _process_point_workload,
    _sample_frame,
)
from run_surface_scaling import _local_configuration, _window


def _run_once(
    *,
    entry,
    env,
    client,
    chunks,
    partitions_per_batch: int,
    batches_in_flight: int,
):
    worker_before = distributed_worker_runtime_snapshot(client)
    node_before = distributed_node_runtime_snapshot(client)

    result = _process_point_workload(
        entry=entry,
        env=env,
        client=client,
        chunks=chunks,
        write_root=None,
        run_key="point-concurrency-profile",
        partitions_per_batch=partitions_per_batch,
        batches_in_flight=batches_in_flight,
    )

    node_after = distributed_node_runtime_snapshot(client)
    worker_after = distributed_worker_runtime_snapshot(client)

    worker = aggregate_worker_runtime_delta(worker_before, worker_after)
    nodes = distributed_node_runtime_delta(node_before, node_after)
    aggregate = dict(nodes.get("aggregate") or {})

    pipeline_seconds = float(result["pipeline_seconds"])
    worker_cpu_seconds = worker.get("cpu_total_seconds")
    busy_cores = (
        None
        if worker_cpu_seconds is None or pipeline_seconds <= 0
        else float(worker_cpu_seconds) / pipeline_seconds
    )

    operation_memory = dict(result.get("operation_memory") or {})
    expected_rows = int(entry["target_points"])
    expected_partitions = int(entry["partitions"])

    return {
        "point_count": expected_rows,
        "partitions": expected_partitions,
        "partitions_per_batch": int(result["partitions_per_batch"]),
        "batches": int(result["batches"]),
        "batches_in_flight": int(result["batches_in_flight"]),
        "pipeline_seconds": pipeline_seconds,
        "throughput_points_s": expected_rows / pipeline_seconds,
        "read_seconds_sum": float(result["read_seconds"]),
        "assembly_seconds_sum": float(result["assembly_seconds"]),
        "geometry_seconds_sum": float(result["geometry_seconds"]),
        "sampling_call_seconds_sum": float(result["sampling_seconds"]),
        "sampling_call_seconds_max": result["sampling_call_seconds_max"],
        "sampling_overlap_window_seconds": result[
            "sampling_overlap_window_seconds"
        ],
        "sampling_overlap_factor": result["sampling_overlap_factor"],
        "worker_cpu_seconds": worker_cpu_seconds,
        "worker_busy_cores_pipeline": busy_cores,
        "node_cpu_busy_fraction_mean": aggregate.get("mean_cpu_busy_fraction"),
        "node_cpu_iowait_fraction_mean": aggregate.get("mean_cpu_iowait_fraction"),
        "node_disk_read_bytes": aggregate.get("disk_read_bytes"),
        "node_major_page_faults": aggregate.get("major_page_faults"),
        "node_workingset_refaults": aggregate.get("workingset_refaults"),
        "operation_peak_process_tree_rss_mb": operation_memory.get(
            "operation_peak_process_tree_rss_mb"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--geometry", default="4x3")
    parser.add_argument("--partitions-per-batch", type=int, default=16)
    parser.add_argument("--batches-in-flight", type=int, required=True)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--warmup-points", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    if min(
        args.point_count,
        args.partitions_per_batch,
        args.batches_in_flight,
        args.spatial_chunk,
        args.chunk_mb,
        args.warmup_points,
        args.repeats,
    ) <= 0:
        parser.error("point/chunk/batch/concurrency/repeat values must be positive")

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
    chunks = {
        "band": -1,
        "y": args.spatial_chunk,
        "x": args.spatial_chunk,
    }

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
        raise RuntimeError("Point concurrency profile requires a Dask client.")

    try:
        client.wait_for_workers(workers, timeout=args.worker_startup_timeout)

        warmup_path = Path(entry["directory"]) / "part-000000.parquet"
        warmup_frame = pd.read_parquet(
            warmup_path,
            columns=["point_id", "u", "v", "used"],
        )
        if len(warmup_frame) > args.warmup_points:
            warmup_frame = warmup_frame.iloc[: args.warmup_points].copy()
        warmup_points, warmup_sampled = _sample_frame(
            warmup_frame,
            env=env,
            client=client,
            chunks=chunks,
        )
        del warmup_frame, warmup_points, warmup_sampled
        gc.collect()

        for repeat in range(1, args.repeats + 1):
            result = _run_once(
                entry=entry,
                env=env,
                client=client,
                chunks=chunks,
                partitions_per_batch=args.partitions_per_batch,
                batches_in_flight=args.batches_in_flight,
            )
            record = {
                **result,
                "repeat": repeat,
                "geometry": args.geometry,
                "workers": workers,
                "threads_per_worker": threads,
                "spatial_chunk": args.spatial_chunk,
                "raster_gib": raster_workload.logical_gib,
            }
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")

            print(
                f"repeat {repeat}: in_flight={args.batches_in_flight} "
                f"pipeline={record['pipeline_seconds']:.3f}s "
                f"points/s={record['throughput_points_s']:,.0f} "
                f"busy_cores={record['worker_busy_cores_pipeline']:.2f} "
                f"overlap={record['sampling_overlap_factor']:.2f}x"
            )
            gc.collect()
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


if __name__ == "__main__":
    main()