"""Instrumented capacity/strong/weak surface-scaling runner.

This companion keeps the validated workload/planning logic in
``run_surface_scaling.py`` unchanged and replaces only its repeat loop with
mechanism telemetry suitable for publication campaigns.

The primary benchmark JSONL receives compact, analysis-friendly summary fields.
Full worker/driver/scheduler/node/task-stream diagnostics are written beside it
as ``<output>.diagnostics.jsonl``.
"""

from __future__ import annotations

import gc
import json
import time
import uuid
from pathlib import Path
from typing import Any

import run_surface_scaling as base
from hsa.compute import (
    append_benchmark_record,
    benchmark_timer,
    make_benchmark_record,
    spatial_task_count,
)
from hsa.compute.scaling_telemetry import (
    distributed_node_runtime_delta,
    distributed_node_runtime_snapshot,
    scheduler_process_delta,
    scheduler_process_snapshot,
)
from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    distributed_worker_runtime_snapshot,
    process_runtime_delta,
    process_runtime_snapshot,
    summarize_task_stream,
)
from hsa.compute.persistence import persist_distributed, release_distributed
from hsa.rsf import predict_rsf_surface_chunked
from run_worker_geometry import _model


RUN_ID = uuid.uuid4().hex


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _scheduler_snapshot(client) -> dict[str, Any]:
    try:
        return dict(client.run_on_scheduler(scheduler_process_snapshot))
    except Exception as exc:
        return {"available": False, "reason": f"scheduler snapshot failed: {exc}"}


def _task_events(client, *, start: float, stop: float):
    try:
        return client.get_task_stream(start=start, stop=stop)
    except Exception:
        return []


def _add_rates(
    *,
    worker: dict[str, Any],
    driver: dict[str, Any],
    scheduler: dict[str, Any],
    nodes: dict[str, Any],
    wall: float,
    workers: int,
    threads_per_worker: int,
) -> None:
    worker_cpu = worker.get("cpu_total_seconds")
    if worker_cpu is not None and wall > 0:
        worker["mean_busy_execution_threads"] = worker_cpu / wall
        worker["execution_thread_utilization_fraction"] = (
            worker_cpu / (wall * workers * threads_per_worker)
        )

    for key in (
        "context_switches",
        "voluntary_context_switches",
        "involuntary_context_switches",
    ):
        value = worker.get(key)
        if value is not None and wall > 0:
            worker[f"{key}_per_second"] = value / wall

    driver_cpu = driver.get("cpu_total_seconds")
    if driver_cpu is not None and wall > 0:
        driver["cpu_fraction_of_one_core"] = driver_cpu / wall

    scheduler_cpu = scheduler.get("cpu_total_seconds")
    if scheduler_cpu is not None and wall > 0:
        scheduler["cpu_fraction_of_one_core"] = scheduler_cpu / wall

    aggregate = dict(nodes.get("aggregate") or {})
    node_ctx = aggregate.get("ctx_switches")
    if node_ctx is not None and wall > 0:
        aggregate["context_switches_per_second"] = node_ctx / wall
    nodes["aggregate"] = aggregate


def _summary_metadata(
    *,
    diagnostics_path: Path,
    run_key: str,
    worker: dict[str, Any],
    driver: dict[str, Any],
    scheduler: dict[str, Any],
    nodes: dict[str, Any],
    tasks: dict[str, Any],
) -> dict[str, Any]:
    node = dict(nodes.get("aggregate") or {})
    return {
        "run_id": RUN_ID,
        "run_key": run_key,
        "diagnostics_jsonl": str(diagnostics_path),
        "worker_cpu_seconds": worker.get("cpu_total_seconds"),
        "worker_busy_execution_threads": worker.get("mean_busy_execution_threads"),
        "worker_execution_thread_utilization_fraction": worker.get(
            "execution_thread_utilization_fraction"
        ),
        "worker_context_switches": worker.get("context_switches"),
        "worker_context_switches_per_second": worker.get(
            "context_switches_per_second"
        ),
        "worker_involuntary_context_switches": worker.get(
            "involuntary_context_switches"
        ),
        "worker_involuntary_context_switches_per_second": worker.get(
            "involuntary_context_switches_per_second"
        ),
        "worker_pss_end_total_bytes": worker.get("end_pss_bytes_total"),
        "worker_uss_end_total_bytes": worker.get("end_uss_bytes_total"),
        "driver_cpu_seconds": driver.get("cpu_total_seconds"),
        "driver_cpu_fraction_of_one_core": driver.get("cpu_fraction_of_one_core"),
        "scheduler_cpu_seconds": scheduler.get("cpu_total_seconds"),
        "scheduler_cpu_fraction_of_one_core": scheduler.get(
            "cpu_fraction_of_one_core"
        ),
        "scheduler_context_switches": scheduler.get("context_switches"),
        "node_count_observed": len(nodes.get("hosts_matched") or []),
        "node_cpu_busy_fraction_mean": node.get("mean_cpu_busy_fraction"),
        "node_cpu_iowait_fraction_mean": node.get("mean_cpu_iowait_fraction"),
        "node_disk_read_bytes": node.get("disk_read_bytes"),
        "node_disk_write_bytes": node.get("disk_write_bytes"),
        "node_context_switches": node.get("ctx_switches"),
        "node_major_page_faults": node.get("major_page_faults"),
        "node_page_faults": node.get("page_faults"),
        "node_page_scans": node.get("page_scans"),
        "node_page_steals": node.get("page_steals"),
        "node_workingset_refaults": node.get("workingset_refaults"),
        "node_swap_pages_in": node.get("swap_pages_in"),
        "node_swap_pages_out": node.get("swap_pages_out"),
        "task_stream_tasks": tasks.get("tasks"),
        "task_compute_seconds": tasks.get("compute_seconds"),
        "task_transfer_seconds": tasks.get("transfer_seconds"),
        "task_deserialize_seconds": tasks.get("deserialize_seconds"),
        "task_compute_parallelism": tasks.get("compute_parallelism"),
        "task_compute_mean_seconds": tasks.get("compute_task_mean_seconds"),
        "task_compute_median_seconds": tasks.get("compute_task_median_seconds"),
        "task_stream_span_seconds": tasks.get("span_seconds"),
        "task_stream_nbytes_observed": tasks.get("nbytes_observed"),
    }


def _surface_prediction(env, chunks, client, model, scaler, spec, meta):
    predicted = predict_rsf_surface_chunked(
        env,
        model,
        scaler,
        spec,
        meta,
        chunks=chunks,
    )
    persisted = persist_distributed(client, predicted.data)
    return predicted, persisted


def _instrumented_run_surface_repeats(
    *,
    client,
    env,
    output: Path,
    chunks: dict[str, int],
    workers: int,
    threads_per_worker: int,
    chunk_mb: int,
    warmup_repeats: int,
    repeats: int,
    metadata: dict[str, Any],
) -> None:
    task_count = spatial_task_count(env, chunks)
    model, scaler, spec, meta = _model(env)
    diagnostics_path = output.with_suffix(output.suffix + ".diagnostics.jsonl")

    # The first call installs/initializes the task-stream plugin on Dask versions
    # where it is lazy. Keep that setup outside all measured observations.
    try:
        client.get_task_stream()
    except Exception:
        pass

    for warmup in range(1, warmup_repeats + 1):
        predicted = persisted = None
        print(
            f"warm-up {warmup}/{warmup_repeats}: "
            f"{workers}x{threads_per_worker}, "
            f"{metadata['actual_raster_gib']:.3f} GiB"
        )
        try:
            with benchmark_timer(client=client) as timer:
                predicted, persisted = _surface_prediction(
                    env, chunks, client, model, scaler, spec, meta
                )
            print(f"warm-up wall={timer['wall_seconds']:.3f} s")
        finally:
            release_distributed(client, persisted)
            del predicted, persisted
            gc.collect()

    for repeat in range(1, repeats + 1):
        run_key = (
            f"{RUN_ID}-"
            f"{metadata['scaling_mode']}-"
            f"{metadata['executed_geometry']}-"
            f"{metadata['actual_raster_gib']:.3f}gib-"
            f"r{repeat:03d}"
        )

        predicted = persisted = None
        worker_before = distributed_worker_runtime_snapshot(client)
        driver_before = process_runtime_snapshot()
        scheduler_before = _scheduler_snapshot(client)
        nodes_before = distributed_node_runtime_snapshot(client)
        task_start = time.time()

        try:
            # The timed interval contains only graph construction plus distributed
            # materialization. Telemetry snapshots happen outside the wall clock.
            with benchmark_timer(client=client) as timer:
                predicted, persisted = _surface_prediction(
                    env, chunks, client, model, scaler, spec, meta
                )
            task_stop = time.time()

            # Snapshot while the distributed output is still resident. This keeps
            # process/node state aligned with the completed operation rather than
            # measuring after cancellation/release.
            nodes_after = distributed_node_runtime_snapshot(client)
            scheduler_after = _scheduler_snapshot(client)
            driver_after = process_runtime_snapshot()
            worker_after = distributed_worker_runtime_snapshot(client)
            events = _task_events(client, start=task_start, stop=task_stop)

            wall = float(timer["wall_seconds"])
            worker = aggregate_worker_runtime_delta(worker_before, worker_after)
            driver = process_runtime_delta(driver_before, driver_after)
            scheduler = scheduler_process_delta(
                scheduler_before,
                scheduler_after,
                wall_seconds=wall,
            )
            nodes = distributed_node_runtime_delta(nodes_before, nodes_after)
            tasks = summarize_task_stream(events)

            _add_rates(
                worker=worker,
                driver=driver,
                scheduler=scheduler,
                nodes=nodes,
                wall=wall,
                workers=workers,
                threads_per_worker=threads_per_worker,
            )

            diagnostic = {
                "run_id": RUN_ID,
                "run_key": run_key,
                "benchmark": f"surface_{metadata['scaling_mode']}_scaling",
                "repeat": repeat,
                "wall_seconds": wall,
                "geometry": metadata["executed_geometry"],
                "workers": workers,
                "threads_per_worker": threads_per_worker,
                "physical_cores_used": metadata["physical_cores_used"],
                "target_raster_gib": metadata["target_raster_gib"],
                "actual_raster_gib": metadata["actual_raster_gib"],
                "raster_to_memory_ratio": metadata["raster_to_memory_ratio"],
                "worker": worker,
                "driver": driver,
                "scheduler": scheduler,
                "nodes": nodes,
                "task_stream": tasks,
            }
            _append_jsonl(diagnostics_path, diagnostic)

            record_metadata = {
                **metadata,
                "repeat": repeat,
                "warmup_repeats": warmup_repeats,
                "spatial_task_count": task_count,
                "computational_chunks": dict(chunks),
                "output_partitions": int(getattr(persisted, "npartitions", 1)),
                **_summary_metadata(
                    diagnostics_path=diagnostics_path,
                    run_key=run_key,
                    worker=worker,
                    driver=driver,
                    scheduler=scheduler,
                    nodes=nodes,
                    tasks=tasks,
                ),
            }
            record = make_benchmark_record(
                diagnostic["benchmark"],
                wall,
                rows=int(env.sizes["x"] * env.sizes["y"]),
                workers=workers,
                threads_per_worker=threads_per_worker,
                chunk_mb=chunk_mb,
                tasks=task_count,
                bytes_processed=int(env.nbytes),
                metadata=record_metadata,
                operation_memory=timer,
                client=client,
            )
            append_benchmark_record(record, output)

            print(
                f"repeat {repeat}: {wall:.3f} s; "
                f"busy_threads={worker.get('mean_busy_execution_threads')}; "
                f"iowait={dict(nodes.get('aggregate') or {}).get('mean_cpu_iowait_fraction')}; "
                f"major_faults={dict(nodes.get('aggregate') or {}).get('major_page_faults')}; "
                f"disk_read={dict(nodes.get('aggregate') or {}).get('disk_read_bytes')}"
            )
        finally:
            release_distributed(client, persisted)
            del predicted, persisted
            gc.collect()


# Replace only the timing/repeat layer. CLI parsing, workload resolution, local
# execution and Slurm topology planning remain exactly those of the lean runner.
base._run_surface_repeats = _instrumented_run_surface_repeats


if __name__ == "__main__":
    base.main()
