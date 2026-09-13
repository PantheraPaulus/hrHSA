"""Mechanism-oriented CoolMUC-4 raster campaign inside an existing allocation.

The publication benchmark asks *how fast?*.  This runner asks *why?* while still
using the production planner, Slurm worker launcher and scientific surface kernel.
It can run one-node and multi-node subsets inside a larger cm4_std allocation so a
strong-scaling comparison is made within the same allocation/time window.

Per observation it records:

* worker CPU seconds, context switches and end-state memory;
* Dask task-stream compute/transfer/deserialisation time and task concurrency;
* per-node CPU/iowait/disk/network deltas;
* driver CPU time;
* strict physical-core topology validation and the execution plan.

These counters are deliberately lightweight.  Hardware PMU experiments belong in
separate direct-kernel perf runs so profiling does not perturb publication timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import socket
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import xarray as xr

from hsa.compute import (
    append_benchmark_record,
    build_run_manifest,
    coolmuc4_plan,
    current_slurm_allocation,
    finalize_run_manifest,
    slurm_allocation_client,
    spatial_task_count,
    validate_worker_topology,
    write_run_manifest,
)
from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    distributed_worker_runtime_snapshot,
    process_runtime_delta,
    process_runtime_snapshot,
    summarize_task_stream,
    system_runtime_delta,
    system_runtime_snapshot,
)
from run_planned_execution import _allocation_physical_cores_per_node
from run_worker_geometry import _model, _run_once


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")

def _scheduler_process_snapshot(dask_scheduler=None) -> dict[str, Any]:
    """Collect process telemetry inside the Dask scheduler process."""
    payload = process_runtime_snapshot()
    payload["hostname"] = socket.gethostname()
    return payload


def _node_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "system": system_runtime_snapshot(),
    }
    try:
        import psutil

        net = psutil.net_io_counters()
        payload["network"] = {} if net is None else {
            "bytes_sent": int(getattr(net, "bytes_sent", 0)),
            "bytes_recv": int(getattr(net, "bytes_recv", 0)),
            "packets_sent": int(getattr(net, "packets_sent", 0)),
            "packets_recv": int(getattr(net, "packets_recv", 0)),
        }
    except Exception:
        payload["network"] = {}
    return payload


def _host_snapshots(client) -> dict[str, dict[str, Any]]:
    observed = client.run(_node_snapshot)
    by_host: dict[str, dict[str, Any]] = {}
    for worker, payload in sorted(observed.items()):
        host = str(payload.get("hostname") or worker)
        by_host.setdefault(host, dict(payload))
    return by_host


def _host_deltas(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for host in sorted(set(before).intersection(after)):
        system = system_runtime_delta(before[host].get("system"), after[host].get("system"))
        network: dict[str, float] = {}
        left = dict(before[host].get("network") or {})
        right = dict(after[host].get("network") or {})
        for key in ("bytes_sent", "bytes_recv", "packets_sent", "packets_recv"):
            try:
                network[key] = max(0.0, float(right.get(key, 0)) - float(left.get(key, 0)))
            except (TypeError, ValueError):
                pass
        result[host] = {"system": system, "network": network}
    return result


def _sum_host_field(hosts: dict[str, dict[str, Any]], section: str, field: str) -> float | None:
    values: list[float] = []
    for payload in hosts.values():
        try:
            value = payload.get(section, {}).get(field)
            if value is not None:
                values.append(float(value))
        except (TypeError, ValueError):
            pass
    return None if not values else float(sum(values))


def _parse_geometries(text: str) -> list[str]:
    result = [value.strip().lower() for value in text.split(",") if value.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one geometry is required")
    return result


def _parse_nodes(text: str) -> list[int]:
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("node counts must be positive")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanism-oriented CoolMUC-4 surface diagnostics")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometries", type=_parse_geometries,
                        default=["8x14", "16x7", "56x2", "112x1"])
    parser.add_argument("--node-counts", type=_parse_nodes, default=[1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--interface", default=None)
    args = parser.parse_args()

    if args.repeats <= 0 or args.warmup_repeats < 0:
        parser.error("repeats must be positive and warmup-repeats non-negative")

    allocation = current_slurm_allocation()
    if allocation is None:
        raise RuntimeError("this diagnostic must run inside Slurm")
    if max(args.node_counts) > allocation.nodes:
        raise RuntimeError(
            f"requested node subset {max(args.node_counts)} exceeds allocation of {allocation.nodes} node(s)"
        )

    root = args.root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    env = xr.open_zarr(root / "environment.zarr", chunks={})["environment"]
    chunks = {"band": -1, "y": args.spatial_chunk, "x": args.spatial_chunk}
    task_count = spatial_task_count(env, chunks)
    model, scaler, spec, meta = _model(env)

    physical_cores = _allocation_physical_cores_per_node(allocation)
    campaign = {
        "campaign": "coolmuc4-mechanisms-v1",
        "allocation": {
            "job_id": allocation.job_id,
            "cluster": allocation.cluster,
            "partition": allocation.partition,
            "nodes": allocation.nodes,
            "reported_cpus_per_node": allocation.cpus_per_node,
            "physical_plan_cores_per_node": physical_cores,
            "node_list": allocation.node_list,
        },
        "raster_shape": list(env.shape),
        "raster_dims": list(env.dims),
        "raster_dtype": str(env.dtype),
        "raster_uncompressed_gib": env.nbytes / 1024**3,
        "spatial_task_count": task_count,
        "node_counts": args.node_counts,
        "geometries": args.geometries,
        "repeats": args.repeats,
        "warmup_repeats": args.warmup_repeats,
        "interpretation_note": (
            "Compare worker CPU occupancy, task-stream concurrency/transfer and per-node "
            "I/O/network deltas alongside wall time; these are mechanism diagnostics, not PMU counters."
        ),
    }
    (out / "campaign.json").write_text(json.dumps(campaign, indent=2, default=str) + "\n", encoding="utf-8")
    diagnostics_path = out / "coolmuc4_mechanisms.jsonl"
    diagnostics_path.unlink(missing_ok=True)

    for nodes in args.node_counts:
        for geometry in args.geometries:
            plan = coolmuc4_plan(
                workload="surface_prediction",
                nodes=nodes,
                cores_per_node=physical_cores,
                geometry=geometry,
                task_count=task_count,
                chunk_mb=args.chunk_mb,
            )
            # A one-node subset inside a cm4_std allocation is still physically
            # running under cm4_std; preserve that fact in provenance.
            if allocation.partition and plan.partition != allocation.partition:
                plan = replace(plan, partition=allocation.partition)

            label = f"n{nodes}-{geometry}"
            print(f"\n=== {label} ===")
            print(plan.explain())
            output = out / f"surface-{label}.jsonl"
            output.unlink(missing_ok=True)
            manifest_path = output.with_suffix(output.suffix + ".manifest.json")
            scheduler_file = out / f"scheduler-{label}-{allocation.job_id}.json"
            benchmark_args = SimpleNamespace(
                workers=plan.total_workers,
                threads_per_worker=plan.geometry.threads_per_worker,
                chunk_mb=plan.chunk_mb,
                warmup_repeats=args.warmup_repeats,
            )

            manifest = None
            try:
                with slurm_allocation_client(
                    plan,
                    scheduler_file=scheduler_file,
                    interface=args.interface,
                    worker_startup_timeout=args.worker_startup_timeout,
                    validate_topology=False,
                ) as client:
                    validation = validate_worker_topology(client, plan, strict_affinity=True)
                    print(validation.explain())
                    if not validation.ok:
                        raise RuntimeError("strict physical-core topology validation failed")

                    try:
                        client.get_task_stream()
                    except Exception:
                        pass

                    manifest = build_run_manifest(
                        plan,
                        client=client,
                        dataset={
                            "root": str(root),
                            "environment_shape": list(env.shape),
                            "environment_dtype": str(env.dtype),
                            "computational_chunks": chunks,
                            "spatial_task_count": task_count,
                        },
                        extra={
                            "campaign": "coolmuc4-mechanisms-v1",
                            "node_subset": nodes,
                            "output_jsonl": str(output),
                            "diagnostics_jsonl": str(diagnostics_path),
                        },
                        strict_affinity=True,
                    )
                    write_run_manifest(manifest, manifest_path)

                    for warmup in range(1, args.warmup_repeats + 1):
                        print(f"warm-up {warmup}/{args.warmup_repeats}")
                        _run_once(
                            "surface", points=None, env=env, chunks=chunks, client=client,
                            args=benchmark_args, task_count=task_count, repeat=0,
                            model=model, scaler=scaler, spec=spec, meta=meta,
                        )

                    for repeat in range(1, args.repeats + 1):
                        worker_before = distributed_worker_runtime_snapshot(client)
                        driver_before = process_runtime_snapshot()
                        scheduler_before = client.run_on_scheduler(_scheduler_process_snapshot)
                        hosts_before = _host_snapshots(client)
                        task_start = time.time()

                        record = _run_once(
                            "surface", points=None, env=env, chunks=chunks, client=client,
                            args=benchmark_args, task_count=task_count, repeat=repeat,
                            model=model, scaler=scaler, spec=spec, meta=meta,
                        )

                        task_stop = time.time()
                        hosts_after = _host_snapshots(client)
                        scheduler_after = client.run_on_scheduler(_scheduler_process_snapshot)
                        driver_after = process_runtime_snapshot()
                        worker_after = distributed_worker_runtime_snapshot(client)
                        try:
                            events = client.get_task_stream(start=task_start, stop=task_stop)
                        except Exception:
                            events = []

                        worker = aggregate_worker_runtime_delta(worker_before, worker_after)
                        driver = process_runtime_delta(driver_before, driver_after)
                        scheduler = process_runtime_delta(scheduler_before, scheduler_after)
                        scheduler["hostname"] = scheduler_after.get("hostname")
                        hosts = _host_deltas(hosts_before, hosts_after)
                        tasks = summarize_task_stream(events)
                        wall = float(record.wall_seconds)
                        scheduler_cpu = scheduler.get("cpu_total_seconds")
                        if scheduler_cpu is not None:
                            scheduler["cpu_fraction_of_one_core"] = scheduler_cpu / wall
                        worker_cpu = worker.get("cpu_total_seconds")
                        if worker_cpu is not None:
                            worker["mean_busy_execution_threads"] = worker_cpu / wall
                            worker["execution_thread_utilization_fraction"] = (
                                worker_cpu / (wall * plan.total_workers * plan.geometry.threads_per_worker)
                            )

                        diagnostic = {
                            "run_key": f"{label}-r{repeat}",
                            "repeat": repeat,
                            "nodes": nodes,
                            "geometry": geometry,
                            "wall_seconds": wall,
                            "execution_plan": plan.as_dict(),
                            "topology_validation": validation.as_dict(),
                            "worker": worker,
                            "driver": driver,
                            "scheduler": scheduler,
                            "hosts": hosts,
                            "task_stream": tasks,
                        }
                        _append_jsonl(diagnostics_path, diagnostic)

                        record.metadata.update({
                            "campaign": "coolmuc4-mechanisms-v1",
                            "node_subset": nodes,
                            "execution_plan": plan.as_dict(),
                            "topology_validation": validation.as_dict(),
                            "diagnostics_jsonl": str(diagnostics_path),
                            "worker_cpu_seconds": worker_cpu,
                            "worker_execution_thread_utilization_fraction": worker.get("execution_thread_utilization_fraction"),
                            "worker_context_switches": worker.get("context_switches"),
                            "task_compute_seconds": tasks.get("compute_seconds"),
                            "task_transfer_seconds": tasks.get("transfer_seconds"),
                            "task_deserialize_seconds": tasks.get("deserialize_seconds"),
                            "task_compute_parallelism": tasks.get("compute_parallelism"),
                            "host_disk_read_bytes_total": _sum_host_field(hosts, "system", "disk_read_bytes"),
                            "host_iowait_fraction_sum": _sum_host_field(hosts, "system", "cpu_iowait_fraction"),
                            "host_network_bytes_recv_total": _sum_host_field(hosts, "network", "bytes_recv"),
                            "host_network_bytes_sent_total": _sum_host_field(hosts, "network", "bytes_sent"),
                            "scheduler_cpu_seconds": scheduler.get("cpu_total_seconds"),
                            "scheduler_context_switches": scheduler.get("context_switches"),
                            "scheduler_cpu_fraction_of_one_core": scheduler.get("cpu_fraction_of_one_core"),
                        })
                        append_benchmark_record(record, output)
                        print(pd.Series({
                            "wall_seconds": wall,
                            "worker_cpu_seconds": worker_cpu,
                            "worker_utilization": worker.get("execution_thread_utilization_fraction"),
                            "task_compute_parallelism": tasks.get("compute_parallelism"),
                            "task_transfer_seconds": tasks.get("transfer_seconds"),
                        }).to_string())
                        gc.collect()

                if manifest is not None:
                    finalize_run_manifest(manifest, manifest_path, status="completed")
            except BaseException:
                if manifest is not None:
                    finalize_run_manifest(manifest, manifest_path, status="failed")
                raise

    print("\nMechanism campaign complete:", out)
    print("Diagnostics:", diagnostics_path)


if __name__ == "__main__":
    main()
