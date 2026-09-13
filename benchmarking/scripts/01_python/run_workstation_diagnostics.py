"""Mechanism-oriented workstation geometry benchmark for hrHSA raster prediction.

Unlike ``run_workstation.py`` (which asks *what is fast?*), this campaign records
low-overhead counters that help answer *why?*. It runs the same surface kernel at
fixed CPU budgets while varying Dask process/thread geometry, once on physical
cores only and optionally once with SMT/logical CPUs enabled.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil
import xarray as xr

from hsa.compute import (
    ExecutionConfig,
    append_benchmark_record,
    benchmark_timer,
    make_benchmark_record,
    spatial_task_count,
    task_density,
)
from hsa.compute.persistence import persist_distributed, release_distributed
from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    distributed_worker_runtime_snapshot,
    process_runtime_delta,
    process_runtime_snapshot,
    summarize_task_stream,
    system_runtime_delta,
    system_runtime_snapshot,
)
from hsa.rsf import predict_rsf_surface_chunked
from run_workstation import PROFILES, _deterministic_model, _memory_limit


def _cpu_list(text: str) -> list[int]:
    result: list[int] = []
    for token in text.strip().split(","):
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(token))
    return sorted(set(result))


def _allowed_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(psutil.cpu_count(logical=True) or 1))


def _physical_cpus(allowed: list[int]) -> tuple[list[int], list[dict[str, Any]]]:
    rows = []
    for cpu in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            rows.append(
                {
                    "cpu": cpu,
                    "core_id": int((base / "core_id").read_text().strip()),
                    "package_id": int((base / "physical_package_id").read_text().strip()),
                    "thread_siblings": _cpu_list((base / "thread_siblings_list").read_text()),
                }
            )
        except (FileNotFoundError, ValueError, OSError):
            pass

    if len(rows) == len(allowed):
        representatives: dict[tuple[int, int], int] = {}
        for row in rows:
            key = (row["package_id"], row["core_id"])
            representatives[key] = min(representatives.get(key, row["cpu"]), row["cpu"])
        return sorted(representatives.values()), rows

    if psutil.cpu_count(logical=False) == len(allowed):
        return allowed, rows

    raise RuntimeError(
        "Cannot map logical CPUs to physical cores from Linux sysfs. "
        "Use --modes smt only rather than treating an arbitrary subset as physical."
    )


@contextmanager
def _affinity(cpus: list[int]):
    if not hasattr(os, "sched_setaffinity"):
        if cpus != _allowed_cpus():
            raise RuntimeError("Physical-core pinning requires Linux sched_setaffinity")
        yield
        return
    previous = set(os.sched_getaffinity(0))
    requested = set(cpus)
    if not requested.issubset(previous):
        raise ValueError("Requested CPU set lies outside the current process affinity")
    os.sched_setaffinity(0, requested)
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


def _geometries(text: str, budget: int) -> list[tuple[int, int]]:
    if text.lower() == "auto":
        return [(w, budget // w) for w in range(budget, 0, -1) if budget % w == 0]
    result = []
    for item in text.split(","):
        workers, threads = (int(v) for v in item.lower().strip().split("x", 1))
        if workers * threads != budget:
            raise ValueError(f"{item} does not use the fixed {budget}-thread budget")
        result.append((workers, threads))
    return result


def _block_repeats(total: int, blocks: int) -> list[int]:
    if total < blocks:
        raise ValueError("--repeats must be at least --blocks")
    base, remainder = divmod(total, blocks)
    return [base + (i < remainder) for i in range(blocks)]


def _cmd(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _machine(root: Path, allowed: list[int], physical: list[int], topology) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    governors = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
        try:
            governors.add(path.read_text().strip())
        except OSError:
            pass
    return {
        "physical_cores_reported": psutil.cpu_count(logical=False),
        "logical_cpus_reported": psutil.cpu_count(logical=True),
        "allowed_logical_cpus": allowed,
        "physical_cpu_representatives": physical,
        "linux_cpu_topology": topology,
        "memory_total_gib": vm.total / 1024**3,
        "memory_available_gib": vm.available / 1024**3,
        "governors": sorted(governors),
        "lscpu": _cmd("lscpu", "-J"),
        "lscpu_extended": _cmd("lscpu", "-e=CPU,CORE,SOCKET,NODE,CACHE,ONLINE,MAXMHZ,MINMHZ"),
        "numactl_hardware": _cmd("numactl", "--hardware"),
        "filesystem": _cmd("findmnt", "-T", str(root)),
        "block_devices": _cmd("lsblk", "-o", "NAME,TYPE,SIZE,ROTA,MODEL,FSTYPE,MOUNTPOINTS"),
        "system_snapshot": system_runtime_snapshot(),
    }


def _append(payload: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _surface(env, chunks, client, model, scaler, spec, meta, dtype):
    predicted = predict_rsf_surface_chunked(
        env, model, scaler, spec, meta, chunks=chunks, compute_dtype=dtype
    )
    return predicted, persist_distributed(client, predicted.data)


def _run_geometry(
    *, mode, block, workers, threads, repeats, warmups, cpus, env, chunks,
    chunk_mb, total_memory_gib, memory_fraction, tmpdir, benchmark_path,
    diagnostic_path, task_count, dtype, repeat_start,
):
    memory_limit = _memory_limit(total_memory_gib, workers, memory_fraction)
    model, scaler, spec, meta = _deterministic_model(env)
    print(f"\n[{mode} block {block}] {workers}x{threads}; {len(cpus)} CPUs; {memory_limit}/worker")

    with _affinity(cpus):
        execution = ExecutionConfig(
            backend="local", n_workers=workers, threads_per_worker=threads,
            processes=True, memory_limit=memory_limit, local_directory=str(tmpdir),
            dashboard_address=None, chunk_mb=chunk_mb, worker_startup_timeout=300,
        )
        client, cluster = execution.create_client()
        if client is None:
            raise RuntimeError("A Dask client is required")
        try:
            client.wait_for_workers(workers, timeout=300)
            try:
                client.get_task_stream()
            except Exception:
                pass
            observed = distributed_worker_runtime_snapshot(client)
            print("  worker cpusets:", {k: v.get("cpu_affinity") for k, v in observed.items()})

            for warmup in range(1, warmups + 1):
                predicted = persisted = None
                try:
                    started = time.perf_counter()
                    predicted, persisted = _surface(env, chunks, client, model, scaler, spec, meta, dtype)
                    print(f"  warm-up {warmup}/{warmups}: {time.perf_counter()-started:.3f} s")
                finally:
                    release_distributed(client, persisted)
                    del predicted, persisted
                    gc.collect()

            for local_repeat in range(repeats):
                repeat = repeat_start + local_repeat
                run_key = f"{mode}-b{block:02d}-{workers}x{threads}-r{repeat:03d}"
                predicted = persisted = None
                wb = distributed_worker_runtime_snapshot(client)
                db = process_runtime_snapshot()
                sb = system_runtime_snapshot()
                task_start = time.time()
                try:
                    with benchmark_timer(client=client) as timer:
                        predicted, persisted = _surface(env, chunks, client, model, scaler, spec, meta, dtype)
                    task_stop = time.time()
                    sa = system_runtime_snapshot()
                    da = process_runtime_snapshot()
                    wa = distributed_worker_runtime_snapshot(client)
                    try:
                        events = client.get_task_stream(start=task_start, stop=task_stop)
                    except Exception:
                        events = []

                    worker = aggregate_worker_runtime_delta(wb, wa)
                    driver = process_runtime_delta(db, da)
                    system = system_runtime_delta(sb, sa)
                    tasks = summarize_task_stream(events)
                    wall = float(timer["wall_seconds"])
                    if worker.get("cpu_total_seconds") is not None:
                        worker["mean_busy_execution_threads"] = worker["cpu_total_seconds"] / wall
                        worker["execution_thread_utilization_fraction"] = (
                            worker["cpu_total_seconds"] / (wall * workers * threads)
                        )
                    if driver.get("cpu_total_seconds") is not None:
                        driver["cpu_fraction_of_one_core"] = driver["cpu_total_seconds"] / wall
                    if system.get("ctx_switches") is not None:
                        system["context_switches_per_second"] = system["ctx_switches"] / wall

                    diagnostic = {
                        "run_key": run_key, "mode": mode, "block": block,
                        "repeat": repeat, "geometry": f"{workers}x{threads}",
                        "workers": workers, "threads_per_worker": threads,
                        "execution_threads": workers * threads, "cpuset": cpus,
                        "wall_seconds": wall, "worker": worker, "driver": driver,
                        "system": system, "task_stream": tasks,
                    }
                    _append(diagnostic, diagnostic_path)

                    record = make_benchmark_record(
                        "workstation_diagnostic_surface_prediction", wall,
                        rows=int(env.sizes["x"] * env.sizes["y"]), workers=workers,
                        threads_per_worker=threads, chunk_mb=chunk_mb, tasks=task_count,
                        bytes_processed=int(env.nbytes), operation_memory=timer, client=client,
                        metadata={
                            "campaign": "workstation_geometry_mechanisms_v1",
                            "run_key": run_key, "mode": mode, "block": block,
                            "repeat": repeat, "worker_geometry": f"{workers}x{threads}",
                            "total_execution_threads": workers * threads, "cpuset": cpus,
                            "task_density_per_worker": task_density(task_count, workers),
                            "raster_size_x": int(env.sizes["x"]),
                            "raster_size_y": int(env.sizes["y"]),
                            "bands": int(env.sizes["band"]),
                            "surface_compute_dtype": dtype,
                            "materialization": "distributed_persist_no_final_assembly",
                            "diagnostics_jsonl": str(diagnostic_path),
                            "worker_cpu_seconds": worker.get("cpu_total_seconds"),
                            "worker_context_switches": worker.get("context_switches"),
                            "worker_pss_end_total_bytes": worker.get("end_pss_bytes_total"),
                            "driver_cpu_seconds": driver.get("cpu_total_seconds"),
                            "system_context_switches": system.get("ctx_switches"),
                            "system_cpu_busy_fraction": system.get("cpu_busy_fraction"),
                            "system_iowait_fraction": system.get("cpu_iowait_fraction"),
                            "system_disk_read_bytes": system.get("disk_read_bytes"),
                            "task_compute_seconds": tasks.get("compute_seconds"),
                            "task_transfer_seconds": tasks.get("transfer_seconds"),
                            "task_compute_parallelism": tasks.get("compute_parallelism"),
                        },
                    )
                    append_benchmark_record(record, benchmark_path)
                    print(
                        f"  repeat {repeat}: {wall:.3f} s; "
                        f"worker CPU={worker.get('cpu_total_seconds')}; "
                        f"ctx={system.get('ctx_switches')}"
                    )
                finally:
                    release_distributed(client, persisted)
                    del predicted, persisted
                    gc.collect()
        finally:
            client.close()
            if cluster is not None:
                cluster.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Investigate the workstation Dask geometry envelope")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--profile", choices=sorted(PROFILES), default="stress")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--warmup-repeats", type=int, default=1)
    p.add_argument("--storage-chunk", type=int, default=1024)
    p.add_argument("--chunk-mb", type=int, default=256)
    p.add_argument("--managed-memory-fraction", type=float, default=0.75)
    p.add_argument("--surface-compute-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--modes", default="physical,smt")
    p.add_argument("--physical-geometries", default="auto")
    p.add_argument("--smt-geometries", default="auto")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--force-prepare", action="store_true")
    args = p.parse_args()

    if args.repeats <= 0 or args.blocks <= 0 or args.warmup_repeats < 0:
        p.error("repeats/blocks must be positive and warmup-repeats non-negative")
    if not 0 < args.managed_memory_fraction < 1:
        p.error("--managed-memory-fraction must lie in (0, 1)")
    repeats_by_block = _block_repeats(args.repeats, args.blocks)

    modes = [v.strip().lower() for v in args.modes.split(",") if v.strip()]
    if set(modes) - {"physical", "smt"}:
        p.error("--modes may contain only physical,smt")

    root, out = args.root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    data_root, tmpdir = root / "data", root / "dask-tmp-diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["TMPDIR"] = str(tmpdir)

    profile = PROFILES[args.profile]
    if not args.skip_prepare:
        command = [
            sys.executable, str(Path(__file__).with_name("prepare_data.py")),
            "--root", str(data_root), "--size", str(profile["size"]),
            "--bands", str(profile["bands"]), "--storage-chunk", str(args.storage_chunk),
            "--points", str(profile["points"]),
            "--workers", str(min(psutil.cpu_count(logical=False) or 1, 8)),
        ]
        if args.force_prepare:
            command.append("--force")
        subprocess.run(command, check=True)

    env = xr.open_zarr(data_root / "environment.zarr", chunks={})["environment"]
    chunks = {"band": -1, "y": args.storage_chunk, "x": args.storage_chunk}
    task_count = spatial_task_count(env, chunks)
    memory_gib = psutil.virtual_memory().total / 1024**3
    allowed = _allowed_cpus()
    physical, topology = _physical_cpus(allowed)
    if len(physical) == len(allowed) and "smt" in modes:
        print("No additional SMT siblings are visible; skipping SMT mode.")
        modes.remove("smt")

    (out / "machine.json").write_text(json.dumps(_machine(root, allowed, physical, topology), indent=2, default=str) + "\n")
    campaign = {
        "campaign": "workstation_geometry_mechanisms_v1", "profile": args.profile,
        "raster_shape": list(env.shape), "raster_dims": list(env.dims),
        "raster_dtype": str(env.dtype), "raster_uncompressed_gib": env.nbytes / 1024**3,
        "spatial_task_count": task_count, "storage_chunk": args.storage_chunk,
        "chunk_mb": args.chunk_mb, "managed_memory_fraction": args.managed_memory_fraction,
        "surface_compute_dtype": args.surface_compute_dtype, "modes": modes,
        "repeats": args.repeats, "blocks": args.blocks, "warmup_repeats": args.warmup_repeats,
        "physical_cpu_budget": len(physical), "logical_cpu_budget": len(allowed),
        "design_note": "Odd blocks run process-heavy to thread-heavy; even blocks reverse the order.",
    }
    (out / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    diag_path = out / "workstation_diagnostics.jsonl"
    diag_path.unlink(missing_ok=True)

    specs = []
    if "physical" in modes:
        specs.append(("physical", physical, _geometries(args.physical_geometries, len(physical))))
    if "smt" in modes:
        specs.append(("smt", allowed, _geometries(args.smt_geometries, len(allowed))))

    for mode, cpus, geometries in specs:
        benchmark_path = out / f"surface_{mode}.jsonl"
        benchmark_path.unlink(missing_ok=True)
        repeat_start = 1
        for block, block_repeats in enumerate(repeats_by_block, start=1):
            order = geometries if block % 2 else list(reversed(geometries))
            print(f"\n=== {mode.upper()} block {block}: " + ", ".join(f"{w}x{t}" for w, t in order))
            for workers, threads in order:
                _run_geometry(
                    mode=mode, block=block, workers=workers, threads=threads,
                    repeats=block_repeats, warmups=args.warmup_repeats, cpus=cpus,
                    env=env, chunks=chunks, chunk_mb=args.chunk_mb,
                    total_memory_gib=memory_gib, memory_fraction=args.managed_memory_fraction,
                    tmpdir=tmpdir, benchmark_path=benchmark_path, diagnostic_path=diag_path,
                    task_count=task_count, dtype=args.surface_compute_dtype,
                    repeat_start=repeat_start,
                )
            repeat_start += block_repeats

    print("\nComplete. Analyze these files with workstation_geometry_diagnostics.ipynb:")
    print(" ", out / "machine.json")
    print(" ", out / "campaign.json")
    print(" ", diag_path)
    for mode, _, _ in specs:
        print(" ", out / f"surface_{mode}.jsonl")


if __name__ == "__main__":
    main()
