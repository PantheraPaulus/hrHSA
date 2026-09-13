"""Launch workload-balanced point shards across one full node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from launch_point_spatial_shards import (
    _physical_core_groups,
    _set_affinity,
    _socket_cpu_sets,
    _wait_until_ready,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=1_000_000_000)
    parser.add_argument("--raster-gib", type=float, default=192.0)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers-per-shard", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, required=True)
    parser.add_argument("--graph-partitions", type=int, required=True)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--managed-memory-gib", type=float, required=True)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--barrier-timeout", type=float, default=600.0)
    args = parser.parse_args()

    required_per_shard = args.workers_per_shard * args.threads_per_worker
    allowed = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    grouped = _physical_core_groups(allowed)
    required_total = required_per_shard * args.shard_count
    if len(grouped) != required_total:
        raise RuntimeError(
            f"Balanced launcher needs {required_total} physical cores for "
            f"{args.shard_count} shards but batch affinity exposes {len(grouped)}."
        )

    cpu_sets = _socket_cpu_sets(allowed, args.shard_count)
    for index, cpu_set in enumerate(cpu_sets):
        physical = _physical_core_groups(cpu_set)
        if len(physical) != required_per_shard:
            raise RuntimeError(
                f"Balanced shard {index} got {len(physical)} physical cores; "
                f"expected {required_per_shard}."
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir = output_dir / "barrier"
    if barrier_dir.exists():
        shutil.rmtree(barrier_dir)
    barrier_dir.mkdir(parents=True)
    launcher_summary_path = output_dir / "launcher-summary.json"
    launcher_summary_path.unlink(missing_ok=True)

    runner = (
        Path(__file__).resolve().parent
        / "profile_point_partitioned_spatial_balanced_shard.py"
    )
    processes: list[tuple[int, subprocess.Popen, object]] = []
    started = time.perf_counter()
    all_ready = None
    try:
        # Keep setup/validation serialized for now because the shared production
        # validation helper still uses one temporary Parquet fixture.  The timed
        # sampling phase remains synchronized and fully concurrent at the barrier.
        for shard_index, cpu_set in enumerate(cpu_sets):
            log_path = output_dir / f"shard-{shard_index:02d}.launch.log"
            handle = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(runner),
                "--root", str(args.root),
                "--output-dir", str(output_dir),
                "--point-count", str(args.point_count),
                "--raster-gib", str(args.raster_gib),
                "--shard-count", str(args.shard_count),
                "--shard-index", str(shard_index),
                "--workers", str(args.workers_per_shard),
                "--threads-per-worker", str(args.threads_per_worker),
                "--graph-partitions", str(args.graph_partitions),
                "--spatial-chunk", str(args.spatial_chunk),
                "--managed-memory-gib", str(args.managed_memory_gib),
                "--validation-points", str(args.validation_points),
                "--worker-startup-timeout", str(args.worker_startup_timeout),
                "--repeats", str(args.repeats),
                "--barrier-dir", str(barrier_dir),
                "--barrier-timeout", str(args.barrier_timeout),
            ]
            print(
                f"launch balanced shard {shard_index}: "
                f"physical_cores={required_per_shard} "
                f"logical_cpus={len(cpu_set)} log={log_path}",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                preexec_fn=lambda cpus=cpu_set: _set_affinity(cpus),
            )
            processes.append((shard_index, process, handle))
            _wait_until_ready(
                shard_index=shard_index,
                process=process,
                barrier_dir=barrier_dir,
                log_path=log_path,
                timeout=args.worker_startup_timeout,
            )

        all_ready = time.perf_counter()
        print(
            f"all {args.shard_count} shards ready; setup_to_barrier="
            f"{all_ready - started:.3f}s",
            flush=True,
        )

        failed = None
        while True:
            remaining = 0
            for shard_index, process, _ in processes:
                code = process.poll()
                if code is None:
                    remaining += 1
                elif code != 0 and failed is None:
                    failed = (shard_index, code)
            if failed is not None or remaining == 0:
                break
            time.sleep(0.2)

        if failed is not None:
            for _, process, _ in processes:
                if process.poll() is None:
                    process.terminate()
            shard_index, code = failed
            raise RuntimeError(
                f"Balanced shard {shard_index} exited with status {code}; "
                "inspect its launch log."
            )

        for _, process, _ in processes:
            process.wait(timeout=30)
    finally:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, handle in processes:
            try:
                handle.close()
            except Exception:
                pass

    finished = time.perf_counter()
    launcher_summary = {
        "shard_count": args.shard_count,
        "workers_per_shard": args.workers_per_shard,
        "threads_per_worker": args.threads_per_worker,
        "point_count": args.point_count,
        "target_raster_gib": args.raster_gib,
        "repeats": args.repeats,
        "setup_to_barrier_seconds": (
            None if all_ready is None else all_ready - started
        ),
        "launcher_total_seconds": finished - started,
    }
    launcher_summary_path.write_text(
        json.dumps(launcher_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"all {args.shard_count} balanced shards completed; launcher wall="
        f"{finished - started:.3f}s",
        flush=True,
    )
    print(f"launcher summary: {launcher_summary_path}", flush=True)
    for shard_index in range(args.shard_count):
        log_path = output_dir / f"shard-{shard_index:02d}.launch.log"
        print(f"--- {log_path.name} ---")
        print(log_path.read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    main()
