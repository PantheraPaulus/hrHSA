"""Launch concurrent spatial point shards inside one full-node Slurm batch step.

Nested ``srun --cpus-per-task`` accounting on CoolMUC-4 exposes logical CPU ids
in a way that can halve the intended physical-core budget when SMT sibling masks
are expanded by ``--cpu-bind=cores``. This launcher avoids that ambiguity: it
uses the batch step's full-node affinity, groups logical CPUs by physical socket
and core, and starts one shard process per socket with an explicit Linux affinity
mask.

Shard setup/validation is intentionally serialized. The production validation
helper writes a shared scratch Parquet fixture, so allowing multiple shards to
validate simultaneously can race on that file. Each shard therefore reaches its
filesystem timing barrier before the next shard is launched. Once all shards are
ready, the existing barrier releases them together for the timed sampling phase.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def _cpu_topology(cpu_id: int) -> tuple[int, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{int(cpu_id)}/topology")
    try:
        package = int((base / "physical_package_id").read_text().strip())
        core = int((base / "core_id").read_text().strip())
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot resolve physical topology for CPU {cpu_id}.") from exc
    return package, core


def _physical_core_groups(cpu_ids: list[int]) -> dict[tuple[int, int], list[int]]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for cpu_id in cpu_ids:
        grouped.setdefault(_cpu_topology(cpu_id), []).append(int(cpu_id))
    return {key: sorted(value) for key, value in grouped.items()}


def _socket_cpu_sets(cpu_ids: list[int], shard_count: int) -> list[list[int]]:
    grouped = _physical_core_groups(cpu_ids)
    by_socket: dict[int, list[tuple[tuple[int, int], list[int]]]] = {}
    for key, siblings in grouped.items():
        by_socket.setdefault(key[0], []).append((key, siblings))
    sockets = []
    for package in sorted(by_socket):
        groups = sorted(by_socket[package], key=lambda item: item[0][1])
        sockets.append(sorted(cpu for _, siblings in groups for cpu in siblings))

    if len(sockets) == shard_count:
        return sockets

    # Fallback for systems where package ids are unavailable/unexpected: split
    # physical-core groups evenly while preserving sibling groups.
    core_groups = [grouped[key] for key in sorted(grouped)]
    if len(core_groups) % shard_count:
        raise RuntimeError(
            f"Cannot divide {len(core_groups)} physical cores evenly into {shard_count} shards."
        )
    width = len(core_groups) // shard_count
    return [
        sorted(cpu for group in core_groups[i * width : (i + 1) * width] for cpu in group)
        for i in range(shard_count)
    ]


def _set_affinity(cpu_ids: list[int]) -> None:
    os.sched_setaffinity(0, set(cpu_ids))


def _wait_until_ready(
    *,
    shard_index: int,
    process: subprocess.Popen,
    barrier_dir: Path,
    log_path: Path,
    timeout: float,
) -> None:
    """Wait until one shard has finished setup/validation and entered its barrier."""
    ready = barrier_dir / f"ready-{shard_index:02d}"
    deadline = time.monotonic() + float(timeout)
    while not ready.exists():
        code = process.poll()
        if code is not None:
            raise RuntimeError(
                f"Spatial shard {shard_index} exited with status {code} before "
                f"reaching the timing barrier; inspect {log_path}."
            )
        if time.monotonic() >= deadline:
            process.terminate()
            raise TimeoutError(
                f"Spatial shard {shard_index} did not reach its timing barrier "
                f"within {timeout:.0f}s; inspect {log_path}."
            )
        time.sleep(0.1)
    print(
        f"shard {shard_index} validation complete; waiting at timing barrier",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=1_000_000_000)
    parser.add_argument("--raster-gib", type=float, default=192.0)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--workers-per-shard", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=14)
    parser.add_argument("--graph-partitions", type=int, default=128)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--managed-memory-gib", type=float, default=96.0)
    parser.add_argument("--validation-points", type=int, default=100_000)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--barrier-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.shard_count <= 0:
        parser.error("shard-count must be positive")
    required_per_shard = args.workers_per_shard * args.threads_per_worker
    allowed = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    grouped = _physical_core_groups(allowed)
    required_total = required_per_shard * args.shard_count
    if len(grouped) != required_total:
        raise RuntimeError(
            "Full-node shard launcher expected "
            f"{required_total} physical cores but the batch step exposes "
            f"{len(grouped)} physical cores across {len(allowed)} logical CPUs. "
            "Do not launch this helper from a nested srun step."
        )

    socket_sets = _socket_cpu_sets(allowed, args.shard_count)
    for index, cpu_set in enumerate(socket_sets):
        physical = _physical_core_groups(cpu_set)
        if len(physical) != required_per_shard:
            raise RuntimeError(
                f"Shard {index} affinity contains {len(physical)} physical cores; "
                f"expected {required_per_shard}."
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir = output_dir / "barrier"
    if barrier_dir.exists():
        shutil.rmtree(barrier_dir)
    barrier_dir.mkdir(parents=True)

    runner = (
        Path(__file__).resolve().parent / "profile_point_partitioned_spatial_shard.py"
    )
    processes: list[tuple[int, subprocess.Popen, object]] = []
    started = time.perf_counter()
    try:
        # Launch shards one at a time until each has completed setup/validation
        # and is waiting at the timing barrier. This avoids a concurrent write/read
        # race on the validation Parquet fixture while preserving a synchronized,
        # fully concurrent timed sampling phase.
        for shard_index, cpu_set in enumerate(socket_sets):
            log_path = output_dir / f"shard-{shard_index:02d}.launch.log"
            log_handle = log_path.open("w", encoding="utf-8")
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
            physical_count = len(_physical_core_groups(cpu_set))
            print(
                f"launch shard {shard_index}: physical_cores={physical_count} "
                f"logical_cpus={len(cpu_set)} log={log_path}",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                preexec_fn=lambda cpus=cpu_set: _set_affinity(cpus),
            )
            processes.append((shard_index, process, log_handle))
            _wait_until_ready(
                shard_index=shard_index,
                process=process,
                barrier_dir=barrier_dir,
                log_path=log_path,
                timeout=args.worker_startup_timeout,
            )

        failed: tuple[int, int] | None = None
        while True:
            remaining = 0
            for shard_index, process, _ in processes:
                code = process.poll()
                if code is None:
                    remaining += 1
                elif code != 0 and failed is None:
                    failed = (shard_index, code)
            if failed is not None:
                for _, process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            if remaining == 0:
                break
            time.sleep(0.2)

        for _, process, _ in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        if failed is not None:
            shard_index, code = failed
            raise RuntimeError(
                f"Spatial shard {shard_index} exited with status {code}; "
                f"inspect {output_dir / f'shard-{shard_index:02d}.launch.log'}."
            )
    finally:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, handle in processes:
            try:
                handle.close()
            except Exception:
                pass

    print(
        f"all {args.shard_count} shards completed; concurrent launcher wall="
        f"{time.perf_counter() - started:.3f}s",
        flush=True,
    )
    for shard_index in range(args.shard_count):
        log_path = output_dir / f"shard-{shard_index:02d}.launch.log"
        print(f"--- {log_path.name} ---")
        print(log_path.read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    main()
