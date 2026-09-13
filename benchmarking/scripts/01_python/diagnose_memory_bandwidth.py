"""Measure a first-touch, pinned NumPy memory-bandwidth envelope.

This is a mechanism diagnostic, not a scientific benchmark. It allocates three
large float64 arrays, first-touches each thread's slice on the CPU that will later
process it, then times a STREAM-like vector add::

    a = b + c

The reported bandwidth is *nominal* array traffic (two reads + one write). It is
useful for relative scaling across core counts and sockets; it is not a claim
about exact DRAM-controller traffic because write allocate, cache effects and the
processor's memory hierarchy can change physical traffic.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from hsa.compute import discover_runtime_topology
from hsa.compute.telemetry import process_runtime_delta, process_runtime_snapshot

def _run_owned_slice(
    s: slice,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    iterations: int,
    ready: threading.Barrier,
    go: threading.Barrier,
) -> None:
    # This function remains on one pinned executor thread for its full
    # lifetime, so first-touch and timed accesses occur on the same CPU.
    a[s].fill(0.0)
    b[s].fill(1.0)
    c[s].fill(2.0)

    # Untimed warm-up.
    np.add(b[s], c[s], out=a[s])

    ready.wait()
    go.wait()

    for _ in range(iterations):
        np.add(b[s], c[s], out=a[s])

class _PinRegistry:
    def __init__(self, cpus: list[int]):
        self._cpus: queue.Queue[int] = queue.Queue()
        for cpu in cpus:
            self._cpus.put(cpu)
        self.mapping: dict[int, int] = {}
        self._lock = threading.Lock()

    def initializer(self) -> None:
        cpu = self._cpus.get_nowait()
        os.sched_setaffinity(0, {cpu})
        with self._lock:
            self.mapping[threading.get_native_id()] = cpu


def _parse_counts(text: str, maximum: int) -> list[int]:
    values = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    if not values or values[0] <= 0 or values[-1] > maximum:
        raise ValueError(f"thread counts must lie in 1..{maximum}")
    return values


def _slices(length: int, n: int) -> list[slice]:
    bounds = np.linspace(0, length, n + 1, dtype=np.int64)
    return [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(n)]


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned first-touch NumPy memory-bandwidth diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", default="1,2,4,7,14,28,56,112")
    parser.add_argument("--total-gib", type=float, default=12.0,
                        help="Combined size of the three arrays in GiB")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.total_gib <= 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("sizes, iterations and repeats must be positive")

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    topology = discover_runtime_topology()
    cpus = list(topology.physical_cpu_representatives or topology.cpu_affinity)
    counts = _parse_counts(args.threads, len(cpus))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps({
            "experiment": "pinned_first_touch_numpy_add_v1",
            "topology": topology.as_dict(),
            "physical_cpu_order": cpus,
            "thread_counts": counts,
            "total_gib": args.total_gib,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "traffic_definition": "nominal two reads plus one write",
        }, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    # Three float64 arrays share the requested total allocation. One vector-add
    # iteration therefore has nominal traffic equal to their combined size.
    total_bytes = int(args.total_gib * 1024**3)
    n_values = max(1, total_bytes // (3 * np.dtype("float64").itemsize))
    actual_total_bytes = n_values * 3 * np.dtype("float64").itemsize
    nominal_bytes_per_iteration = actual_total_bytes

    for threads in counts:
        selected = cpus[:threads]
        slices = _slices(n_values, threads)
        print(f"\n=== memory add: {threads} threads, {actual_total_bytes / 1024**3:.2f} GiB arrays total ===")

        for repeat in range(1, args.repeats + 1):
            # np.empty reserves virtual address space; the parallel fills below
            # first-touch the physical pages on the worker CPU that will use them.
            a = np.empty(n_values, dtype=np.float64)
            b = np.empty(n_values, dtype=np.float64)
            c = np.empty(n_values, dtype=np.float64)
            pins = _PinRegistry(selected)
            ready = threading.Barrier(threads + 1)
            go = threading.Barrier(threads + 1)

            with ThreadPoolExecutor(
                max_workers=threads,
                initializer=pins.initializer,
            ) as executor:
                futures = [
                    executor.submit(
                        _run_owned_slice,
                        s,
                        a,
                        b,
                        c,
                        args.iterations,
                        ready,
                        go,
                    )
                    for s in slices
                ]

                # Wait until every pinned worker has first-touched and warmed its
                # own slice.
                ready.wait()

                before = process_runtime_snapshot()
                started = time.perf_counter()

                # Release all workers into the timed region together.
                go.wait()

                for future in futures:
                    future.result()

                wall = time.perf_counter() - started
                after = process_runtime_snapshot()

                checksum = float(
                    sum(a[s.start] for s in slices if s.start < n_values)
                )

            process = process_runtime_delta(before, after)
            traffic = nominal_bytes_per_iteration * args.iterations
            payload = {
                "threads": threads,
                "selected_cpus": selected,
                "repeat": repeat,
                "wall_seconds": wall,
                "iterations": args.iterations,
                "nominal_bytes": traffic,
                "nominal_gib_per_second": traffic / wall / 1024**3,
                "process_cpu_seconds": process.get("cpu_total_seconds"),
                "mean_busy_cores": None if process.get("cpu_total_seconds") is None else process["cpu_total_seconds"] / wall,
                "checksum": checksum,
                "thread_cpu_mapping": pins.mapping,
            }
            _append(output, payload)
            print(f"repeat {repeat}: {wall:.3f} s | {payload['nominal_gib_per_second']:.1f} GiB/s nominal")

            del a, b, c


if __name__ == "__main__":
    main()
