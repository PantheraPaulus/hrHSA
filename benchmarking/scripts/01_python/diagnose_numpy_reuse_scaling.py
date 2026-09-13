"""Stage-3 diagnostic: remove per-task large-array allocation from the NumPy kernel.

Stage 2 showed that the negative scaling above ~4 threads remains after Dask and
xarray are removed.  This experiment keeps the same raw-NumPy arithmetic but
gives every ThreadPoolExecutor thread a persistent workspace (eta, scratch and
standardisation buffers) that is allocated during warm-up and then reused.

Question
--------
Is the 4-thread optimum caused mainly by concurrent allocation/freeing of large
NumPy arrays, or does it remain when the timed kernel performs essentially only
array arithmetic over already allocated memory?

Interpretation
--------------
* If 6/12-thread scaling improves strongly, allocator/temporary-array pressure
  is an important part of the Stage-2 collapse.
* If the same 4-thread optimum remains while all requested cores stay busy,
  repeated allocation is not the main mechanism and shared cache/memory
  bandwidth/resource contention becomes the leading explanation.

This is a diagnostic microbenchmark, not a production timing.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from hsa.compute.telemetry import (
    process_runtime_delta,
    process_runtime_snapshot,
    system_runtime_delta,
    system_runtime_snapshot,
)
from diagnose_numpy_thread_scaling import _load_raw_blocks, _parse_threads
from run_workstation_diagnostics import _affinity, _allowed_cpus, _physical_cpus


class _Workspace:
    """Persistent per-thread buffers for one block shape."""

    def __init__(self, shape: tuple[int, int], dtype: np.dtype):
        self.eta = np.empty(shape, dtype=dtype)
        self.scratch = np.empty(shape, dtype=dtype)
        self.z0 = np.empty(shape, dtype=dtype)
        self.z1 = np.empty(shape, dtype=dtype)
        self.ztemp = np.empty(shape, dtype=dtype)


class _WorkspaceRegistry:
    """Thread-local workspaces plus a small counter used to verify warm-up."""

    def __init__(self):
        self.local = threading.local()
        self._seen: set[int] = set()
        self._lock = threading.Lock()

    def get(self, shape: tuple[int, int], dtype: np.dtype) -> _Workspace:
        workspace = getattr(self.local, "workspace", None)
        if workspace is None:
            workspace = _Workspace(shape, dtype)
            self.local.workspace = workspace
            with self._lock:
                self._seen.add(threading.get_ident())
        return workspace

    @property
    def workspaces_created(self) -> int:
        with self._lock:
            return len(self._seen)


def _numpy_block_predict_reuse(
    data: np.ndarray,
    compute_dtype: str,
    registry: _WorkspaceRegistry,
) -> float:
    """Mirror Stage 2 while reusing all large temporary arrays per thread."""

    compute = np.dtype(compute_dtype)
    scalar = compute.type
    shape = (int(data.shape[1]), int(data.shape[2]))
    ws = registry.get(shape, compute)

    # np.full allocated eta in Stage 2.  Here the array already exists.
    ws.eta.fill(scalar(-0.5))

    for index in range(int(data.shape[0])):
        layer = np.asarray(data[index], dtype=compute)

        if index == 0:
            z = ws.z0
        elif index == 1:
            z = ws.z1
        else:
            # Bands >=2 are not needed again after their linear contribution, so
            # one persistent temporary can be reused for all of them.
            z = ws.ztemp

        np.subtract(layer, scalar(0.0), out=z)
        np.divide(z, scalar(1.0), out=z)

        coefficient = scalar(0.08 if index % 2 == 0 else -0.08)
        if index < 2:
            np.multiply(z, coefficient, out=ws.scratch)
            np.add(ws.eta, ws.scratch, out=ws.eta)
        else:
            np.multiply(z, coefficient, out=z)
            np.add(ws.eta, z, out=ws.eta)

    np.multiply(ws.z0, ws.z0, out=ws.scratch)
    np.multiply(ws.scratch, scalar(0.02), out=ws.scratch)
    np.add(ws.eta, ws.scratch, out=ws.eta)

    np.multiply(ws.z0, ws.z1, out=ws.scratch)
    np.multiply(ws.scratch, scalar(-0.015), out=ws.scratch)
    np.add(ws.eta, ws.scratch, out=ws.eta)

    np.exp(ws.eta, out=ws.eta)
    value = float(ws.eta[0, 0])
    return 0.0 if not np.isfinite(value) else value


def _run_batch(
    executor: ThreadPoolExecutor,
    blocks: list[np.ndarray],
    compute_dtype: str,
    registry: _WorkspaceRegistry,
    n_tasks: int,
) -> float:
    work = [blocks[index % len(blocks)] for index in range(n_tasks)]
    return float(
        sum(
            executor.map(
                lambda block: _numpy_block_predict_reuse(
                    block,
                    compute_dtype,
                    registry,
                ),
                work,
            )
        )
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure allocation-free raw-NumPy block scaling across threads"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", default="auto")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--resident-blocks", type=int, default=24)
    parser.add_argument("--tasks", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup-tasks", type=int, default=24)
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    if args.block_size <= 0 or args.resident_blocks <= 0 or args.tasks <= 0:
        parser.error("block-size, resident-blocks and tasks must be positive")
    if args.repeats <= 0 or args.warmup_tasks < 0:
        parser.error("repeats must be positive and warmup-tasks non-negative")

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    allowed = _allowed_cpus()
    physical_cpus, topology = _physical_cpus(allowed)
    thread_counts = _parse_threads(args.threads, len(physical_cpus))

    env_path = root / "data" / "environment.zarr"
    if not env_path.exists():
        raise FileNotFoundError(f"Expected prepared raster at {env_path}")

    env = xr.open_zarr(env_path, chunks=None)["environment"]
    if int(env.sizes["band"]) < 2:
        raise ValueError("Stage-3 kernel requires at least two bands")

    print(
        f"Preloading {args.resident_blocks} real {args.block_size}x{args.block_size} "
        "blocks as raw NumPy arrays ..."
    )
    blocks = _load_raw_blocks(
        env,
        args.block_size,
        args.resident_blocks,
        args.compute_dtype,
    )

    metadata = {
        "experiment": "raw_numpy_reused_workspace_thread_scaling_v1",
        "question": (
            "Does negative thread scaling remain when large NumPy temporaries "
            "are allocated once per thread and reused?"
        ),
        "physical_cpu_budget": len(physical_cpus),
        "physical_cpus": physical_cpus,
        "allowed_logical_cpus": allowed,
        "topology": topology,
        "thread_counts": thread_counts,
        "block_size": args.block_size,
        "resident_blocks": args.resident_blocks,
        "tasks_per_observation": args.tasks,
        "warmup_tasks": args.warmup_tasks,
        "repeats": args.repeats,
        "compute_dtype": args.compute_dtype,
        "storage_in_timed_region": False,
        "xarray_in_timed_kernel": False,
        "dask_in_timed_kernel": False,
        "large_temporary_allocation_in_timed_kernel": False,
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    print("Physical CPU representatives:", physical_cpus)
    print("Thread counts:", thread_counts)
    print("Results:", output)

    with _affinity(physical_cpus):
        for threads in thread_counts:
            print(f"\n=== {threads} thread{'s' if threads != 1 else ''} ===")
            registry = _WorkspaceRegistry()
            with ThreadPoolExecutor(
                max_workers=threads,
                thread_name_prefix="hrhsa-numpy-reuse",
            ) as executor:
                # Run enough warm-up work to force all pool threads to participate
                # and therefore allocate their workspaces outside timed regions.
                warmup_tasks = max(args.warmup_tasks, threads * 4)
                if warmup_tasks:
                    attempts = 0
                    while registry.workspaces_created < threads and attempts < 4:
                        start = time.perf_counter()
                        _run_batch(
                            executor,
                            blocks,
                            args.compute_dtype,
                            registry,
                            warmup_tasks,
                        )
                        attempts += 1
                        print(
                            f"warm-up {attempts}: {time.perf_counter() - start:.3f} s | "
                            f"workspaces={registry.workspaces_created}/{threads}"
                        )
                    if registry.workspaces_created < threads:
                        raise RuntimeError(
                            "Could not activate every ThreadPoolExecutor thread during warm-up; "
                            f"created {registry.workspaces_created}/{threads} workspaces"
                        )

                for repeat in range(1, args.repeats + 1):
                    process_before = process_runtime_snapshot()
                    system_before = system_runtime_snapshot()
                    start = time.perf_counter()
                    checksum = _run_batch(
                        executor,
                        blocks,
                        args.compute_dtype,
                        registry,
                        args.tasks,
                    )
                    wall = time.perf_counter() - start
                    system_after = system_runtime_snapshot()
                    process_after = process_runtime_snapshot()

                    process = process_runtime_delta(process_before, process_after)
                    system = system_runtime_delta(system_before, system_after)
                    cpu_seconds = process.get("cpu_total_seconds")
                    busy_cores = None if cpu_seconds is None else cpu_seconds / wall
                    utilization = (
                        None
                        if cpu_seconds is None
                        else cpu_seconds / (wall * threads)
                    )

                    payload = {
                        "threads": threads,
                        "repeat": repeat,
                        "wall_seconds": wall,
                        "tasks": args.tasks,
                        "tasks_per_second": args.tasks / wall,
                        "checksum": checksum,
                        "process_cpu_seconds": cpu_seconds,
                        "mean_busy_cores": busy_cores,
                        "requested_thread_utilization_fraction": utilization,
                        "process_context_switches": process.get("context_switches"),
                        "process_voluntary_context_switches": process.get(
                            "voluntary_context_switches"
                        ),
                        "process_involuntary_context_switches": process.get(
                            "involuntary_context_switches"
                        ),
                        "system_context_switches": system.get("ctx_switches"),
                        "system_cpu_busy_fraction": system.get("cpu_busy_fraction"),
                        "system_iowait_fraction": system.get("cpu_iowait_fraction"),
                        "system_disk_read_bytes": system.get("disk_read_bytes"),
                        "cpu_frequency_mean_mhz_before": system.get(
                            "cpu_frequency_mean_mhz_before"
                        ),
                        "cpu_frequency_mean_mhz_after": system.get(
                            "cpu_frequency_mean_mhz_after"
                        ),
                        "temperature_mean_c_before": system.get(
                            "temperature_mean_c_before"
                        ),
                        "temperature_mean_c_after": system.get(
                            "temperature_mean_c_after"
                        ),
                    }
                    _append_jsonl(output, payload)
                    print(
                        f"repeat {repeat}: {wall:.3f} s | "
                        f"{args.tasks / wall:.1f} blocks/s | "
                        f"busy cores={busy_cores if busy_cores is not None else 'NA'} | "
                        f"util={utilization if utilization is not None else 'NA'}"
                    )

    print("\nDone.")
    print("Diagnostic JSONL:", output)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
