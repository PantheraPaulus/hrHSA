"""Stage-2 diagnostic for workstation thread scaling.

This script removes Dask *and* xarray from the timed numerical kernel while
preserving the arithmetic pattern of hrHSA's fused RSF surface prediction.
It asks:

    Does the negative scaling above a few threads remain when the workload is
    reduced to raw NumPy array arithmetic inside one Python process?

Interpretation:
- if this stripped NumPy kernel shows the same 4-thread optimum and degradation
  at 6/12 threads, the dominant mechanism is below Dask/xarray (for example
  memory bandwidth/cache pressure or allocator/resource contention);
- if this kernel scales well while ``diagnose_block_thread_scaling.py`` does not,
  xarray/Python bookkeeping around ``_block_predict`` becomes a stronger suspect.

The experiment uses the same real Zarr blocks as Stage 1, preloads them before
timing, fixes total work across thread counts, and restricts execution to one
logical CPU representative per physical core.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

from hsa.compute.telemetry import (
    process_runtime_delta,
    process_runtime_snapshot,
    system_runtime_delta,
    system_runtime_snapshot,
)
from run_workstation_diagnostics import _affinity, _allowed_cpus, _physical_cpus


def _parse_threads(text: str, physical_budget: int) -> list[int]:
    if text.strip().lower() == "auto":
        candidates = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
        values = [value for value in candidates if value <= physical_budget]
        if physical_budget not in values:
            values.append(physical_budget)
        return sorted(set(values))

    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise ValueError("--threads must contain positive integers")
    if values[-1] > physical_budget:
        raise ValueError(
            f"Thread count {values[-1]} exceeds the {physical_budget}-physical-core budget"
        )
    return values


def _load_raw_blocks(
    env: xr.DataArray,
    block_size: int,
    n_blocks: int,
    compute_dtype: str,
) -> list[np.ndarray]:
    """Load distinct real raster blocks and convert them to plain NumPy arrays."""
    ny = int(env.sizes["y"])
    nx = int(env.sizes["x"])
    y_starts = list(range(0, ny - block_size + 1, block_size))
    x_starts = list(range(0, nx - block_size + 1, block_size))
    positions = [(y0, x0) for y0 in y_starts for x0 in x_starts]
    if not positions:
        raise ValueError("Raster is smaller than --block-size")
    if n_blocks > len(positions):
        raise ValueError(
            f"Requested {n_blocks} distinct blocks but raster contains only {len(positions)} "
            f"full {block_size}x{block_size} blocks"
        )

    indices = np.linspace(0, len(positions) - 1, n_blocks, dtype=int)
    dtype = np.dtype(compute_dtype)
    blocks: list[np.ndarray] = []
    for index in indices:
        y0, x0 = positions[int(index)]
        block = env.isel(
            y=slice(y0, y0 + block_size),
            x=slice(x0, x0 + block_size),
        ).load()
        data = np.asarray(block.transpose("band", "y", "x").values, dtype=dtype)
        blocks.append(np.ascontiguousarray(data))
    return blocks


def _numpy_block_predict(data: np.ndarray, compute_dtype: str) -> float:
    """Mirror the deterministic hrHSA fused kernel using only raw NumPy arrays.

    The deterministic workstation model uses six linear bands, a quadratic term
    for band 0, and an interaction between bands 0 and 1.  This function keeps
    the same allocation/reuse pattern as ``hsa.rsf.surface_fast._block_predict``
    but removes xarray objects, coordinate handling, dictionary lookups and model
    metadata from the timed kernel.
    """
    compute = np.dtype(compute_dtype)
    scalar = compute.type
    shape = (int(data.shape[1]), int(data.shape[2]))

    eta = np.full(shape, scalar(-0.5), dtype=compute)
    scratch = np.empty(shape, dtype=compute)
    retained: list[np.ndarray | None] = [None, None]

    for index in range(int(data.shape[0])):
        layer = np.asarray(data[index], dtype=compute)
        z = np.empty(shape, dtype=compute)
        np.subtract(layer, scalar(0.0), out=z)
        np.divide(z, scalar(1.0), out=z)

        coefficient = scalar(0.08 if index % 2 == 0 else -0.08)
        needs_later = index < 2
        if needs_later:
            np.multiply(z, coefficient, out=scratch)
            np.add(eta, scratch, out=eta)
            retained[index] = z
        else:
            np.multiply(z, coefficient, out=z)
            np.add(eta, z, out=eta)

    z0 = retained[0]
    z1 = retained[1]
    if z0 is None or z1 is None:
        raise RuntimeError("Expected at least two raster bands")

    np.multiply(z0, z0, out=scratch)
    np.multiply(scratch, scalar(0.02), out=scratch)
    np.add(eta, scratch, out=eta)

    np.multiply(z0, z1, out=scratch)
    np.multiply(scratch, scalar(-0.015), out=scratch)
    np.add(eta, scratch, out=eta)

    np.exp(eta, out=eta)
    value = float(eta[0, 0])
    return 0.0 if not np.isfinite(value) else value


def _run_batch(
    executor: ThreadPoolExecutor,
    blocks: list[np.ndarray],
    compute_dtype: str,
    n_tasks: int,
) -> float:
    work = [blocks[index % len(blocks)] for index in range(n_tasks)]
    return float(
        sum(executor.map(lambda block: _numpy_block_predict(block, compute_dtype), work))
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure raw-NumPy RSF-like block scaling across threads"
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

    # These ufunc-heavy kernels do not normally invoke BLAS, but keeping nested
    # native thread pools at one thread makes the intended concurrency explicit.
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
        raise ValueError("Stage-2 kernel requires at least two bands")

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
        "experiment": "raw_numpy_thread_scaling_v1",
        "question": "Does negative thread scaling remain after removing Dask and xarray?",
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
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    print("Physical CPU representatives:", physical_cpus)
    print("Thread counts:", thread_counts)
    print("Results:", output)

    with _affinity(physical_cpus):
        for threads in thread_counts:
            print(f"\n=== {threads} thread{'s' if threads != 1 else ''} ===")
            with ThreadPoolExecutor(
                max_workers=threads,
                thread_name_prefix="hrhsa-numpy",
            ) as executor:
                if args.warmup_tasks:
                    start = time.perf_counter()
                    _run_batch(executor, blocks, args.compute_dtype, args.warmup_tasks)
                    print(f"warm-up: {time.perf_counter() - start:.3f} s")

                for repeat in range(1, args.repeats + 1):
                    process_before = process_runtime_snapshot()
                    system_before = system_runtime_snapshot()
                    start = time.perf_counter()
                    checksum = _run_batch(executor, blocks, args.compute_dtype, args.tasks)
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
