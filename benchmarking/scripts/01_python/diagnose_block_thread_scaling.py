"""Stage-1 diagnostic for the workstation performance investigation.

This script removes Dask from the experiment while keeping hrHSA's real fused
RSF block kernel.  It asks a narrow question:

    If several threads call ``_block_predict`` concurrently inside one Python
    process, does throughput deteriorate as the thread count rises?

That makes it a useful discriminator between a Dask-specific effect and
intra-process contention in Python/xarray/NumPy or the hardware memory/cache
subsystem.

The experiment deliberately:
- uses one logical CPU representative per physical core;
- preloads a fixed set of real Zarr blocks so storage is outside timed regions;
- keeps total scientific work fixed across thread counts;
- creates the ThreadPoolExecutor outside timed regions so pool startup is not
  confused with kernel throughput;
- records process/system CPU and context-switch counters alongside wall time.

This is a diagnostic benchmark, not a production performance number.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hsa.compute.telemetry import (
    process_runtime_delta,
    process_runtime_snapshot,
    system_runtime_delta,
    system_runtime_snapshot,
)
from hsa.rsf.surface_fast import _block_predict
from run_workstation import _deterministic_model
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


def _load_blocks(env: xr.DataArray, block_size: int, n_blocks: int) -> list[xr.DataArray]:
    """Preload distinct real raster blocks into RAM before any timing begins."""

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

    # Spread selections over the raster rather than taking only adjacent blocks.
    indices = np.linspace(0, len(positions) - 1, n_blocks, dtype=int)
    blocks: list[xr.DataArray] = []
    for index in indices:
        y0, x0 = positions[int(index)]
        block = (
            env.isel(
                y=slice(y0, y0 + block_size),
                x=slice(x0, x0 + block_size),
            )
            .load()
        )
        blocks.append(block)
    return blocks


def _kernel_kwargs(env: xr.DataArray, compute_dtype: str) -> dict[str, Any]:
    model, scaler, spec, meta = _deterministic_model(env)
    coefficients = {str(name): float(value) for name, value in pd.Series(model.params).items()}
    return {
        "coefficients": coefficients,
        "means": [float(value) for value in scaler.mean_],
        "scales": [float(value) for value in scaler.scale_],
        "spec": spec,
        "meta": meta,
        "dtype": "float32",
        "compute_dtype": compute_dtype,
    }


def _evaluate(block: xr.DataArray, kwargs: dict[str, Any]) -> float:
    """Run the real eager hrHSA block kernel and return a tiny checksum."""

    result = _block_predict(block, **kwargs)
    value = float(result.values[0, 0])
    return 0.0 if not np.isfinite(value) else value


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _run_batch(
    executor: ThreadPoolExecutor,
    blocks: list[xr.DataArray],
    kwargs: dict[str, Any],
    n_tasks: int,
) -> float:
    work = [blocks[index % len(blocks)] for index in range(n_tasks)]
    # executor.map preserves lazy iteration. Materialising the small scalar results
    # guarantees all block computations have completed before the timer stops.
    return float(sum(executor.map(lambda block: _evaluate(block, kwargs), work)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure direct hrHSA block-kernel scaling across threads without Dask"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Workstation campaign root containing data/environment.zarr",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", default="auto")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--resident-blocks",
        type=int,
        default=24,
        help="Distinct real blocks preloaded into RAM and cycled during the benchmark",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=240,
        help="Fixed number of block evaluations per timed observation",
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup-tasks", type=int, default=24)
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    if args.block_size <= 0 or args.resident_blocks <= 0 or args.tasks <= 0:
        parser.error("block-size, resident-blocks and tasks must be positive")
    if args.repeats <= 0 or args.warmup_tasks < 0:
        parser.error("repeats must be positive and warmup-tasks non-negative")

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    # Keep nested numerical-library thread pools from silently multiplying our
    # requested ThreadPoolExecutor thread count.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    allowed = _allowed_cpus()
    physical_cpus, topology = _physical_cpus(allowed)
    thread_counts = _parse_threads(args.threads, len(physical_cpus))

    env_path = root / "data" / "environment.zarr"
    if not env_path.exists():
        raise FileNotFoundError(f"Expected prepared raster at {env_path}")

    # chunks=None avoids constructing a Dask-backed xarray object; selected blocks
    # are read synchronously during preload and are NumPy-backed during timing.
    env = xr.open_zarr(env_path, chunks=None)["environment"]
    print(f"Preloading {args.resident_blocks} real {args.block_size}x{args.block_size} blocks ...")
    blocks = _load_blocks(env, args.block_size, args.resident_blocks)
    kwargs = _kernel_kwargs(env, args.compute_dtype)

    metadata = {
        "experiment": "direct_block_thread_scaling_v1",
        "question": "Does the real hrHSA block kernel degrade with threads when Dask is removed?",
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
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    print("Physical CPU representatives:", physical_cpus)
    print("Thread counts:", thread_counts)
    print("Results:", output)

    with _affinity(physical_cpus):
        for threads in thread_counts:
            print(f"\n=== {threads} thread{'s' if threads != 1 else ''} ===")
            # Creating threads while the parent is pinned ensures worker threads
            # inherit the same physical-core-only affinity mask.
            with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="hrhsa-kernel") as executor:
                if args.warmup_tasks:
                    start = time.perf_counter()
                    _run_batch(executor, blocks, kwargs, args.warmup_tasks)
                    print(f"warm-up: {time.perf_counter() - start:.3f} s")

                for repeat in range(1, args.repeats + 1):
                    process_before = process_runtime_snapshot()
                    system_before = system_runtime_snapshot()
                    start = time.perf_counter()
                    checksum = _run_batch(executor, blocks, kwargs, args.tasks)
                    wall = time.perf_counter() - start
                    system_after = system_runtime_snapshot()
                    process_after = process_runtime_snapshot()

                    process = process_runtime_delta(process_before, process_after)
                    system = system_runtime_delta(system_before, system_after)
                    cpu_seconds = process.get("cpu_total_seconds")
                    busy_cores = None if cpu_seconds is None else cpu_seconds / wall
                    utilization = (
                        None if cpu_seconds is None else cpu_seconds / (wall * threads)
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
                        "process_voluntary_context_switches": process.get("voluntary_context_switches"),
                        "process_involuntary_context_switches": process.get("involuntary_context_switches"),
                        "system_context_switches": system.get("ctx_switches"),
                        "system_cpu_busy_fraction": system.get("cpu_busy_fraction"),
                        "system_iowait_fraction": system.get("cpu_iowait_fraction"),
                        "system_disk_read_bytes": system.get("disk_read_bytes"),
                        "cpu_frequency_mean_mhz_before": system.get("cpu_frequency_mean_mhz_before"),
                        "cpu_frequency_mean_mhz_after": system.get("cpu_frequency_mean_mhz_after"),
                        "temperature_mean_c_before": system.get("temperature_mean_c_before"),
                        "temperature_mean_c_after": system.get("temperature_mean_c_after"),
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
