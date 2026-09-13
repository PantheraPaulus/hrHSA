"""Stage-4 diagnostic: pin each NumPy worker thread to one physical CPU.

Stages 2 and 3 showed that the negative scaling above ~4 threads remains after
Dask/xarray are removed and after large temporary arrays are reused. The first
perf experiment then showed a sharp rise in CPU migrations and cache-miss rate
at 12 threads.

This experiment keeps the Stage-3 reused-workspace NumPy kernel but pins each
ThreadPoolExecutor worker to a distinct physical CPU. It can either select the
first N physical CPUs (``--threads``) or an explicit CPU placement
(``--cpus 0,3,6,9``), which makes cache-domain placement experiments possible.

Interpretation
--------------
* If explicit pinning improves throughput and IPC strongly, thread placement /
  cache locality is important.
* If pinning leaves throughput and IPC poor, the dominant limit is deeper in
  the shared memory hierarchy (cache capacity/bandwidth/latency).

This is a diagnostic microbenchmark, not a production timing.
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

import xarray as xr

from hsa.compute.telemetry import (
    process_runtime_delta,
    process_runtime_snapshot,
    system_runtime_delta,
    system_runtime_snapshot,
)
from diagnose_numpy_reuse_scaling import (
    _WorkspaceRegistry,
    _numpy_block_predict_reuse,
)
from diagnose_numpy_thread_scaling import _load_raw_blocks, _parse_threads
from run_workstation_diagnostics import _affinity, _allowed_cpus, _physical_cpus


class _ThreadPinRegistry:
    """Assign one physical CPU to each executor worker exactly once."""

    def __init__(self, cpus: list[int]):
        self._cpus: queue.Queue[int] = queue.Queue()
        for cpu in cpus:
            self._cpus.put(cpu)
        self._mapping: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def initializer(self) -> None:
        try:
            cpu = self._cpus.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError("More executor threads started than CPUs supplied") from exc

        # On Linux pid=0 means the calling thread, so this pins the executor
        # worker itself rather than the whole Python process.
        os.sched_setaffinity(0, {cpu})
        native_id = threading.get_native_id()
        with self._lock:
            self._mapping[native_id] = {
                "cpu": cpu,
                "affinity": sorted(os.sched_getaffinity(0)),
                "python_ident": threading.get_ident(),
            }

    def mapping(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return dict(self._mapping)


def _parse_explicit_cpus(text: str | None, physical_cpus: list[int]) -> list[int] | None:
    if text is None or not text.strip():
        return None
    cpus = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not cpus:
        raise ValueError("--cpus must contain at least one CPU id")
    if len(cpus) != len(set(cpus)):
        raise ValueError("--cpus must not contain duplicate CPU ids")
    unavailable = [cpu for cpu in cpus if cpu not in physical_cpus]
    if unavailable:
        raise ValueError(
            f"--cpus contains CPUs that are not physical-core representatives in the "
            f"current affinity mask: {unavailable}; available={physical_cpus}"
        )
    return cpus


def _run_batch(
    executor: ThreadPoolExecutor,
    blocks,
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
        description="Measure reused-workspace NumPy scaling with one CPU pinned per thread"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", default="12")
    parser.add_argument(
        "--cpus",
        default=None,
        help=(
            "Optional explicit comma-separated physical CPU placement, e.g. 0,3,6,9. "
            "When supplied it overrides --threads and uses one worker per listed CPU."
        ),
    )
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--resident-blocks", type=int, default=12)
    parser.add_argument("--tasks", type=int, default=4800)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-tasks", type=int, default=0)
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

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
    explicit_cpus = _parse_explicit_cpus(args.cpus, physical_cpus)
    if explicit_cpus is None:
        thread_counts = _parse_threads(args.threads, len(physical_cpus))
    else:
        thread_counts = [len(explicit_cpus)]

    env_path = root / "data" / "environment.zarr"
    if not env_path.exists():
        raise FileNotFoundError(f"Expected prepared raster at {env_path}")

    env = xr.open_zarr(env_path, chunks=None)["environment"]
    blocks = _load_raw_blocks(
        env,
        args.block_size,
        args.resident_blocks,
        args.compute_dtype,
    )

    metadata = {
        "experiment": "raw_numpy_reused_workspace_pinned_threads_v2",
        "question": "How does explicit per-thread CPU/cache-domain placement affect scaling?",
        "physical_cpu_budget": len(physical_cpus),
        "physical_cpus": physical_cpus,
        "allowed_logical_cpus": allowed,
        "topology": topology,
        "thread_counts": thread_counts,
        "explicit_cpus": explicit_cpus,
        "block_size": args.block_size,
        "resident_blocks": args.resident_blocks,
        "tasks_per_observation": args.tasks,
        "repeats": args.repeats,
        "compute_dtype": args.compute_dtype,
        "per_thread_pinning": True,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )

    with _affinity(physical_cpus):
        for threads in thread_counts:
            selected_cpus = explicit_cpus if explicit_cpus is not None else physical_cpus[:threads]
            if len(selected_cpus) != threads:
                raise RuntimeError("Selected CPU count does not match worker-thread count")

            pinning = _ThreadPinRegistry(selected_cpus)
            workspaces = _WorkspaceRegistry()

            print(f"\n=== {threads} pinned thread{'s' if threads != 1 else ''} ===")
            print("CPUs:", selected_cpus)

            with ThreadPoolExecutor(
                max_workers=threads,
                thread_name_prefix="hrhsa-pinned",
                initializer=pinning.initializer,
            ) as executor:
                if args.warmup_tasks:
                    _run_batch(
                        executor,
                        blocks,
                        args.compute_dtype,
                        workspaces,
                        args.warmup_tasks,
                    )

                for repeat in range(1, args.repeats + 1):
                    process_before = process_runtime_snapshot()
                    system_before = system_runtime_snapshot()
                    start = time.perf_counter()
                    checksum = _run_batch(
                        executor,
                        blocks,
                        args.compute_dtype,
                        workspaces,
                        args.tasks,
                    )
                    wall = time.perf_counter() - start
                    system_after = system_runtime_snapshot()
                    process_after = process_runtime_snapshot()

                    process = process_runtime_delta(process_before, process_after)
                    system = system_runtime_delta(system_before, system_after)
                    cpu_seconds = process.get("cpu_total_seconds")

                    payload = {
                        "threads": threads,
                        "selected_cpus": selected_cpus,
                        "repeat": repeat,
                        "wall_seconds": wall,
                        "tasks": args.tasks,
                        "tasks_per_second": args.tasks / wall,
                        "checksum": checksum,
                        "process_cpu_seconds": cpu_seconds,
                        "mean_busy_cores": None if cpu_seconds is None else cpu_seconds / wall,
                        "requested_thread_utilization_fraction": (
                            None
                            if cpu_seconds is None
                            else cpu_seconds / (wall * threads)
                        ),
                        "process_context_switches": process.get("context_switches"),
                        "system_cpu_busy_fraction": system.get("cpu_busy_fraction"),
                        "system_iowait_fraction": system.get("cpu_iowait_fraction"),
                        "thread_cpu_mapping": pinning.mapping(),
                        "workspaces_created": workspaces.workspaces_created,
                    }
                    _append_jsonl(output, payload)
                    print(
                        f"repeat {repeat}: {wall:.3f} s | "
                        f"{args.tasks / wall:.1f} blocks/s | "
                        f"busy cores={payload['mean_busy_cores']}"
                    )

            print("Pinned mapping:", pinning.mapping())


if __name__ == "__main__":
    main()
