"""Small, dependency-light utilities for reproducible hrHSA performance runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class BenchmarkRecord:
    """One machine-readable performance observation."""

    benchmark: str
    wall_seconds: float
    rows: int | None = None
    workers: int | None = None
    threads_per_worker: int | None = None
    chunk_mb: int | None = None
    tasks: int | None = None
    tasks_per_worker: float | None = None
    bytes_processed: int | None = None
    throughput_rows_s: float | None = None
    throughput_mb_s: float | None = None
    peak_rss_mb: float | None = None
    current_rss_mb: float | None = None
    operation_peak_rss_mb: float | None = None
    operation_peak_process_tree_rss_mb: float | None = None
    worker_peak_rss_total_mb: float | None = None
    worker_peak_rss_max_mb: float | None = None
    worker_current_rss_total_mb: float | None = None
    worker_current_rss_max_mb: float | None = None
    worker_operation_peak_rss_total_mb: float | None = None
    worker_operation_peak_rss_max_mb: float | None = None
    memory_samples: int | None = None
    git_commit: str | None = None
    hostname: str | None = None
    python: str | None = None
    platform: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_git_commit() -> str | None:
    """Return the benchmark source commit.

    Batch launchers may inject ``HRHSA_GIT_COMMIT`` at submission time.  This is
    preferred on HPC systems because compute nodes need not provide the ``git``
    executable and Slurm jobs deliberately use ``--export=NONE``.  Interactive
    and workstation runs fall back to querying the current checkout directly.
    """
    injected = os.environ.get("HRHSA_GIT_COMMIT", "").strip()
    if injected:
        return injected

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def peak_rss_mb() -> float | None:
    """Return process lifetime peak RSS in MiB on Unix-like systems."""
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() == "Darwin":
            return value / 1024**2
        return value / 1024
    except Exception:
        return None


def current_rss_mb() -> float | None:
    """Return current process RSS in MiB when psutil is available."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / 1024**2
    except Exception:
        return None


def process_tree_rss_mb() -> float | None:
    """Return current RSS of this process plus all live descendants in MiB."""
    try:
        import psutil

        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        total = 0
        observed = False
        for process in processes:
            try:
                total += int(process.memory_info().rss)
                observed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return float(total) / 1024**2 if observed else None
    except Exception:
        return None


def _worker_peak_rss_mb() -> float | None:
    return peak_rss_mb()


def _worker_current_rss_mb() -> float | None:
    return current_rss_mb()


def _aggregate_worker_metric(client, function) -> tuple[float | None, float | None]:
    if client is None:
        return None, None
    try:
        observed = client.run(function)
        values = [float(value) for value in observed.values() if value is not None]
        if not values:
            return None, None
        return float(sum(values)), float(max(values))
    except Exception:
        return None, None


def distributed_worker_peak_rss(client) -> tuple[float | None, float | None]:
    """Return total and maximum lifetime peak RSS across active Dask workers."""
    return _aggregate_worker_metric(client, _worker_peak_rss_mb)


def distributed_worker_current_rss(client) -> tuple[float | None, float | None]:
    """Return total and maximum current RSS across active Dask workers."""
    return _aggregate_worker_metric(client, _worker_current_rss_mb)


def distributed_worker_scheduler_rss(client) -> tuple[float | None, float | None]:
    """Read worker RSS from scheduler heartbeat metrics without worker RPCs.

    Dask workers already report process memory in their normal heartbeats.  Using
    those cached scheduler metrics for high-frequency benchmark monitoring avoids
    calling ``client.run`` on every worker at every sample, which can materially
    perturb large distributed timing runs and produces noisy out-of-band logs.
    """
    if client is None:
        return None, None
    try:
        info = client.scheduler_info()
        values = []
        for worker in info.get("workers", {}).values():
            value = worker.get("metrics", {}).get("memory")
            if value is not None:
                values.append(float(value) / 1024**2)
        if not values:
            return None, None
        return float(sum(values)), float(max(values))
    except Exception:
        return None, None


def _max_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return float(candidate)
    return max(float(current), float(candidate))


@contextmanager
def operation_memory_monitor(*, client=None, interval_seconds: float = 0.25):
    """Sample operation-local driver, child-process and Dask-worker RSS.

    Process-lifetime peak RSS cannot be reset between benchmark stages. This
    monitor samples current RSS while one operation is running so the resulting
    maxima can be compared across scaling points. The process-tree metric captures
    child processes such as PyMC chain workers. Remote Dask-worker RSS is read from
    scheduler heartbeat metrics so monitoring does not inject repeated out-of-band
    function calls into every worker during the timed region.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    state: dict[str, Any] = {
        "operation_peak_rss_mb": None,
        "operation_peak_process_tree_rss_mb": None,
        "worker_operation_peak_rss_total_mb": None,
        "worker_operation_peak_rss_max_mb": None,
        "memory_samples": 0,
        "memory_sample_interval_seconds": float(interval_seconds),
    }
    stop = threading.Event()

    def sample() -> None:
        state["operation_peak_rss_mb"] = _max_optional(
            state["operation_peak_rss_mb"], current_rss_mb()
        )
        state["operation_peak_process_tree_rss_mb"] = _max_optional(
            state["operation_peak_process_tree_rss_mb"], process_tree_rss_mb()
        )
        worker_total, worker_max = distributed_worker_scheduler_rss(client)
        state["worker_operation_peak_rss_total_mb"] = _max_optional(
            state["worker_operation_peak_rss_total_mb"], worker_total
        )
        state["worker_operation_peak_rss_max_mb"] = _max_optional(
            state["worker_operation_peak_rss_max_mb"], worker_max
        )
        state["memory_samples"] += 1

    def monitor() -> None:
        sample()
        while not stop.wait(interval_seconds):
            sample()

    thread = threading.Thread(
        target=monitor,
        name="hrhsa-memory-monitor",
        daemon=True,
    )
    thread.start()
    try:
        yield state
    finally:
        stop.set()
        thread.join()
        sample()


@contextmanager
def benchmark_timer(*, client=None, memory_interval_seconds: float = 0.25):
    """Time one operation while collecting operation-local memory maxima."""
    state: dict[str, Any] = {}
    start = time.perf_counter()
    with operation_memory_monitor(
        client=client,
        interval_seconds=memory_interval_seconds,
    ) as memory:
        try:
            yield state
        finally:
            state["wall_seconds"] = time.perf_counter() - start
    state.update(memory)


def _runtime_metadata(metadata: dict[str, Any] | None, *, client=None) -> dict[str, Any] | None:
    payload = dict(metadata or {})
    for name in (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_PARTITION",
    ):
        value = os.environ.get(name)
        if value is not None:
            payload.setdefault(name.lower(), value)

    if client is not None:
        try:
            info = client.scheduler_info()
            payload.setdefault("observed_workers", len(info.get("workers", {})))
        except Exception:
            pass
    return payload or None


@contextmanager
def wall_timer():
    """Context manager exposing elapsed wall time through a mutable mapping."""
    state: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["wall_seconds"] = time.perf_counter() - start


def make_benchmark_record(
    benchmark: str,
    wall_seconds: float,
    *,
    rows: int | None = None,
    workers: int | None = None,
    threads_per_worker: int | None = None,
    chunk_mb: int | None = None,
    tasks: int | None = None,
    bytes_processed: int | None = None,
    metadata: dict[str, Any] | None = None,
    operation_memory: dict[str, Any] | None = None,
    client=None,
) -> BenchmarkRecord:
    """Construct a benchmark record with environment metadata and throughput."""
    throughput_rows_s = None
    if rows is not None and wall_seconds > 0:
        throughput_rows_s = float(rows) / float(wall_seconds)
    throughput_mb_s = None
    if bytes_processed is not None and wall_seconds > 0:
        throughput_mb_s = bytes_processed / 1024**2 / float(wall_seconds)

    tasks_per_worker = None
    if tasks is not None and workers is not None and workers > 0:
        tasks_per_worker = float(tasks) / float(workers)

    worker_peak_total, worker_peak_max = distributed_worker_peak_rss(client)
    worker_current_total, worker_current_max = distributed_worker_current_rss(client)
    memory = dict(operation_memory or {})

    return BenchmarkRecord(
        benchmark=benchmark,
        wall_seconds=float(wall_seconds),
        rows=None if rows is None else int(rows),
        workers=workers,
        threads_per_worker=threads_per_worker,
        chunk_mb=chunk_mb,
        tasks=None if tasks is None else int(tasks),
        tasks_per_worker=tasks_per_worker,
        bytes_processed=bytes_processed,
        throughput_rows_s=throughput_rows_s,
        throughput_mb_s=throughput_mb_s,
        peak_rss_mb=peak_rss_mb(),
        current_rss_mb=current_rss_mb(),
        operation_peak_rss_mb=memory.get("operation_peak_rss_mb"),
        operation_peak_process_tree_rss_mb=memory.get(
            "operation_peak_process_tree_rss_mb"
        ),
        worker_peak_rss_total_mb=worker_peak_total,
        worker_peak_rss_max_mb=worker_peak_max,
        worker_current_rss_total_mb=worker_current_total,
        worker_current_rss_max_mb=worker_current_max,
        worker_operation_peak_rss_total_mb=memory.get(
            "worker_operation_peak_rss_total_mb"
        ),
        worker_operation_peak_rss_max_mb=memory.get(
            "worker_operation_peak_rss_max_mb"
        ),
        memory_samples=memory.get("memory_samples"),
        git_commit=current_git_commit(),
        hostname=platform.node() or os.environ.get("HOSTNAME"),
        python=platform.python_version(),
        platform=platform.platform(),
        metadata=_runtime_metadata(metadata, client=client),
    )


def append_benchmark_record(
    record: BenchmarkRecord | dict[str, Any],
    path: str | Path,
) -> None:
    """Append one JSON record to a JSONL benchmark log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.to_dict() if isinstance(record, BenchmarkRecord) else dict(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def read_benchmark_records(path: str | Path) -> pd.DataFrame:
    """Read a JSONL benchmark log into a dataframe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.json_normalize(records)


def scaling_table(
    results: pd.DataFrame,
    *,
    workers_col: str = "workers",
    time_col: str = "wall_seconds",
) -> pd.DataFrame:
    """Calculate strong-scaling speedup and parallel efficiency."""
    if workers_col not in results or time_col not in results:
        raise KeyError(f"results must contain {workers_col!r} and {time_col!r}.")
    out = results.sort_values(workers_col).copy()
    baseline_workers = float(out.iloc[0][workers_col])
    baseline_time = float(out.iloc[0][time_col])
    out["speedup"] = baseline_time / out[time_col].astype(float)
    relative_workers = out[workers_col].astype(float) / baseline_workers
    out["parallel_efficiency"] = out["speedup"] / relative_workers
    return out