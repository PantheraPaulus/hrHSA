"""Low-overhead runtime telemetry for mechanism-oriented performance benchmarks.

The helpers in this module deliberately collect counters rather than trying to be a
full profiler. They are cheap enough to wrap repeated benchmark observations and
portable enough to reuse on a workstation or inside an HPC allocation.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Iterable


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _asdict(value: Any) -> dict[str, Any]:
    try:
        return dict(value._asdict())
    except Exception:
        return {}


def process_runtime_snapshot(pid: int | None = None) -> dict[str, Any]:
    """Snapshot inexpensive counters for one process.

    psutil is imported lazily so importing :mod:`hsa.compute` does not make it a
    hard dependency for users who never request telemetry. The function is also
    safe to send to Dask workers with ``Client.run``.
    """

    try:
        import psutil
    except Exception as exc:
        return {"available": False, "reason": f"psutil unavailable: {exc}"}

    try:
        process = psutil.Process(pid)
        with process.oneshot():
            cpu = process.cpu_times()
            ctx = process.num_ctx_switches()
            memory = process.memory_info()
            try:
                full_memory = process.memory_full_info()
            except (psutil.AccessDenied, AttributeError, NotImplementedError):
                full_memory = None
            try:
                io = process.io_counters()
            except (psutil.AccessDenied, AttributeError, NotImplementedError):
                io = None
            try:
                affinity = process.cpu_affinity()
            except (psutil.AccessDenied, AttributeError, NotImplementedError):
                affinity = None

            payload: dict[str, Any] = {
                "available": True,
                "pid": process.pid,
                "cpu_user_seconds": float(cpu.user),
                "cpu_system_seconds": float(cpu.system),
                "cpu_total_seconds": float(cpu.user + cpu.system),
                "voluntary_context_switches": int(ctx.voluntary),
                "involuntary_context_switches": int(ctx.involuntary),
                "context_switches": int(ctx.voluntary + ctx.involuntary),
                "rss_bytes": int(memory.rss),
                "vms_bytes": int(memory.vms),
                "num_threads": int(process.num_threads()),
                "cpu_affinity": None if affinity is None else sorted(int(v) for v in affinity),
            }
            if full_memory is not None:
                payload["uss_bytes"] = int(getattr(full_memory, "uss", 0))
                payload["pss_bytes"] = int(getattr(full_memory, "pss", 0))
            if io is not None:
                payload.update(
                    {
                        "read_bytes": int(getattr(io, "read_bytes", 0)),
                        "write_bytes": int(getattr(io, "write_bytes", 0)),
                        "read_count": int(getattr(io, "read_count", 0)),
                        "write_count": int(getattr(io, "write_count", 0)),
                    }
                )
            return payload
    except Exception as exc:
        return {"available": False, "reason": f"process snapshot failed: {exc}"}


def distributed_worker_runtime_snapshot(client) -> dict[str, dict[str, Any]]:
    """Return :func:`process_runtime_snapshot` for every active Dask worker."""

    if client is None:
        return {}
    try:
        return dict(client.run(process_runtime_snapshot))
    except Exception:
        return {}


_PROCESS_COUNTERS = (
    "cpu_user_seconds",
    "cpu_system_seconds",
    "cpu_total_seconds",
    "voluntary_context_switches",
    "involuntary_context_switches",
    "context_switches",
    "read_bytes",
    "write_bytes",
    "read_count",
    "write_count",
)


def process_runtime_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    """Subtract monotonic process counters and retain useful end-state memory."""

    before = dict(before or {})
    after = dict(after or {})
    if not before.get("available") or not after.get("available"):
        return {"available": False}

    payload: dict[str, Any] = {"available": True, "pid": after.get("pid")}
    for field in _PROCESS_COUNTERS:
        left = _safe_float(before.get(field))
        right = _safe_float(after.get(field))
        if left is not None and right is not None:
            payload[field] = max(0.0, right - left)

    for field in ("rss_bytes", "pss_bytes", "uss_bytes", "vms_bytes", "num_threads"):
        if field in after:
            payload[f"end_{field}"] = after[field]

    if "cpu_affinity" in after:
        payload["cpu_affinity"] = after["cpu_affinity"]
    return payload


def aggregate_worker_runtime_delta(
    before: dict[str, dict[str, Any]] | None,
    after: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Aggregate process-counter deltas across Dask workers."""

    before = dict(before or {})
    after = dict(after or {})
    matched = sorted(set(before).intersection(after))
    deltas = [process_runtime_delta(before[key], after[key]) for key in matched]
    valid = [delta for delta in deltas if delta.get("available")]

    payload: dict[str, Any] = {
        "available": bool(valid),
        "workers_before": len(before),
        "workers_after": len(after),
        "workers_matched": len(matched),
    }
    if not valid:
        return payload

    for field in _PROCESS_COUNTERS:
        values = [_safe_float(delta.get(field)) for delta in valid]
        observed = [value for value in values if value is not None]
        if observed:
            payload[field] = float(sum(observed))

    for field in ("end_rss_bytes", "end_pss_bytes", "end_uss_bytes", "end_vms_bytes"):
        values = [_safe_float(delta.get(field)) for delta in valid]
        observed = [value for value in values if value is not None]
        if observed:
            payload[f"{field}_total"] = float(sum(observed))
            payload[f"{field}_max"] = float(max(observed))

    thread_counts = [_safe_float(delta.get("end_num_threads")) for delta in valid]
    observed_threads = [value for value in thread_counts if value is not None]
    if observed_threads:
        payload["end_num_threads_total"] = float(sum(observed_threads))

    payload["affinities"] = {
        key: after[key].get("cpu_affinity")
        for key in matched
        if after[key].get("cpu_affinity") is not None
    }
    return payload


def _temperature_summary(psutil_module) -> dict[str, float] | None:
    try:
        temperatures = psutil_module.sensors_temperatures(fahrenheit=False)
    except (AttributeError, NotImplementedError):
        return None
    except Exception:
        return None

    values: list[float] = []
    for entries in temperatures.values():
        for entry in entries:
            current = _safe_float(getattr(entry, "current", None))
            if current is not None:
                values.append(current)
    if not values:
        return None
    return {
        "temperature_mean_c": float(sum(values) / len(values)),
        "temperature_max_c": float(max(values)),
    }


def system_runtime_snapshot() -> dict[str, Any]:
    """Snapshot system-wide counters that can reveal I/O or scheduling pressure."""

    try:
        import psutil
    except Exception as exc:
        return {"available": False, "reason": f"psutil unavailable: {exc}"}

    payload: dict[str, Any] = {"available": True, "timestamp": time.time()}

    try:
        payload["cpu_times"] = {
            key: float(value) for key, value in _asdict(psutil.cpu_times()).items()
        }
    except Exception:
        payload["cpu_times"] = {}

    try:
        payload["cpu_stats"] = {
            key: float(value) for key, value in _asdict(psutil.cpu_stats()).items()
        }
    except Exception:
        payload["cpu_stats"] = {}

    try:
        disk = psutil.disk_io_counters()
        payload["disk_io"] = (
            {} if disk is None else {key: float(value) for key, value in _asdict(disk).items()}
        )
    except Exception:
        payload["disk_io"] = {}

    try:
        vm = psutil.virtual_memory()
        payload["memory"] = {
            "available_bytes": int(vm.available),
            "used_bytes": int(vm.used),
            "percent": float(vm.percent),
        }
    except Exception:
        payload["memory"] = {}

    try:
        frequency = psutil.cpu_freq(percpu=True)
        current = [_safe_float(getattr(item, "current", None)) for item in (frequency or [])]
        current = [value for value in current if value is not None]
        if current:
            payload["cpu_frequency_mean_mhz"] = float(sum(current) / len(current))
            payload["cpu_frequency_min_mhz"] = float(min(current))
            payload["cpu_frequency_max_mhz"] = float(max(current))
    except Exception:
        pass

    try:
        payload["load_average"] = tuple(float(value) for value in psutil.getloadavg())
    except Exception:
        pass

    temperature = _temperature_summary(psutil)
    if temperature:
        payload.update(temperature)
    return payload


_CPU_TIME_FIELDS = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
)


def _counter_delta(before: dict[str, Any], after: dict[str, Any], field: str) -> float | None:
    left = _safe_float(before.get(field))
    right = _safe_float(after.get(field))
    if left is None or right is None:
        return None
    return max(0.0, right - left)


def system_runtime_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert two system snapshots into operation-local diagnostic deltas."""

    before = dict(before or {})
    after = dict(after or {})
    if not before.get("available") or not after.get("available"):
        return {"available": False}

    payload: dict[str, Any] = {"available": True}
    start = _safe_float(before.get("timestamp"))
    stop = _safe_float(after.get("timestamp"))
    if start is not None and stop is not None:
        payload["elapsed_seconds"] = max(0.0, stop - start)

    cpu_before = dict(before.get("cpu_times") or {})
    cpu_after = dict(after.get("cpu_times") or {})
    cpu_delta: dict[str, float] = {}
    for field in _CPU_TIME_FIELDS:
        delta = _counter_delta(cpu_before, cpu_after, field)
        if delta is not None:
            cpu_delta[field] = delta
    payload["cpu_times_delta"] = cpu_delta

    total = sum(cpu_delta.values())
    idle = cpu_delta.get("idle", 0.0) + cpu_delta.get("iowait", 0.0)
    if total > 0:
        payload["cpu_busy_fraction"] = max(0.0, min(1.0, (total - idle) / total))
        payload["cpu_iowait_fraction"] = max(
            0.0, min(1.0, cpu_delta.get("iowait", 0.0) / total)
        )

    stats_before = dict(before.get("cpu_stats") or {})
    stats_after = dict(after.get("cpu_stats") or {})
    for field in ("ctx_switches", "interrupts", "soft_interrupts", "syscalls"):
        delta = _counter_delta(stats_before, stats_after, field)
        if delta is not None:
            payload[field] = delta

    disk_before = dict(before.get("disk_io") or {})
    disk_after = dict(after.get("disk_io") or {})
    for field in (
        "read_bytes",
        "write_bytes",
        "read_count",
        "write_count",
        "read_time",
        "write_time",
    ):
        delta = _counter_delta(disk_before, disk_after, field)
        if delta is not None:
            payload[f"disk_{field}"] = delta

    for field in (
        "cpu_frequency_mean_mhz",
        "cpu_frequency_min_mhz",
        "cpu_frequency_max_mhz",
        "temperature_mean_c",
        "temperature_max_c",
    ):
        left = _safe_float(before.get(field))
        right = _safe_float(after.get(field))
        if left is not None:
            payload[f"{field}_before"] = left
        if right is not None:
            payload[f"{field}_after"] = right

    memory_before = dict(before.get("memory") or {})
    memory_after = dict(after.get("memory") or {})
    for field in ("available_bytes", "used_bytes", "percent"):
        left = _safe_float(memory_before.get(field))
        right = _safe_float(memory_after.get(field))
        if left is not None:
            payload[f"memory_{field}_before"] = left
        if right is not None:
            payload[f"memory_{field}_after"] = right
    return payload


def summarize_task_stream(events: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    """Summarize Dask task-stream events into interpretable timing counters.

    ``compute_parallelism`` is not CPU utilization. It is summed compute-task time
    divided by the task-stream wall span, so it approximates the average number of
    tasks computing concurrently.
    """

    items = list(events or [])
    action_seconds: dict[str, float] = defaultdict(float)
    compute_durations: list[float] = []
    starts: list[float] = []
    stops: list[float] = []
    bytes_observed = 0.0

    for event in items:
        nbytes = _safe_float(event.get("nbytes"))
        if nbytes is not None:
            bytes_observed += max(0.0, nbytes)

        for interval in event.get("startstops", ()) or ():
            action = str(interval.get("action", "unknown"))
            start = _safe_float(interval.get("start"))
            stop = _safe_float(interval.get("stop"))
            if start is None or stop is None or stop < start:
                continue
            duration = stop - start
            action_seconds[action] += duration
            starts.append(start)
            stops.append(stop)
            if action == "compute":
                compute_durations.append(duration)

    payload: dict[str, Any] = {
        "tasks": len(items),
        "action_seconds": dict(sorted(action_seconds.items())),
        "nbytes_observed": bytes_observed,
        "compute_seconds": action_seconds.get("compute", 0.0),
        "transfer_seconds": action_seconds.get("transfer", 0.0),
        "deserialize_seconds": action_seconds.get("deserialize", 0.0),
    }

    if starts and stops:
        span = max(stops) - min(starts)
        payload["span_seconds"] = span
        if span > 0:
            payload["compute_parallelism"] = payload["compute_seconds"] / span

    if compute_durations:
        ordered = sorted(compute_durations)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        payload["compute_task_mean_seconds"] = float(sum(compute_durations) / len(compute_durations))
        payload["compute_task_median_seconds"] = float(median)
        payload["compute_task_min_seconds"] = float(min(compute_durations))
        payload["compute_task_max_seconds"] = float(max(compute_durations))
    return payload


__all__ = [
    "aggregate_worker_runtime_delta",
    "distributed_worker_runtime_snapshot",
    "process_runtime_delta",
    "process_runtime_snapshot",
    "summarize_task_stream",
    "system_runtime_delta",
    "system_runtime_snapshot",
]
