"""Telemetry helpers for capacity/strong/weak scaling campaigns.

This module builds on :mod:`hsa.compute.telemetry` with two pieces that matter
especially for out-of-core raster workloads:

* Linux VM/page-reclaim counters from ``/proc/vmstat``;
* one system-wide snapshot per Dask worker host, de-duplicated by hostname.

All helpers are best-effort. Missing counters remain absent rather than making a
benchmark fail, which keeps the same code usable on workstations and HPC nodes.
"""

from __future__ import annotations

import math
import socket
from pathlib import Path
from typing import Any

from hsa.compute.telemetry import (
    process_runtime_delta,
    process_runtime_snapshot,
    system_runtime_delta,
    system_runtime_snapshot,
)


_VMSTAT_FIELDS = (
    "pgfault",
    "pgmajfault",
    "pgscan_kswapd",
    "pgscan_direct",
    "pgsteal_kswapd",
    "pgsteal_direct",
    "pswpin",
    "pswpout",
    "workingset_refault_anon",
    "workingset_refault_file",
    "workingset_activate_anon",
    "workingset_activate_file",
)


def _safe_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def linux_vmstat_snapshot(path: str | Path = "/proc/vmstat") -> dict[str, Any]:
    """Read selected monotonic Linux VM/page-reclaim counters."""

    source = Path(path)
    try:
        observed: dict[str, int] = {}
        for line in source.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition(" ")
            if name not in _VMSTAT_FIELDS:
                continue
            try:
                observed[name] = int(value.strip())
            except ValueError:
                continue
        return {
            "available": bool(observed),
            "path": str(source),
            "counters": observed,
        }
    except Exception as exc:
        return {
            "available": False,
            "path": str(source),
            "reason": f"vmstat snapshot failed: {exc}",
            "counters": {},
        }


def linux_vmstat_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    """Subtract monotonic ``/proc/vmstat`` counters and add useful totals."""

    before = dict(before or {})
    after = dict(after or {})
    left = dict(before.get("counters") or {})
    right = dict(after.get("counters") or {})
    counters: dict[str, float] = {}

    for field in sorted(set(left).intersection(right)):
        a = _safe_number(left.get(field))
        b = _safe_number(right.get(field))
        if a is not None and b is not None:
            counters[field] = max(0.0, b - a)

    payload: dict[str, Any] = {
        "available": bool(counters),
        "counters": counters,
    }
    if not counters:
        return payload

    payload["page_faults"] = counters.get("pgfault")
    payload["major_page_faults"] = counters.get("pgmajfault")
    payload["swap_pages_in"] = counters.get("pswpin")
    payload["swap_pages_out"] = counters.get("pswpout")
    payload["page_scans"] = sum(
        value for key, value in counters.items() if key.startswith("pgscan_")
    )
    payload["page_steals"] = sum(
        value for key, value in counters.items() if key.startswith("pgsteal_")
    )
    payload["workingset_refaults"] = sum(
        value for key, value in counters.items() if key.startswith("workingset_refault_")
    )
    return payload


def scheduler_process_snapshot(dask_scheduler=None) -> dict[str, Any]:
    """Collect process telemetry inside the Dask scheduler process."""

    payload = process_runtime_snapshot()
    payload["hostname"] = socket.gethostname()
    return payload


def scheduler_process_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Return scheduler-process deltas with a one-core utilization fraction."""

    payload = process_runtime_delta(before, after)
    if after:
        payload["hostname"] = after.get("hostname")
    cpu = _safe_number(payload.get("cpu_total_seconds"))
    wall = _safe_number(wall_seconds)
    if cpu is not None and wall is not None and wall > 0:
        payload["cpu_fraction_of_one_core"] = cpu / wall
    return payload


def node_runtime_snapshot() -> dict[str, Any]:
    """Snapshot node-wide counters from inside a Dask worker process."""

    return {
        "hostname": socket.gethostname(),
        "system": system_runtime_snapshot(),
        "vmstat": linux_vmstat_snapshot(),
    }


def distributed_node_runtime_snapshot(client) -> dict[str, dict[str, Any]]:
    """Return one node-wide snapshot per worker host.

    ``Client.run`` necessarily executes on every worker. Results are sorted by
    worker key and only the first observation for each hostname is retained, so
    node-wide counters are never multiplied by the number of worker processes.
    """

    if client is None:
        return {}
    try:
        raw = dict(client.run(node_runtime_snapshot))
    except Exception:
        return {}

    by_host: dict[str, dict[str, Any]] = {}
    for worker, payload in sorted(raw.items()):
        hostname = str(payload.get("hostname") or worker)
        by_host.setdefault(hostname, payload)
    return by_host


def distributed_node_runtime_delta(
    before: dict[str, dict[str, Any]] | None,
    after: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Compute per-host and aggregate node-wide runtime deltas."""

    before = dict(before or {})
    after = dict(after or {})
    matched = sorted(set(before).intersection(after))
    per_host: dict[str, dict[str, Any]] = {}

    for host in matched:
        per_host[host] = {
            "system": system_runtime_delta(
                before[host].get("system"),
                after[host].get("system"),
            ),
            "vmstat": linux_vmstat_delta(
                before[host].get("vmstat"),
                after[host].get("vmstat"),
            ),
        }

    payload: dict[str, Any] = {
        "available": bool(per_host),
        "hosts_before": sorted(before),
        "hosts_after": sorted(after),
        "hosts_matched": matched,
        "per_host": per_host,
    }
    if not per_host:
        return payload

    def sum_system(field: str) -> float | None:
        values = [
            _safe_number(item["system"].get(field))
            for item in per_host.values()
        ]
        values = [value for value in values if value is not None]
        return None if not values else float(sum(values))

    def mean_system(field: str) -> float | None:
        values = [
            _safe_number(item["system"].get(field))
            for item in per_host.values()
        ]
        values = [value for value in values if value is not None]
        return None if not values else float(sum(values) / len(values))

    aggregate: dict[str, Any] = {}
    for field in (
        "disk_read_bytes",
        "disk_write_bytes",
        "disk_read_count",
        "disk_write_count",
        "ctx_switches",
        "interrupts",
        "soft_interrupts",
    ):
        value = sum_system(field)
        if value is not None:
            aggregate[field] = value

    for field in ("cpu_busy_fraction", "cpu_iowait_fraction"):
        value = mean_system(field)
        if value is not None:
            aggregate[f"mean_{field}"] = value

    vm_fields = (
        "page_faults",
        "major_page_faults",
        "swap_pages_in",
        "swap_pages_out",
        "page_scans",
        "page_steals",
        "workingset_refaults",
    )
    for field in vm_fields:
        values = [
            _safe_number(item["vmstat"].get(field))
            for item in per_host.values()
        ]
        values = [value for value in values if value is not None]
        if values:
            aggregate[field] = float(sum(values))

    payload["aggregate"] = aggregate
    return payload
