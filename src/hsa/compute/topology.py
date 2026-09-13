"""Runtime CPU/NUMA/cache discovery and Dask placement validation."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SYSFS = Path("/sys/devices/system/cpu")


@dataclass(frozen=True)
class RuntimeTopology:
    hostname: str
    pid: int
    cpu_affinity: tuple[int, ...]
    affinity_cpu_count: int
    physical_core_count: int | None
    socket_ids: tuple[int, ...]
    numa_node_ids: tuple[int, ...]
    cpu_model: str | None
    memory_total_gib: float | None
    slurm_procid: str | None
    slurm_localid: str | None
    slurm_nodeid: str | None
    dask_worker_name: str | None = None
    dask_nthreads: int | None = None
    dask_memory_limit_gib: float | None = None
    local_directory: str | None = None
    physical_cpu_representatives: tuple[int, ...] = ()
    physical_core_ids: tuple[tuple[int, int], ...] = ()
    llc_level: int | None = None
    llc_size_bytes: int | None = None
    llc_domains: tuple[tuple[int, ...], ...] = ()

    @property
    def llc_domain_count(self) -> int:
        return len(self.llc_domains)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("cpu_affinity", "socket_ids", "numa_node_ids", "physical_cpu_representatives"):
            out[key] = list(out[key])
        out["physical_core_ids"] = [list(v) for v in self.physical_core_ids]
        out["llc_domains"] = [list(v) for v in self.llc_domains]
        out["llc_domain_count"] = self.llc_domain_count
        return out


@dataclass(frozen=True)
class TopologyValidation:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    workers_per_host: dict[str, int]
    unique_cpus_per_host: dict[str, int]
    unique_physical_cores_per_host: dict[str, int]
    affinity_sizes: tuple[int, ...]
    physical_core_counts: tuple[int, ...]
    observed_workers: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings),
            "workers_per_host": dict(self.workers_per_host),
            "unique_cpus_per_host": dict(self.unique_cpus_per_host),
            "unique_physical_cores_per_host": dict(self.unique_physical_cores_per_host),
            "affinity_sizes": list(self.affinity_sizes),
            "physical_core_counts": list(self.physical_core_counts),
            "observed_workers": self.observed_workers,
        }

    def explain(self) -> str:
        rows = [
            f"topology ok                : {self.ok}",
            f"observed workers           : {self.observed_workers}",
            f"workers / host             : {self.workers_per_host}",
            f"logical CPUs / host        : {self.unique_cpus_per_host}",
            f"physical cores / host      : {self.unique_physical_cores_per_host}",
            f"logical affinity sizes     : {list(self.affinity_sizes)}",
            f"physical cores / worker    : {list(self.physical_core_counts)}",
        ]
        rows += [f"ERROR   : {v}" for v in self.errors]
        rows += [f"WARNING : {v}" for v in self.warnings]
        return "\n".join(rows)


def _cpu_affinity() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return tuple(range(os.cpu_count() or 1))


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    out: set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            if b < a:
                raise ValueError(value)
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return tuple(sorted(out))


def _sysfs_topology_rows(cpus: tuple[int, ...], *, sysfs_root: Path = _SYSFS):
    """Read CPU/core/socket identities without spawning subprocesses."""
    rows = []
    for cpu in cpus:
        root = sysfs_root / f"cpu{cpu}"
        try:
            core = int((root / "topology/core_id").read_text().strip())
            socket = int((root / "topology/physical_package_id").read_text().strip())
        except (OSError, ValueError):
            continue
        node = -1
        try:
            for path in root.glob("node[0-9]*"):
                m = re.fullmatch(r"node(\d+)", path.name)
                if m:
                    node = int(m.group(1)); break
        except OSError:
            pass
        rows.append((int(cpu), core, socket, node))
    return rows


def _lscpu_rows():
    """Fallback for non-Linux environments where CPU sysfs is unavailable."""
    exe = shutil.which("lscpu")
    if exe is None:
        return []
    try:
        p = subprocess.run([exe, "-p=CPU,CORE,SOCKET,NODE"], check=True, capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    rows = []
    for line in p.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(tuple(map(int, line.split(","))))
        except ValueError:
            continue
    return [r for r in rows if len(r) == 4]


def _physical_core_ids(rows):
    return tuple(sorted({(socket, core) for _, core, socket, _ in rows}))


def _physical_cpu_representatives(rows):
    seen, out = set(), []
    for cpu, core, socket, _ in sorted(rows):
        key = (socket, core)
        if key not in seen:
            seen.add(key); out.append(cpu)
    return tuple(out)


def _parse_cache_size_bytes(value: str) -> int:
    m = re.fullmatch(r"\s*(\d+)\s*([KMG]?)\s*", value, re.I)
    if not m:
        raise ValueError(value)
    return int(m.group(1)) * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m.group(2).upper()]


def _last_level_cache_domains(cpus: tuple[int, ...], *, sysfs_root: Path = _SYSFS):
    allowed = set(cpus); records = []
    for cpu in sorted(allowed):
        try:
            indexes = list((sysfs_root / f"cpu{cpu}/cache").glob("index*"))
        except OSError:
            continue
        for index in indexes:
            try:
                level = int((index / "level").read_text().strip())
                kind = (index / "type").read_text().strip().lower()
                shared = set(_parse_cpu_list((index / "shared_cpu_list").read_text()))
                size = _parse_cache_size_bytes((index / "size").read_text())
            except (OSError, ValueError):
                continue
            domain = tuple(sorted(allowed & shared))
            if kind in {"data", "unified"} and domain:
                records.append((level, size, domain))
    if not records:
        return None, None, ()
    highest = max(v[0] for v in records)
    selected = [(size, domain) for level, size, domain in records if level == highest]
    domains = tuple(sorted({d for _, d in selected}, key=lambda d: (d[0], len(d), d)))
    return highest, min(size for size, _ in selected), domains


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor().strip() or None


def _memory_total_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    return None


def discover_runtime_topology(*, sysfs_root: Path = _SYSFS) -> RuntimeTopology:
    affinity = _cpu_affinity()
    rows = _sysfs_topology_rows(affinity, sysfs_root=sysfs_root)
    if not rows:
        allowed = set(affinity)
        rows = [r for r in _lscpu_rows() if r[0] in allowed]
    ids = _physical_core_ids(rows) if rows else ()
    reps = _physical_cpu_representatives(rows) if rows else ()
    sockets = tuple(sorted({r[2] for r in rows if r[2] >= 0}))
    nodes = tuple(sorted({r[3] for r in rows if r[3] >= 0}))
    llc_level, llc_size, llc_domains = _last_level_cache_domains(affinity, sysfs_root=sysfs_root)

    name = None; nthreads = None; memory = None; local = None
    try:
        from dask.distributed import get_worker
        w = get_worker(); name = str(w.name); nthreads = int(w.nthreads)
        if getattr(w, "memory_limit", None) is not None:
            memory = float(w.memory_limit) / 1024**3
        local = str(getattr(w, "local_directory", "")) or None
    except Exception:
        pass

    return RuntimeTopology(
        hostname=platform.node() or os.environ.get("HOSTNAME", "unknown"), pid=os.getpid(),
        cpu_affinity=affinity, affinity_cpu_count=len(affinity),
        physical_core_count=len(ids) if ids else None, socket_ids=sockets, numa_node_ids=nodes,
        cpu_model=_cpu_model(), memory_total_gib=_memory_total_gib(),
        slurm_procid=os.environ.get("SLURM_PROCID"), slurm_localid=os.environ.get("SLURM_LOCALID"),
        slurm_nodeid=os.environ.get("SLURM_NODEID"), dask_worker_name=name, dask_nthreads=nthreads,
        dask_memory_limit_gib=memory, local_directory=local,
        physical_cpu_representatives=reps, physical_core_ids=ids,
        llc_level=llc_level, llc_size_bytes=llc_size, llc_domains=llc_domains,
    )


def spread_physical_cpus_across_llc(topology: RuntimeTopology, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    reps = list(topology.physical_cpu_representatives or topology.cpu_affinity)
    if count > len(reps):
        raise ValueError(f"Requested {count} physical CPUs but only {len(reps)} representatives are visible")
    if not topology.llc_domains:
        return tuple(reps[:count])
    buckets, assigned = [], set()
    for domain in topology.llc_domains:
        bucket = [cpu for cpu in reps if cpu in set(domain)]
        if bucket:
            buckets.append(bucket); assigned.update(bucket)
    leftovers = [cpu for cpu in reps if cpu not in assigned]
    if leftovers:
        buckets.append(leftovers)
    ordered, depth = [], 0
    while len(ordered) < count:
        added = False
        for bucket in buckets:
            if depth < len(bucket):
                ordered.append(bucket[depth]); added = True
                if len(ordered) == count:
                    break
        if not added:
            break
        depth += 1
    if len(ordered) < count:
        raise RuntimeError("Could not construct a complete LLC-spread CPU order")
    return tuple(ordered)


def _runtime_topology_dict():
    return discover_runtime_topology().as_dict()


def collect_worker_topology(client):
    if client is None:
        return {}
    observed = {str(k): dict(v) for k, v in client.run(_runtime_topology_dict).items()}
    try:
        scheduler_workers = dict((client.scheduler_info() or {}).get("workers") or {})
    except Exception:
        scheduler_workers = {}
    for address, facts in observed.items():
        meta = dict(scheduler_workers.get(address) or {})
        if meta.get("nthreads") is not None:
            facts["dask_nthreads"] = int(meta["nthreads"])
        if meta.get("name") is not None:
            facts["dask_worker_name"] = str(meta["name"])
        if meta.get("memory_limit") is not None:
            facts["dask_memory_limit_gib"] = float(meta["memory_limit"]) / 1024**3
    return observed


def _physical_ids_from_facts(facts):
    raw = facts.get("physical_core_ids")
    if not raw:
        return None
    try:
        return {(int(v[0]), int(v[1])) for v in raw}
    except (TypeError, ValueError, IndexError):
        return None


def validate_worker_topology(client, plan, *, strict_affinity: bool = False) -> TopologyValidation:
    workers = collect_worker_topology(client); errors, warnings = [], []
    expected_total = int(plan.total_workers)
    if len(workers) != expected_total:
        errors.append(f"expected {expected_total} workers, observed {len(workers)}")
    by_host = defaultdict(list)
    for facts in workers.values():
        by_host[str(facts.get("hostname", "unknown"))].append(facts)
    workers_per_host = {h: len(v) for h, v in sorted(by_host.items())}
    if len(by_host) != int(plan.nodes):
        errors.append(f"expected {plan.nodes} worker host(s), observed {len(by_host)}")
    expected_per_host = int(plan.geometry.workers)
    for host, items in by_host.items():
        if len(items) != expected_per_host:
            errors.append(f"{host}: expected {expected_per_host} workers, observed {len(items)}")

    expected_threads = int(plan.geometry.threads_per_worker)
    thread_counts = Counter(int(f["dask_nthreads"]) for f in workers.values() if f.get("dask_nthreads") is not None)
    if thread_counts and set(thread_counts) != {expected_threads}:
        errors.append(f"expected {expected_threads} Dask threads/worker, observed {dict(thread_counts)}")
    elif not thread_counts:
        (errors if strict_affinity else warnings).append("Dask thread counts unavailable from worker/scheduler metadata")

    affinity_sizes = tuple(sorted({int(f["affinity_cpu_count"]) for f in workers.values() if f.get("affinity_cpu_count") is not None}))
    physical_counts = tuple(sorted({int(f["physical_core_count"]) for f in workers.values() if f.get("physical_core_count") is not None}))
    have_counts = sum(f.get("physical_core_count") is not None for f in workers.values())
    if workers and have_counts == len(workers):
        if physical_counts != (expected_threads,):
            (errors if strict_affinity else warnings).append(
                f"expected {expected_threads} physical cores per worker, observed {list(physical_counts)}"
            )
    else:
        (errors if strict_affinity else warnings).append(
            f"physical core counts unavailable for {len(workers)-have_counts}/{len(workers)} workers; logical affinity is provenance only"
        )

    unique_logical, unique_physical = {}, {}
    for host, items in by_host.items():
        logical_sets = [set(map(int, f.get("cpu_affinity", []))) for f in items]
        logical_union = set().union(*logical_sets) if logical_sets else set()
        unique_logical[host] = len(logical_union)
        physical_sets = [_physical_ids_from_facts(f) for f in items]
        resolved = [s for s in physical_sets if s is not None]
        if len(resolved) == len(physical_sets) and physical_sets:
            union = set().union(*resolved); unique_physical[host] = len(union)
            overlap = sum(map(len, resolved)) - len(union)
            if overlap:
                (errors if strict_affinity else warnings).append(f"{host}: worker physical-core sets overlap by {overlap} core assignments")
            if len(items) == expected_per_host and len(union) != int(plan.cores_per_node):
                (errors if strict_affinity else warnings).append(
                    f"{host}: workers cover {len(union)} physical cores, plan requests {plan.cores_per_node}"
                )
        else:
            missing = len(physical_sets) - len(resolved)
            (errors if strict_affinity else warnings).append(
                f"{host}: physical core identities unavailable for {missing}/{len(items)} workers; cannot prove non-overlap/coverage"
            )
            overlap = sum(map(len, logical_sets)) - len(logical_union)
            if overlap:
                warnings.append(f"{host}: logical CPU affinity sets overlap by {overlap} assignments")

    return TopologyValidation(
        ok=not errors, errors=tuple(errors), warnings=tuple(warnings), workers_per_host=workers_per_host,
        unique_cpus_per_host=unique_logical, unique_physical_cores_per_host=unique_physical,
        affinity_sizes=affinity_sizes, physical_core_counts=physical_counts, observed_workers=len(workers),
    )
