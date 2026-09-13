"""Capture local workstation topology and numerical runtime for hrHSA benchmarks."""

from __future__ import annotations

import io
import json
import os
import platform
import socket
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import psutil

from hsa.compute.telemetry import system_runtime_snapshot


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return platform.processor() or None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _numpy_runtime() -> str:
    stream = io.StringIO()
    try:
        with redirect_stdout(stream):
            np.show_runtime()
        return stream.getvalue().strip()
    except Exception as exc:
        return f"numpy.show_runtime failed: {exc}"


def _thread_siblings() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu*/topology/thread_siblings_list")):
        try:
            result[path.parts[-3]] = path.read_text().strip()
        except OSError:
            continue
    return result


def _governors() -> list[str]:
    result: set[str] = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
        try:
            result.add(path.read_text().strip())
        except OSError:
            continue
    return sorted(result)


def main() -> None:
    affinity = None
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except Exception:
        pass

    vm = psutil.virtual_memory()
    payload = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "cpu_affinity_count": None if affinity is None else len(affinity),
        "cpu_affinity": affinity,
        "thread_siblings": _thread_siblings(),
        "cpu_governors": _governors(),
        "memory_total_gib": vm.total / 1024**3,
        "memory_available_gib": vm.available / 1024**3,
        "lscpu": _command("lscpu", "-J"),
        "lscpu_extended": _command(
            "lscpu", "-e=CPU,CORE,SOCKET,NODE,CACHE,ONLINE,MAXMHZ,MINMHZ"
        ),
        "numactl_hardware": _command("numactl", "--hardware"),
        "filesystem": _command("df", "-Th", "."),
        "block_devices": _command(
            "lsblk", "-o", "NAME,TYPE,SIZE,ROTA,MODEL,FSTYPE,MOUNTPOINTS"
        ),
        "system_snapshot": system_runtime_snapshot(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numpy_runtime": _numpy_runtime(),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
