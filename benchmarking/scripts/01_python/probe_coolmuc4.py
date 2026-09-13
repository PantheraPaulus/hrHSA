"""Print machine/topology information relevant to hrHSA CoolMUC-4 tuning."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess

import numpy as np

from hsa.compute import (
    COOLMUC4,
    COOLMUC4_HARDWARE,
    COOLMUC4_POLICY,
    COOLMUC4_RASTER_112_2026_08,
    coolmuc4_plan,
    current_slurm_allocation,
    discover_runtime_topology,
)


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _interfaces() -> list[dict[str, str]]:
    output = _command("ip", "-o", "-4", "addr", "show", "scope", "global") or ""
    result = []
    for line in output.splitlines():
        match = re.match(r"\d+:\s+([^\s]+)\s+inet\s+([^/]+)/", line)
        if match:
            result.append({"interface": match.group(1), "ipv4": match.group(2)})
    return result


def _numpy_runtime() -> str:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            np.show_runtime()
    except Exception as exc:
        return f"numpy.show_runtime unavailable: {exc}"
    return buffer.getvalue().strip()


def main() -> None:
    allocation = current_slurm_allocation()
    plan = None
    if allocation is not None:
        try:
            plan = coolmuc4_plan(
                workload="surface_prediction",
                nodes=allocation.nodes,
                cores_per_node=allocation.cpus_per_node or COOLMUC4.physical_cores_per_node,
            )
        except Exception:
            plan = None

    payload = {
        "runtime_topology": discover_runtime_topology().as_dict(),
        "lscpu": _command("lscpu", "-J"),
        "numactl_hardware": _command("numactl", "--hardware"),
        "numpy_runtime": _numpy_runtime(),
        "interfaces": _interfaces(),
        "slurm_allocation": None if allocation is None else allocation.__dict__,
        "site_policy": COOLMUC4_POLICY.as_dict(),
        "hardware_profile": COOLMUC4_HARDWARE.as_dict(),
        "benchmark_profile": COOLMUC4_RASTER_112_2026_08.as_dict(),
        "recommended_plan": None if plan is None else plan.as_dict(),
        "node_geometries": [
            geometry.label for geometry in COOLMUC4.node_geometry_candidates()
        ],
        "selected_dask_interface": os.environ.get("HRHSA_DASK_INTERFACE"),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
