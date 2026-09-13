from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "benchmarking" / "scripts" / "01_python"
RUNNER = SCRIPTS / "run_planned_execution.py"


def _load_runner_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "hrhsa_planned_runner_test",
            RUNNER,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_interactive_full_node_prefers_task_decomposition_over_scheduler_units():
    runner = _load_runner_module()
    allocation = SimpleNamespace(
        tasks_per_node=112,
        cpus_per_task=None,
        cpus_per_node=224,
    )
    assert runner._allocation_physical_cores_per_node(allocation) == 112


def test_full_logical_cpu_fallback_maps_to_112_physical_cores():
    runner = _load_runner_module()
    allocation = SimpleNamespace(
        tasks_per_node=None,
        cpus_per_task=None,
        cpus_per_node=224,
    )
    assert runner._allocation_physical_cores_per_node(allocation) == 112


def test_partial_task_allocation_is_not_promoted_to_full_node():
    runner = _load_runner_module()
    allocation = SimpleNamespace(
        tasks_per_node=56,
        cpus_per_task=1,
        cpus_per_node=112,
    )
    assert runner._allocation_physical_cores_per_node(allocation) == 56
