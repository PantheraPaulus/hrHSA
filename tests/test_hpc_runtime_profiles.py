from __future__ import annotations

import json

import pytest

from hsa.compute import (
    COOLMUC4_HARDWARE,
    COOLMUC4_POLICY,
    COOLMUC4_RASTER_112_2026_08,
    build_run_manifest,
    coolmuc4_plan,
    discover_runtime_topology,
    validate_worker_topology,
    write_run_manifest,
)


class FakeClient:
    def __init__(self, workers):
        self.workers = workers

    def run(self, function):
        return self.workers


def _physical_id(index: int) -> tuple[int, int]:
    if index < 56:
        return (0, index)
    return (1, index - 56)


def _worker(name: str, cpus, *, physical_cores=None, host="cm4r01c01s01", threads=14):
    cpu_list = [int(cpu) for cpu in cpus]
    if physical_cores is None:
        physical_cores = cpu_list
    core_ids = sorted({_physical_id(int(core)) for core in physical_cores})
    sockets = sorted({socket for socket, _ in core_ids})
    return {
        "hostname": host,
        "pid": 1000 + int(name),
        "cpu_affinity": cpu_list,
        "affinity_cpu_count": len(cpu_list),
        "physical_core_count": len(core_ids),
        "physical_core_ids": [list(value) for value in core_ids],
        "physical_cpu_representatives": [int(core) for core in physical_cores],
        "socket_ids": sockets,
        "numa_node_ids": sockets,
        "cpu_model": "test-cpu",
        "memory_total_gib": 488.0,
        "slurm_procid": name,
        "slurm_localid": name,
        "slurm_nodeid": "0",
        "dask_worker_name": name,
        "dask_nthreads": threads,
        "dask_memory_limit_gib": 24.0,
        "local_directory": f"/tmp/dask-{name}",
    }


def _eight_by_fourteen_workers():
    return {
        f"tcp://worker-{index}": _worker(str(index), range(index * 14, (index + 1) * 14))
        for index in range(8)
    }


def _smt_eight_by_fourteen_workers():
    workers = {}
    for index in range(8):
        physical = list(range(index * 14, (index + 1) * 14))
        logical = physical + [cpu + 112 for cpu in physical]
        workers[f"tcp://worker-{index}"] = _worker(
            str(index), logical, physical_cores=physical
        )
    return workers


def _smt_112_by_one_workers():
    return {
        f"tcp://worker-{index}": _worker(
            str(index),
            [index, index + 112],
            physical_cores=[index],
            threads=1,
        )
        for index in range(112)
    }


def test_coolmuc_policy_hardware_and_benchmark_evidence_are_separate():
    assert COOLMUC4_POLICY.max_memory_per_node_gib == 488.0
    assert COOLMUC4_POLICY.scheduler_poll_seconds == 600.0
    assert COOLMUC4_POLICY.max_job_steps == 40_000
    assert COOLMUC4_POLICY.interactive_cluster == "inter"
    assert COOLMUC4_POLICY.interactive_partition == "cm4_inter"
    assert COOLMUC4_HARDWARE.physical_cores_per_node == 112
    assert COOLMUC4_HARDWARE.logical_cores_per_node == 224
    assert COOLMUC4_HARDWARE.sockets_per_node == 2
    assert COOLMUC4_HARDWARE.cores_per_socket == 56
    assert COOLMUC4_RASTER_112_2026_08.preferred_geometries[:2] == ("8x14", "16x7")
    assert COOLMUC4_RASTER_112_2026_08.confidence == "provisional"
    assert COOLMUC4_RASTER_112_2026_08.repeat_count == 3


def test_full_node_plan_exposes_profile_memory_and_socket_decomposition():
    plan = coolmuc4_plan(workload="surface_prediction")
    assert plan.geometry.label == "8x14"
    assert plan.workers_per_socket == 4.0
    assert plan.benchmark_profile == COOLMUC4_RASTER_112_2026_08.name
    assert plan.benchmark_confidence == "provisional"
    assert plan.memory_per_node_gib == 224.0
    assert plan.memory_reserve_per_node_gib == 32.0
    assert plan.managed_memory_per_node_gib == pytest.approx(192.0)
    assert plan.managed_memory_per_worker_gib == pytest.approx(24.0)


def test_runtime_topology_discovers_affinity_and_socket_mapping(monkeypatch):
    monkeypatch.setattr("hsa.compute.topology.os.sched_getaffinity", lambda pid: set(range(14)))
    monkeypatch.setattr(
        "hsa.compute.topology._sysfs_topology_rows",
        lambda cpus, sysfs_root=None: [(cpu, cpu, 0, 0) for cpu in cpus],
    )
    monkeypatch.setattr("hsa.compute.topology._cpu_model", lambda: "test-cpu")
    monkeypatch.setattr("hsa.compute.topology._memory_total_gib", lambda: 488.0)
    observed = discover_runtime_topology()
    assert observed.affinity_cpu_count == 14
    assert observed.physical_core_count == 14
    assert observed.physical_core_ids == tuple((0, cpu) for cpu in range(14))
    assert observed.socket_ids == (0,)
    assert observed.numa_node_ids == (0,)
    assert observed.cpu_model == "test-cpu"


def test_runtime_topology_reads_smt_siblings_from_sysfs_without_lscpu(tmp_path, monkeypatch):
    for cpu in (0, 112):
        root = tmp_path / f"cpu{cpu}" / "topology"
        root.mkdir(parents=True)
        (root / "core_id").write_text("0\n")
        (root / "physical_package_id").write_text("0\n")
    monkeypatch.setattr("hsa.compute.topology.os.sched_getaffinity", lambda pid: {0, 112})
    monkeypatch.setattr(
        "hsa.compute.topology._lscpu_rows",
        lambda: (_ for _ in ()).throw(AssertionError("lscpu fallback must not run")),
    )
    observed = discover_runtime_topology(sysfs_root=tmp_path)
    assert observed.cpu_affinity == (0, 112)
    assert observed.affinity_cpu_count == 2
    assert observed.physical_core_count == 1
    assert observed.physical_core_ids == ((0, 0),)
    assert observed.physical_cpu_representatives == (0,)


def test_strict_worker_topology_accepts_disjoint_eight_by_fourteen_layout():
    validation = validate_worker_topology(
        FakeClient(_eight_by_fourteen_workers()),
        coolmuc4_plan(workload="surface_prediction"),
        strict_affinity=True,
    )
    assert validation.ok
    assert validation.errors == ()
    assert validation.warnings == ()
    assert validation.workers_per_host == {"cm4r01c01s01": 8}
    assert validation.unique_cpus_per_host == {"cm4r01c01s01": 112}
    assert validation.unique_physical_cores_per_host == {"cm4r01c01s01": 112}
    assert validation.affinity_sizes == (14,)
    assert validation.physical_core_counts == (14,)


def test_strict_worker_topology_accepts_smt2_core_binding():
    validation = validate_worker_topology(
        FakeClient(_smt_eight_by_fourteen_workers()),
        coolmuc4_plan(workload="surface_prediction"),
        strict_affinity=True,
    )
    assert validation.ok
    assert validation.affinity_sizes == (28,)
    assert validation.physical_core_counts == (14,)
    assert validation.unique_cpus_per_host == {"cm4r01c01s01": 224}
    assert validation.unique_physical_cores_per_host == {"cm4r01c01s01": 112}


def test_strict_worker_topology_accepts_smt2_112_by_one_layout():
    """Regression for cm4_inter job 389195: one core appears as two SMT CPU ids."""
    validation = validate_worker_topology(
        FakeClient(_smt_112_by_one_workers()),
        coolmuc4_plan(workload="surface_prediction", geometry="112x1"),
        strict_affinity=True,
    )
    assert validation.ok
    assert validation.errors == ()
    assert validation.warnings == ()
    assert validation.observed_workers == 112
    assert validation.affinity_sizes == (2,)
    assert validation.physical_core_counts == (1,)
    assert validation.unique_cpus_per_host == {"cm4r01c01s01": 224}
    assert validation.unique_physical_cores_per_host == {"cm4r01c01s01": 112}


def test_strict_worker_topology_rejects_physical_overlap_and_wrong_thread_count():
    workers = _smt_eight_by_fourteen_workers()
    workers["tcp://worker-7"] = _worker(
        "7",
        list(range(98, 112)) + list(range(210, 224)),
        physical_cores=range(0, 14),
        threads=7,
    )
    validation = validate_worker_topology(
        FakeClient(workers),
        coolmuc4_plan(workload="surface_prediction"),
        strict_affinity=True,
    )
    assert not validation.ok
    assert any("Dask threads/worker" in error for error in validation.errors)
    assert any("physical-core sets overlap" in error for error in validation.errors)


def test_run_manifest_records_plan_profiles_and_atomic_json(tmp_path, monkeypatch):
    monkeypatch.setattr("hsa.compute.provenance._git_dirty", lambda: False)
    monkeypatch.setattr("hsa.compute.provenance.current_git_commit", lambda: "abc123")
    monkeypatch.setattr("hsa.compute.provenance._allocation_dict", lambda: None)
    monkeypatch.setattr(
        "hsa.compute.provenance.discover_runtime_topology",
        lambda: type("Observed", (), {"as_dict": lambda self: {"hostname": "login"}})(),
    )
    monkeypatch.setattr(
        "hsa.compute.provenance.software_versions",
        lambda: {"hsa": "test", "dask": "test", "distributed": "test"},
    )
    plan = coolmuc4_plan(workload="surface_prediction")
    manifest = build_run_manifest(plan, dataset={"rows": 123})
    path = tmp_path / "run.manifest.json"
    write_run_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["git_commit"] == "abc123"
    assert payload["git_dirty"] is False
    assert payload["execution_plan"]["geometry"] == "8x14"
    assert payload["execution_plan"]["managed_memory_per_node_gib"] == pytest.approx(192.0)
    assert payload["site_policy"]["max_memory_per_node_gib"] == 488.0
    assert payload["site_policy"]["interactive_cluster"] == "inter"
    assert payload["site_policy"]["interactive_partition"] == "cm4_inter"
    assert payload["hardware_profile"]["cores_per_socket"] == 56
    assert payload["benchmark_profile"]["name"] == COOLMUC4_RASTER_112_2026_08.name
    assert payload["benchmark_profile"]["confidence"] == "provisional"
    assert payload["dataset"] == {"rows": 123}
    assert payload["software"] == {"hsa": "test", "dask": "test", "distributed": "test"}
