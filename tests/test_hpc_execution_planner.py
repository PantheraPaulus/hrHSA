from __future__ import annotations

from types import SimpleNamespace

import pytest

from hsa.compute import (
    COOLMUC4,
    COOLMUC4_POINT_112_2026_09,
    COOLMUC4_RASTER_112_2026_08,
    coolmuc4_plan,
    current_slurm_allocation,
    recommend_worker_geometry,
    submit_slurm_job,
)


def test_coolmuc4_full_node_raster_plan_uses_benchmark_geometry():
    plan = coolmuc4_plan(workload="surface_prediction")
    assert plan.geometry.label == "8x14"
    assert plan.total_cores == 112
    assert plan.total_workers == 8
    assert plan.chunk_mb == 256
    assert plan.memory_per_node_gib == 224.0
    assert plan.managed_memory_per_worker_gib == pytest.approx(192.0 / 8.0)
    assert plan.benchmark_profile == COOLMUC4_RASTER_112_2026_08.name
    assert plan.benchmark_confidence == "provisional"
    assert "8x14 median 10.79 s" in plan.evidence


def test_coolmuc4_full_node_point_plan_uses_point_evidence():
    plan = coolmuc4_plan(workload="point_sampling")
    assert plan.geometry.label == "8x14"
    assert plan.total_cores == 112
    assert plan.total_workers == 8
    assert plan.chunk_mb == 256
    assert plan.benchmark_profile == COOLMUC4_POINT_112_2026_09.name
    assert plan.benchmark_confidence == "provisional"
    assert "1B-point/~194-GiB" in plan.evidence
    assert "256-partition graph window" in plan.evidence


def test_raster_and_point_profiles_are_workload_specific():
    assert COOLMUC4_RASTER_112_2026_08.workloads == ("surface_prediction",)
    assert COOLMUC4_POINT_112_2026_09.workloads == ("point_sampling",)


def test_coolmuc4_multinode_plan_reuses_per_node_geometry():
    plan = coolmuc4_plan(workload="surface_prediction", nodes=2)
    assert plan.partition == "cm4_std"
    assert plan.geometry.label == "8x14"
    assert plan.total_workers == 16
    assert plan.total_cores == 224


def test_task_density_can_only_downscale_geometry():
    geometry = recommend_worker_geometry(
        COOLMUC4.node_geometry_candidates(),
        task_count=16,
        preferred_labels=("8x14", "16x7", "14x8"),
    )
    assert geometry.workers <= 2
    assert geometry.total_threads == 112


def test_current_slurm_allocation_parses_environment(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cm4")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "cm4_std")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "112(x2)")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(224 * 1024))

    allocation = current_slurm_allocation()
    assert allocation is not None
    assert allocation.job_id == "12345"
    assert allocation.nodes == 2
    assert allocation.cpus_per_node == 112
    assert allocation.memory_per_node_gib == 224.0


def test_current_slurm_allocation_treats_zero_memory_as_unconstrained(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "112(x2)")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "0")
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)

    allocation = current_slurm_allocation()
    assert allocation is not None
    assert allocation.memory_per_node_gib is None


def test_submit_slurm_job_uses_script_resources_and_preserves_cluster(monkeypatch, tmp_path):
    script = tmp_path / "job.sbatch"
    script.write_text("#!/bin/bash\n#SBATCH --time=00:10:00\n", encoding="utf-8")

    monkeypatch.setattr("hsa.compute.slurm.shutil.which", lambda name: f"/usr/bin/{name}")

    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return SimpleNamespace(stdout="98765;cm4\n")

    monkeypatch.setattr("hsa.compute.slurm.subprocess.run", fake_run)

    job = submit_slurm_job(script, args=("8", "14"), chdir=tmp_path)
    assert job.job_id == "98765"
    assert job.cluster == "cm4"
    assert calls["command"] == [
        "/usr/bin/sbatch",
        "--parsable",
        str(script),
        "8",
        "14",
    ]
    assert "--nodes" not in " ".join(calls["command"])
    assert calls["kwargs"]["cwd"] == str(tmp_path)
