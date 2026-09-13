from __future__ import annotations

from pathlib import Path

from hsa.compute.workloads import geometry_for_core_budget, raster_workload_for_gib


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "benchmarking" / "scripts" / "01_python"
SLURM = ROOT / "benchmarking" / "scripts" / "02_slurm"


def test_raster_workload_rounds_to_storage_chunk_and_reports_actual_size():
    workload = raster_workload_for_gib(
        1.0,
        bands=8,
        dtype="float32",
        storage_chunk=1024,
    )

    assert workload.size % 1024 == 0
    assert workload.logical_gib > 0
    assert workload.relative_error_fraction >= 0


def test_geometry_for_core_budget_preserves_thread_width_when_possible():
    assert geometry_for_core_budget("8x14", 112) == (8, 14)
    assert geometry_for_core_budget("8x14", 56) == (4, 14)
    assert geometry_for_core_budget("8x14", 28) == (2, 14)


def test_surface_scaling_runner_exposes_capacity_strong_and_weak_modes():
    runner = (SCRIPTS / "run_surface_scaling.py").read_text(encoding="utf-8")
    instrumented = (SCRIPTS / "run_surface_scaling_instrumented.py").read_text(
        encoding="utf-8"
    )

    assert 'choices=("capacity", "strong", "weak")' in runner
    assert '"surface_family_manifest.json"' in runner
    assert "strict_affinity=True" in runner
    assert "logical_gib_per_physical_core" in runner
    assert "distributed_node_runtime_snapshot" in instrumented
    assert "task_compute_parallelism" in instrumented
    assert "node_major_page_faults" in instrumented
    assert ".diagnostics.jsonl" in instrumented


def test_inter_surface_scaling_jobs_keep_capacity_and_scaling_separate():
    capacity = (SLURM / "coolmuc4_inter_capacity_scaling.sbatch").read_text(
        encoding="utf-8"
    )
    scaling = (SLURM / "coolmuc4_inter_scaling.sbatch").read_text(
        encoding="utf-8"
    )
    std_scaling = (SLURM / "coolmuc4_std_surface_scaling.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --clusters=inter" in capacity
    assert "#SBATCH --partition=cm4_inter" in capacity
    assert "#SBATCH --nodes=1" in capacity
    assert "--mode capacity" in capacity
    assert '--raster-gib "$RASTER_GIB"' in capacity
    assert "run_surface_scaling_instrumented.py" in capacity

    # LRZ cm4_inter is single-node. Keep one-node reference measurements there;
    # true multi-node strong/weak scaling belongs to cm4_std.
    assert "#SBATCH --clusters=inter" in scaling
    assert "#SBATCH --partition=cm4_inter" in scaling
    assert "#SBATCH --nodes=1" in scaling
    assert "#SBATCH --time=01:00:00" in scaling
    assert "--mode strong" in scaling
    assert "--mode weak" in scaling
    assert scaling.count("--node-counts 1") == 2
    assert '--gib-per-core "$GIB_PER_CORE"' in scaling
    assert scaling.count("run_surface_scaling_instrumented.py") == 2

    assert "#SBATCH --clusters=cm4" in std_scaling
    assert "#SBATCH --partition=cm4_std" in std_scaling
    assert "--node-counts" in std_scaling
