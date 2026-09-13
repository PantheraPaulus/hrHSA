from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKING = ROOT / "benchmarking"
SCRIPTS = BENCHMARKING / "scripts"
PYTHON = SCRIPTS / "01_python"
SLURM = SCRIPTS / "02_slurm"
SHELL = SCRIPTS / "03_sh"
CONFIG = SCRIPTS / "04_config"

def _directives(path: Path) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("#SBATCH"):
            continue
        value = line[len("#SBATCH") :].strip()
        if value.startswith("-D"):
            result["chdir"] = value[2:].strip()
            continue
        if value.startswith("-J"):
            result["job-name"] = value[2:].strip()
            continue
        if value.startswith("-o"):
            result["output"] = value[2:].strip()
            continue
        if value.startswith("-e"):
            result["error"] = value[2:].strip()
            continue
        if value.startswith("--"):
            item = value[2:]
            if "=" in item:
                key, val = item.split("=", 1)
                result[key] = val
            else:
                result[item] = True
    return result


def _first_executable_line(path: Path) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return line
    raise AssertionError(f"{path.name}: no executable batch body")


def _cores_per_node(directives: dict[str, str | bool]) -> int:
    cpus_per_task = int(directives.get("cpus-per-task", 1))
    if "ntasks-per-node" in directives:
        tasks = int(directives["ntasks-per-node"])
    else:
        tasks = int(directives.get("ntasks", 1))
    return tasks * cpus_per_task


def _memory_gib(value: str | bool | None) -> float | None:
    if value is None or value is True:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([GM])", str(value), flags=re.I)
    if not match:
        raise AssertionError(f"Unsupported memory directive in policy test: {value!r}")
    amount = float(match.group(1))
    return amount if match.group(2).upper() == "G" else amount / 1024.0


def test_public_benchmark_layout_is_numbered_and_role_separated():
    for directory in (PYTHON, SLURM, SHELL, CONFIG):
        assert directory.is_dir(), directory
    assert (SCRIPTS / "README.md").is_file()


def test_lrz_batch_directives_match_current_partition_rules():
    scripts = sorted(SLURM.glob("coolmuc4_*.sbatch"))
    assert scripts

    for path in scripts:
        text = path.read_text(encoding="utf-8")
        d = _directives(path)

        name = str(d.get("job-name", ""))
        assert name, f"{path.name}: missing job name"
        assert len(name) <= 10, f"{path.name}: LRZ asks for job names <=10 chars"
        assert d.get("chdir") == "./", f"{path.name}: use an explicit existing working directory"
        assert "output" in d and "error" in d, f"{path.name}: stdout/stderr paths should be explicit"
        assert d.get("get-user-env") is True, f"{path.name}: missing --get-user-env"
        assert d.get("export") == "NONE", f"{path.name}: LRZ strongly recommends --export=NONE"
        assert "time" in d, f"{path.name}: parallel/production jobs need an explicit walltime"
        assert "module load slurm_setup" in text, f"{path.name}: missing LRZ slurm_setup module"
        assert _first_executable_line(path) == "module load slurm_setup", (
            f"{path.name}: LRZ documents 'module load slurm_setup' as the first executed line"
        )
        assert "source benchmarking/scripts/03_sh/coolmuc4_common.sh" in text, (
            f"{path.name}: CoolMUC jobs must use the shared environment bootstrap"
        )
        assert "activate_hrhsa" in text, (
            f"{path.name}: CoolMUC jobs must activate the validated hrHSA environment"
        )

        partition = str(d.get("partition", ""))
        cluster = str(d.get("clusters", ""))
        cores = _cores_per_node(d)
        memory = _memory_gib(d.get("mem"))

        if partition == "cm4_tiny":
            assert cluster == "cm4"
            assert d.get("qos") == "cm4_tiny"
            assert 17 <= cores <= 112, (
                f"{path.name}: cm4_tiny currently permits 17-112 physical cores, got {cores}"
            )
            if memory is not None:
                assert memory <= 488, f"{path.name}: cm4_tiny request exceeds 488 GiB node limit"
        elif partition == "cm4_std":
            assert cluster == "cm4"
            assert d.get("qos") == "cm4_std"
            nodes = int(d.get("nodes", 0))
            assert 2 <= nodes <= 4, f"{path.name}: cm4_std currently permits 2-4 nodes"
            assert cores == 112, f"{path.name}: cm4_std should fill each 112-core node"
            if memory is not None:
                assert memory <= 488, f"{path.name}: cm4_std request exceeds 488 GiB node limit"
        elif partition == "cm4_inter":
            assert cluster == "inter"
            assert int(d.get("nodes", 0)) == 1
            # cm4_inter is also used for short backfill diagnostics. The repo's
            # LRZ tutorial documents small interactive allocations, so unlike
            # cm4_std these jobs need not reserve the full 112-core node.
            assert 1 <= cores <= 112, (
                f"{path.name}: cm4_inter request must fit on one 112-core node, got {cores}"
            )
            if memory is not None:
                assert memory <= 488, f"{path.name}: cm4_inter request exceeds CoolMUC-4 node memory"
        elif partition == "serial_std":
            assert cluster == "serial"
            assert 1 <= cores <= 16, f"{path.name}: serial_std is the 1-16 core production partition"
        else:
            raise AssertionError(f"{path.name}: unreviewed LRZ partition {partition!r}")


def test_common_bootstrap_loads_lrz_micromamba_on_demand():
    text = (SHELL / "coolmuc4_common.sh").read_text(encoding="utf-8")
    assert "command -v micromamba" in text
    assert "module load micromamba" in text
    assert 'micromamba activate "$env_name"' in text


def test_common_bootstrap_propagates_benchmark_commit():
    text = (SHELL / "coolmuc4_common.sh").read_text(encoding="utf-8")
    assert "HRHSA_GIT_COMMIT" in text
    assert "git rev-parse HEAD" in text


def test_tiny_scaling_helper_pins_submission_commit():
    text = (SHELL / "submit_coolmuc4_tiny_scaling.sh").read_text(encoding="utf-8")
    assert 'git_commit="$(git rev-parse HEAD' in text
    assert "HRHSA_GIT_COMMIT" in text


def test_large_geometry_confirmation_protocol():
    batch = (SLURM / "coolmuc4_large_geometry_confirm.sbatch").read_text(encoding="utf-8")
    submit = (SHELL / "submit_coolmuc4_large_geometry_confirm.sh").read_text(encoding="utf-8")

    assert '--root "$HRHSA_LARGE_DATA_ROOT"' in batch
    assert "--benchmarks surface" in batch
    assert "--warmup-repeats 1" in batch
    assert "--repeats 5" in batch
    assert "WORKERS * THREADS != 112" in batch

    for geometry in ("16:7", "8:14", "28:4"):
        assert f'"{geometry}"' in submit
    assert "Submitted[[:space:]]+batch[[:space:]]+job" in submit
    assert "dependency=afterok" in submit


def test_interactive_surface_is_short_strict_and_repeated():
    text = (SLURM / "coolmuc4_interactive_surface.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --clusters=inter" in text
    assert "#SBATCH --partition=cm4_inter" in text
    assert "#SBATCH --time=00:30:00" in text
    assert 'GEOMETRIES=("8x14" "16x7")' in text
    assert '--geometry "$GEOMETRY"' in text
    assert "--warmup-repeats 1" in text
    assert "--repeats 5" in text
    assert "--strict-affinity" in text
    assert "do not merge these timings" in text or "Keep cm4_inter timings labelled as diagnostics" in text


def test_mechanism_campaign_is_two_node_strict_and_instrumented():
    text = (SLURM / "coolmuc4_mechanism_diagnostics.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --clusters=cm4" in text
    assert "#SBATCH --partition=cm4_std" in text
    assert "#SBATCH --qos=cm4_std" in text
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --ntasks-per-node=112" in text
    assert "#SBATCH --hint=nomultithread" in text
    assert "#SBATCH --time=01:30:00" in text
    assert "diagnose_numpy_pinned_scaling.py" in text
    assert "diagnose_memory_bandwidth.py" in text
    assert "run_coolmuc4_perf_diagnostics.sh" in text
    assert "run_coolmuc4_mechanism_diagnostics.py" in text
    assert "--node-counts 1,2" in text
    assert "--geometries 1x112,2x56,4x28,8x14,16x7,28x4,56x2,112x1" in text
    assert "--warmup-repeats 1" in text
    assert "--repeats 3" in text
    assert "HRHSA_GIT_COMMIT" in text


def test_submission_helpers_do_not_override_slurm_resources():
    for path in sorted(SHELL.glob("submit_coolmuc4_*.sh")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\bsbatch\s+\\?\s*--", text), path.name


def test_no_high_frequency_slurm_polling_in_hpc_shell_scripts():
    for directory, pattern in ((SHELL, "*.sh"), (SLURM, "*.sbatch")):
        for path in sorted(directory.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            assert "watch -n" not in text
            assert not re.search(r"while.*\b(?:squeue|sacct)\b", text, flags=re.S)
