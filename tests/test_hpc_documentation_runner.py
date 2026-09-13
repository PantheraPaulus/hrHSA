from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarking"
    / "scripts"
    / "01_python"
    / "run_documentation_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("run_documentation_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reference_raster_profile_is_large_and_chunk_aligned():
    profile = MODULE.RASTER_PROFILES["reference"]

    assert profile == {
        "size": 49_152,
        "bands": 6,
        "points": 10_000_000,
    }
    assert profile["size"] % 1024 == 0

    manifest = MODULE._expected_raster_manifest(profile, storage_chunk=1024)
    assert manifest["spatial_chunks"] == 2_304


def test_prepared_manifest_must_match_profile_definition():
    expected = {
        "size": 49_152,
        "bands": 6,
        "storage_chunk": 1024,
        "points": 10_000_000,
        "spatial_chunks": 2_304,
    }

    assert MODULE._manifest_matches(dict(expected), expected)

    stale = dict(expected)
    stale["size"] = 24_576
    assert not MODULE._manifest_matches(stale, expected)

    stale = dict(expected)
    stale["points"] = 2_000_000
    assert not MODULE._manifest_matches(stale, expected)


def test_reference_preparation_command_contains_exact_workload(tmp_path):
    command = MODULE._prepare_raster_command(
        root=tmp_path,
        profile=MODULE.RASTER_PROFILES["reference"],
        storage_chunk=1024,
        physical_cores=12,
        force=True,
    )

    joined = " ".join(command)
    assert "--size 49152" in joined
    assert "--bands 6" in joined
    assert "--points 10000000" in joined
    assert "--storage-chunk 1024" in joined
    assert "--workers 8" in joined
    assert command[-1] == "--force"
