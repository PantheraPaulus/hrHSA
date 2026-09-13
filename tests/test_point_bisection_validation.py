from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_dir = (
        Path(__file__).resolve().parents[1]
        / "benchmarking"
        / "scripts"
        / "01_python"
    )
    path = script_dir / "prepare_point_bisection_workstation.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_point_bisection_workstation_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script_dir))
    return module


def test_uniform_bisection_validation_rows_are_balanced():
    module = _load_module()
    rows, hot = module._scenario_row_counts(
        total_rows=100_000_000,
        tile_rows=10,
        tile_cols=10,
        scenario="uniform",
    )
    assert len(rows) == 100
    assert sum(rows) == 100_000_000
    assert min(rows) == max(rows) == 1_000_000
    assert hot == []


def test_moderate_bisection_validation_concentrates_80_percent_in_30_percent_domain():
    module = _load_module()
    rows, hot = module._scenario_row_counts(
        total_rows=100_000_000,
        tile_rows=10,
        tile_cols=10,
        scenario="moderate",
    )
    assert len(hot) == 30
    assert sum(rows[index] for index in hot) == 80_000_000
    assert sum(rows) == 100_000_000
    assert max(rows) < 3_000_000


def test_strong_bisection_validation_concentrates_85_percent_in_20_percent_domain():
    module = _load_module()
    rows, hot = module._scenario_row_counts(
        total_rows=100_000_000,
        tile_rows=10,
        tile_cols=10,
        scenario="strong",
    )
    assert len(hot) == 20
    assert sum(rows[index] for index in hot) == 85_000_000
    assert sum(rows) == 100_000_000
    assert max(rows) == 4_250_000
    assert min(rows) == 187_500
