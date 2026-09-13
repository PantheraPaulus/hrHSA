from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarking"
    / "scripts"
    / "01_python"
    / "validate_bayesian_samplers.py"
)
SPEC = importlib.util.spec_from_file_location("validate_bayesian_samplers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_posterior_agreement_detects_small_matching_shift(tmp_path):
    reference = pd.DataFrame(
        {
            "mean": [1.0, -0.5],
            "sd": [0.2, 0.4],
        },
        index=pd.Index(["alpha", "beta[0]"], name="parameter"),
    )
    candidate = pd.DataFrame(
        {
            "mean": [1.02, -0.46],
            "sd": [0.21, 0.38],
        },
        index=reference.index,
    )
    ref_path = tmp_path / "reference.csv"
    cand_path = tmp_path / "candidate.csv"
    reference.to_csv(ref_path)
    candidate.to_csv(cand_path)

    result = MODULE._posterior_agreement(ref_path, cand_path)

    assert result["common_parameters"] == 2
    assert result["max_abs_mean_diff"] == pytest.approx(0.04)
    assert result["max_mean_diff_pooled_sd"] < 0.2
    assert result["max_relative_sd_diff"] < 0.1
