import numpy as np
import pandas as pd

import hsa.rsf.frequentist_validation as frequentist_validation
from hsa.rsf.schemes import (
    BayesianLOIOResult,
    BlockedBootstrap,
    ContiguousTemporalBlocks,
    FrequentistLOIOResult,
    FrequentistValidationUncertaintyResult,
    LeaveOneIndividualOut,
)


class _DummyAnalysis:
    def __init__(self):
        self.received = None

    def _run_loio(self, scheme, **kwargs):
        self.received = (scheme, kwargs)
        return "ok"


def _fake_bayesian_diagnostics():
    used_scores = np.array(
        [
            [-2.0, -1.5, -1.0, -0.2, 0.1, 0.5, 1.1, 1.8],
            [-1.8, -1.4, -0.8, -0.1, 0.2, 0.7, 1.0, 2.0],
        ]
    )
    available = np.linspace(-2.5, 2.5, 200)
    available_scores = np.vstack([available, available * 0.95])
    used_data = pd.DataFrame(
        {
            "Timestamp": pd.date_range(
                "2024-01-04",
                periods=8,
                freq="7D",
                tz="UTC",
            )
        }
    )
    return {
        "bird-A": {
            "fold": 0,
            "scores": {
                "used_scores": used_scores,
                "available_scores": available_scores,
                "used_data": used_data,
            },
        }
    }


def _fake_frequentist_diagnostics():
    return {
        "bird-A": {
            "fold": 0,
            "test_pred": pd.DataFrame(),
            "rsf": object(),
            "domain": object(),
        }
    }


def _fake_frequentist_scores():
    return {
        "used_ranks": np.array(
            [0.05, 0.18, 0.31, 0.44, 0.58, 0.72, 0.86, 0.97],
            dtype=np.float32,
        ),
        "used_data": pd.DataFrame(
            {
                "Timestamp": pd.date_range(
                    "2024-01-04",
                    periods=8,
                    freq="7D",
                    tz="UTC",
                )
            }
        ),
        "available_scores": np.linspace(0.0, 1.0, 200),
    }


def test_leave_one_individual_out_is_a_strategy():
    analysis = _DummyAnalysis()
    scheme = LeaveOneIndividualOut(
        heldout=["bird-A"],
        sampling_factor_train=20,
        n_background=500,
        n_bins=4,
        seed=9,
    )

    result = scheme.run(analysis, sample_kwargs={"draws": 10})

    assert result == "ok"
    assert analysis.received[0] is scheme
    assert analysis.received[1]["sample_kwargs"]["draws"] == 10


def test_bayesian_result_composes_postfit_validation_strategies():
    loio = BayesianLOIOResult(
        summary=pd.DataFrame({"heldout_ID": ["bird-A"]}),
        params=pd.DataFrame(),
        boyce_bins=pd.DataFrame(),
        diagnostics=_fake_bayesian_diagnostics(),
        scheme=LeaveOneIndividualOut(
            heldout=["bird-A"],
            n_background=200,
            n_bins=4,
            seed=7,
        ),
    )

    uncertainty = loio.evaluate_uncertainty(
        BlockedBootstrap(
            replicates=12,
            block="14D",
        ),
        ContiguousTemporalBlocks(
            folds=2,
            unit="W",
        ),
    )

    assert set(uncertainty.method_summary["method"]) == {
        "bootstrap",
        "temporal_contiguous",
    }
    assert len(
        uncertainty.replicate_summary.query("method == 'bootstrap'")
    ) == 12

    periods = uncertainty.temporal_periods("bird-A")
    assert len(periods) == 2
    assert periods["replicate_label"].str.contains("2024-W").all()
    assert periods["heldout_units"].str.contains("2024-W").all()


def test_frequentist_result_composes_same_postfit_strategies(monkeypatch):
    monkeypatch.setattr(
        frequentist_validation,
        "_prepare_frequentist_scores",
        lambda *args, **kwargs: _fake_frequentist_scores(),
    )

    loio = FrequentistLOIOResult(
        summary=pd.DataFrame({"heldout_ID": ["bird-A"]}),
        params=pd.DataFrame(),
        boyce_bins=pd.DataFrame(),
        diagnostics=_fake_frequentist_diagnostics(),
        calibration_bins=pd.DataFrame(),
        scheme=LeaveOneIndividualOut(
            heldout=["bird-A"],
            n_background=200,
            n_bins=4,
            seed=7,
        ),
    )

    uncertainty = loio.evaluate_uncertainty(
        BlockedBootstrap(
            replicates=12,
            block="14D",
        ),
        ContiguousTemporalBlocks(
            folds=2,
            unit="W",
        ),
    )

    assert isinstance(
        uncertainty,
        FrequentistValidationUncertaintyResult,
    )
    assert set(uncertainty.method_summary["method"]) == {
        "bootstrap",
        "temporal_contiguous",
    }
    assert uncertainty.config["uncertainty_scope"] == "validation_sample_only"

    periods = uncertainty.temporal_periods("bird-A")
    assert len(periods) == 2
    assert periods["replicate_label"].str.contains("2024-W").all()


def test_default_postfit_validation_uses_only_nonredundant_modes():
    loio = BayesianLOIOResult(
        summary=pd.DataFrame({"heldout_ID": ["bird-A"]}),
        params=pd.DataFrame(),
        boyce_bins=pd.DataFrame(),
        diagnostics=_fake_bayesian_diagnostics(),
        scheme=LeaveOneIndividualOut(
            heldout=["bird-A"],
            n_bins=4,
        ),
    )

    result = loio.evaluate_uncertainty(
        BlockedBootstrap(replicates=5, block="14D"),
    )

    assert set(result.method_summary["method"]) == {"bootstrap"}
