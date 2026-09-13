from __future__ import annotations

import numpy as np
import pandas as pd

from hsa.ssf.frequentist import fit_ssf_per_id
from hsa.ssf.issf import ISSFDesign
from hsa.ssf.parallel import fit_issf_per_id


def _choice_frame(*, with_offset: bool) -> pd.DataFrame:
    rng = np.random.default_rng(20260903)
    n_individuals = 4
    strata_per_id = 60
    n_choices = 5
    rows = []
    beta = np.array([0.35, -0.2])

    for individual in range(n_individuals):
        for local_stratum in range(strata_per_id):
            stratum = individual * strata_per_id + local_stratum
            X = rng.normal(size=(n_choices, 2))
            offset = (
                rng.normal(scale=0.15, size=n_choices)
                if with_offset
                else np.zeros(n_choices)
            )
            eta = X @ beta + offset
            probability = np.exp(eta - eta.max())
            probability /= probability.sum()
            chosen = int(rng.choice(n_choices, p=probability))

            for candidate in range(n_choices):
                row = {
                    "id": f"animal-{individual}",
                    "stratum_id": stratum,
                    "candidate_id": candidate,
                    "used": int(candidate == chosen),
                    "x0": X[candidate, 0],
                    "x1": X[candidate, 1],
                }
                if with_offset:
                    row["proposal_offset"] = offset[candidate]
                rows.append(row)

    return pd.DataFrame(rows)


def _sorted_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["id", "predictor", "beta", "se", "lower", "upper", "p", "n_strata"]
    return (
        frame[columns]
        .sort_values(["id", "predictor"])
        .reset_index(drop=True)
    )


def test_process_parallel_ssf_matches_serial_fit_objects():
    frame = _choice_frame(with_offset=False)
    reference_beta = np.array([0.1, -0.1])

    serial = fit_ssf_per_id(
        frame,
        id_col="id",
        predictors=("x0", "x1"),
        reference_beta=reference_beta,
        method="lbfgs",
        workers=1,
        native_threads=1,
    )
    parallel = fit_ssf_per_id(
        frame,
        id_col="id",
        predictors=("x0", "x1"),
        reference_beta=reference_beta,
        method="lbfgs",
        workers=2,
        executor="process",
        native_threads=1,
    )

    assert tuple(parallel.fits) == tuple(serial.fits)
    pd.testing.assert_frame_equal(
        _sorted_summary(parallel.summary),
        _sorted_summary(serial.summary),
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    for animal_id in serial.fits:
        np.testing.assert_allclose(
            parallel.fits[animal_id].result.params,
            serial.fits[animal_id].result.params,
            rtol=1e-10,
            atol=1e-10,
        )


class _PreparedISSF:
    def __init__(self, design: ISSFDesign):
        self.id_col = design.id_col
        self._design = design

    def prepare_design(self, *, scaling=None, center_offset=True):
        return self._design


def _issf_design(frame: pd.DataFrame) -> ISSFDesign:
    return ISSFDesign(
        data=frame,
        predictors=("x0", "x1"),
        endpoint_predictors=("x0", "x1"),
        start_predictors=(),
        directional_predictors=(),
        movement_terms=(),
        interaction_terms=(),
        interaction_columns={},
        scaling={},
        offset_col="proposal_offset",
        proposal_logpdf_col="proposal_logpdf",
        id_col="id",
        stratum_col="stratum_id",
        n_choices=5,
        diagnostics={
            "n_individuals": 4,
            "n_strata_retained": 240,
            "n_choice_rows": len(frame),
        },
    )


def test_process_parallel_issf_helper_matches_serial_fit_objects():
    frame = _choice_frame(with_offset=True)
    analysis = _PreparedISSF(_issf_design(frame))

    serial = fit_issf_per_id(
        analysis,
        method="lbfgs",
        workers=1,
        native_threads=1,
    )
    parallel = fit_issf_per_id(
        analysis,
        method="lbfgs",
        workers=2,
        executor="process",
        native_threads=1,
    )

    assert tuple(parallel.fits) == tuple(serial.fits)
    pd.testing.assert_frame_equal(
        _sorted_summary(parallel.summary),
        _sorted_summary(serial.summary),
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    for animal_id in serial.fits:
        np.testing.assert_allclose(
            parallel.fits[animal_id].result.params,
            serial.fits[animal_id].result.params,
            rtol=1e-10,
            atol=1e-10,
        )
