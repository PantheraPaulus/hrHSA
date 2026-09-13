import numpy as np
import pandas as pd
import pytest

from hsa.ssf import (
    build_hierarchical_issf_model,
    prepare_issf_design,
)
from hsa.ssf.data import build_ssf_choice_arrays
from hsa.ssf.frequentist import fit_conditional_ssf


def _synthetic_issf_table(seed=42, n_strata=120, n_choices=4):
    rng = np.random.default_rng(seed)
    rows = []

    for stratum in range(n_strata):
        animal = "A" if stratum < n_strata // 2 else "B"
        heat_start = rng.normal()
        vrm_start = rng.normal()
        step_length = rng.gamma(shape=1.3, scale=2500.0, size=n_choices) + 1.0
        turn_angle = rng.uniform(-np.pi, np.pi, size=n_choices)
        endpoint_heat = rng.normal(size=n_choices)
        endpoint_vrm = rng.normal(size=n_choices)
        proposal_logpdf = -rng.uniform(5.0, 12.0, size=n_choices)

        step_km = step_length / 1000.0
        eta = (
            0.5 * endpoint_vrm
            + 0.9 * endpoint_heat
            - 0.15 * step_km
            - 0.4 * np.log(step_length)
            + 0.08 * np.cos(turn_angle)
            + 0.025 * heat_start * step_km
            - 0.02 * vrm_start * step_km
            - proposal_logpdf
        )
        eta -= eta.max()
        probability = np.exp(eta)
        probability /= probability.sum()
        chosen = int(rng.choice(n_choices, p=probability))

        for candidate in range(n_choices):
            rows.append(
                {
                    "id": animal,
                    "stratum_id": stratum,
                    "candidate_id": candidate,
                    "used": int(candidate == chosen),
                    "step_length": step_length[candidate],
                    "turn_angle": turn_angle[candidate],
                    "heat_end": endpoint_heat[candidate],
                    "vrm_end": endpoint_vrm[candidate],
                    "heat_start": heat_start,
                    "vrm_start": vrm_start,
                    "proposal_logpdf": proposal_logpdf[candidate],
                }
            )

    return pd.DataFrame(rows)


def _design():
    return prepare_issf_design(
        _synthetic_issf_table(),
        endpoint_predictors=["vrm_end", "heat_end"],
        start_predictors=["heat_start", "vrm_start"],
        id_col="id",
        expected_n_choices=4,
    )


def test_prepare_issf_design_uses_common_environmental_scale_and_natural_movement_terms():
    design = _design()
    data = design.data

    assert design.predictors == (
        "vrm_end_z",
        "heat_end_z",
        "step_length_km",
        "log_step_length",
        "cos_turn_angle",
        "heat_start_x_step_length",
        "heat_start_x_log_step_length",
        "vrm_start_x_step_length",
        "vrm_start_x_log_step_length",
    )
    assert np.isclose(data["vrm_end_z"].mean(), 0.0, atol=1e-12)
    assert np.isclose(data["heat_end_z"].mean(), 0.0, atol=1e-12)

    starts = data.drop_duplicates(["id", "stratum_id"])
    assert np.isclose(starts["heat_start_z"].mean(), 0.0, atol=1e-12)
    assert np.isclose(starts["vrm_start_z"].mean(), 0.0, atol=1e-12)

    np.testing.assert_allclose(
        data["heat_start_x_step_length"],
        data["heat_start_z"] * data["step_length_km"],
    )
    np.testing.assert_allclose(
        data["vrm_start_x_log_step_length"],
        data["vrm_start_z"] * data["log_step_length"],
    )


def test_prepare_issf_design_centers_proposal_offset_within_strata():
    design = _design()
    means = design.data.groupby(["id", "stratum_id"])[
        "proposal_offset"
    ].mean()
    np.testing.assert_allclose(means.to_numpy(), 0.0, atol=1e-12)
    assert design.diagnostics["proposal_correction"]


def test_proposal_corrected_design_fits_with_fast_conditional_engine():
    design = _design()
    model, result = fit_conditional_ssf(
        design.data,
        predictors=design.predictors,
        id_col="id",
        offset_col=design.offset_col,
        engine="fast",
        method="lbfgs",
        maxiter=500,
    )
    assert result.converged
    assert np.isfinite(result.params.to_numpy()).all()
    assert np.isfinite(model.loglike(result.params))


def test_offset_is_invariant_to_stratum_specific_additive_constants():
    design = _design()
    data = design.data.copy()
    data["shifted_offset"] = (
        data["proposal_offset"]
        + data.groupby(["id", "stratum_id"]).ngroup().astype(float)
    )

    _, reference = fit_conditional_ssf(
        data,
        predictors=design.predictors,
        id_col="id",
        offset_col="proposal_offset",
        engine="fast",
        method="lbfgs",
        maxiter=500,
    )
    _, shifted = fit_conditional_ssf(
        data,
        predictors=design.predictors,
        id_col="id",
        offset_col="shifted_offset",
        engine="fast",
        method="lbfgs",
        maxiter=500,
    )
    np.testing.assert_allclose(
        reference.params,
        shifted.params,
        rtol=1e-5,
        atol=1e-7,
    )


def test_hierarchical_issf_model_contains_fixed_offset():
    pytest.importorskip("pymc")
    design = _design()
    arrays = build_ssf_choice_arrays(
        design.data,
        id_col="id",
        predictors=design.predictors,
        offset_col=design.offset_col,
    )
    model = build_hierarchical_issf_model(arrays)
    assert "offset" in model.named_vars
    assert "mu_beta" in model.named_vars
    assert "sigma_beta" in model.named_vars
    assert "beta" in model.named_vars
