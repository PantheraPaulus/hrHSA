"""Integrated step-selection workflows.

The iSSF layer extends movement-informed SSF choice sets with explicit movement
terms, start-conditioned movement interactions, and proposal-density correction.
Environmental endpoint predictors are standardized across retained choice rows,
start-condition predictors are standardized once per stratum, and movement terms
remain on their natural scale so Gamma-like step-length interpretations are
preserved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from hsa.ssf.base import SSFAnalysis, SSFFit
from hsa.ssf.bayesian_diagnostics import (
    plot_bayesian_ssf_diagnostics,
    plot_bayesian_ssf_forests,
    plot_bayesian_ssf_trace,
)
from hsa.ssf.data import (
    SSFChoiceArrays,
    apply_ssf_scaling,
    build_ssf_choice_arrays,
    complete_ssf_strata,
    fit_ssf_scaling,
    score_choice_probabilities,
    stable_softmax,
)
from hsa.ssf.environment import (
    add_movement_terms,
    check_raster_coverage,
    sample_dynamic_covariates_at_points,
    sample_static_covariates_batched,
)
from hsa.ssf.frequentist import fit_conditional_ssf

_UNSET = object()

def _as_tuple(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(dict.fromkeys(map(str, values)))


def _normalize_start_band_mapping(
    bands: Sequence[str] | Mapping[str, str] | str,
) -> dict[str, str]:
    if isinstance(bands, str):
        return {bands: f"{bands}_start"}
    if isinstance(bands, Mapping):
        mapping = {str(source): str(output) for source, output in bands.items()}
    else:
        mapping = {str(source): f"{source}_start" for source in bands}
    if not mapping:
        raise ValueError("At least one start covariate is required.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Start-covariate output names must be unique.")
    return mapping


def issf_interaction_name(start_predictor: str, movement_term: str) -> str:
    """Return the canonical start-condition x movement-term column name."""
    label = "step_length" if movement_term == "step_length_km" else movement_term
    return f"{start_predictor}_x_{label}"


@dataclass(frozen=True)
class ISSFDesign:
    """Model-ready iSSF table plus the metadata needed for interpretation."""

    data: pd.DataFrame
    predictors: tuple[str, ...]
    endpoint_predictors: tuple[str, ...]
    start_predictors: tuple[str, ...]
    directional_predictors: tuple[str, ...]
    movement_terms: tuple[str, ...]
    interaction_terms: tuple[str, ...]
    interaction_columns: dict[tuple[str, str], str]
    scaling: dict[str, dict[str, float]]
    offset_col: str | None
    proposal_logpdf_col: str | None
    id_col: str
    stratum_col: str
    n_choices: int
    diagnostics: dict[str, Any]

    def summary(self) -> pd.Series:
        """Return compact design-size and exclusion diagnostics."""
        return pd.Series(self.diagnostics)

def prepare_issf_design(
    choices: pd.DataFrame,
    *,
    endpoint_predictors: Sequence[str],
    start_predictors: Sequence[str],
    directional_predictors: Sequence[str] = (),
    id_col: str,
    stratum_col: str = "stratum_id",
    used_col: str = "used",
    expected_n_choices: int | None = None,
    movement_terms: Sequence[str] = (
        "step_length_km",
        "log_step_length",
        "cos_turn_angle",
    ),
    interaction_terms: Sequence[str] = (
        "step_length_km",
        "log_step_length",
    ),
    proposal_logpdf_col: str | None = "proposal_logpdf",
    offset_col: str = "proposal_offset",
    center_offset: bool = True,
    scaling: Mapping[str, Mapping[str, float]] | None = None,
) -> ISSFDesign:
    """Prepare one common-scale, proposal-corrected iSSF design table.

    Endpoint predictors are candidate-specific and standardized over retained
    choice rows.

    Start predictors are constant within a choice stratum and are standardized
    over one row per retained stratum. Their main effects are not included
    because they cancel from the conditional likelihood.

    Directional predictors are conditions defined at the step start but whose
    value varies among candidates because they depend on candidate direction,
    e.g. wind support. They are standardized over retained choice rows and
    enter both as main effects and, optionally, as movement interactions.

    Movement terms are not standardized.

    Each requested start or directional predictor is interacted with each
    ``interaction_term``.

    When ``proposal_logpdf_col`` is supplied, the fixed importance-sampling
    offset is ``-log(q)``. It is optionally centered within strata, which is
    exactly likelihood-invariant and improves numerical conditioning.
    """

    endpoint_predictors = _as_tuple(endpoint_predictors)
    start_predictors = _as_tuple(start_predictors)
    directional_predictors = _as_tuple(directional_predictors)
    movement_terms = _as_tuple(movement_terms)
    interaction_terms = _as_tuple(interaction_terms)

    # ------------------------------------------------------------
    # Validate movement interaction specification
    # ------------------------------------------------------------

    unknown_interactions = sorted(
        set(interaction_terms).difference(movement_terms)
    )

    if unknown_interactions:
        raise ValueError(
            "interaction_terms must also be present in movement_terms: "
            f"{unknown_interactions}"
        )


    # ------------------------------------------------------------
    # Movement terms
    # ------------------------------------------------------------

    d = add_movement_terms(choices)

    if "step_length_km" in set(movement_terms) | set(interaction_terms):

        if "step_length" not in d:
            raise KeyError(
                "'step_length' is required to derive "
                "'step_length_km'."
            )

        d["step_length_km"] = (
            pd.to_numeric(
                d["step_length"],
                errors="coerce",
            )
            / 1000.0
        )


    # ------------------------------------------------------------
    # Remove incomplete strata
    # ------------------------------------------------------------

    raw_required = [
        *endpoint_predictors,
        *start_predictors,
        *directional_predictors,
        *movement_terms,
    ]

    if proposal_logpdf_col is not None:
        raw_required.append(
            proposal_logpdf_col
        )

    n_strata_before = int(
        d[
            [id_col, stratum_col]
        ]
        .drop_duplicates()
        .shape[0]
    )

    complete = complete_ssf_strata(
        d,
        predictors=raw_required,
        id_col=id_col,
        stratum_col=stratum_col,
        used_col=used_col,
        expected_n_choices=expected_n_choices,
    )


    # ------------------------------------------------------------
    # Unique start table
    #
    # Only true stratum-level start predictors go here.
    # Directional predictors must NOT go here because they vary
    # among alternatives within a stratum.
    # ------------------------------------------------------------

    start_table = (
        complete[
            [
                id_col,
                stratum_col,
                *start_predictors,
            ]
        ]
        .drop_duplicates(
            [id_col, stratum_col]
        )
        .copy()
    )


    # ------------------------------------------------------------
    # Fit / recover scaling
    # ------------------------------------------------------------

    # Endpoint + directional predictors are both candidate-level.
    choice_level_predictors = (
        *endpoint_predictors,
        *directional_predictors,
    )

    all_scaled_predictors = (
        *choice_level_predictors,
        *start_predictors,
    )

    if scaling is None:

        choice_scaling = fit_ssf_scaling(
            complete,
            choice_level_predictors,
        )

        start_scaling = fit_ssf_scaling(
            start_table,
            start_predictors,
        )

        fitted_scaling = {
            **choice_scaling,
            **start_scaling,
        }

    else:

        missing_scaling = [
            predictor
            for predictor in all_scaled_predictors
            if predictor not in scaling
        ]

        if missing_scaling:
            raise KeyError(
                "Scaling metadata missing for iSSF predictors: "
                f"{missing_scaling}"
            )

        fitted_scaling = {
            predictor: {
                "mean": float(
                    scaling[predictor]["mean"]
                ),
                "sd": float(
                    scaling[predictor]["sd"]
                ),
            }
            for predictor in all_scaled_predictors
        }


    # ------------------------------------------------------------
    # Apply common scaling
    # ------------------------------------------------------------

    complete = apply_ssf_scaling(
        complete,
        all_scaled_predictors,
        fitted_scaling,
    )


    # ------------------------------------------------------------
    # Movement interactions
    # ------------------------------------------------------------

    interaction_columns: dict[
        tuple[str, str],
        str,
    ] = {}


    # Ordinary start predictors:
    # e.g. elevation_start × step length
    for start_predictor in start_predictors:

        start_z = (
            f"{start_predictor}_z"
        )

        for movement_term in interaction_terms:

            column = issf_interaction_name(
                start_predictor,
                movement_term,
            )

            complete[column] = (
                complete[start_z]
                * complete[movement_term]
            )

            interaction_columns[
                (
                    start_predictor,
                    movement_term,
                )
            ] = column


    # Directional start predictors:
    # e.g. wind_start_support × step length
    for directional_predictor in directional_predictors:

        directional_z = (
            f"{directional_predictor}_z"
        )

        for movement_term in interaction_terms:

            column = issf_interaction_name(
                directional_predictor,
                movement_term,
            )

            complete[column] = (
                complete[directional_z]
                * complete[movement_term]
            )

            interaction_columns[
                (
                    directional_predictor,
                    movement_term,
                )
            ] = column


    # ------------------------------------------------------------
    # Final model predictor vector
    #
    # Endpoint predictors       -> main effects
    # Directional predictors    -> main effects
    # Start predictors          -> interactions only
    # Movement terms            -> main effects
    # ------------------------------------------------------------

    model_predictors = (
        *(
            f"{predictor}_z"
            for predictor
            in endpoint_predictors
        ),

        *(
            f"{predictor}_z"
            for predictor
            in directional_predictors
        ),

        *movement_terms,

        *interaction_columns.values(),
    )


    # ------------------------------------------------------------
    # Proposal correction
    # ------------------------------------------------------------

    model_offset_col: str | None = None

    if proposal_logpdf_col is not None:

        complete[offset_col] = (
            -pd.to_numeric(
                complete[
                    proposal_logpdf_col
                ],
                errors="coerce",
            )
        )

        if center_offset:

            complete[offset_col] = (
                complete[offset_col]
                - complete.groupby(
                    [
                        id_col,
                        stratum_col,
                    ],
                    sort=False,
                )[offset_col]
                .transform("mean")
            )

        model_offset_col = offset_col


    # ------------------------------------------------------------
    # Final completeness check
    # ------------------------------------------------------------

    final_required = list(
        model_predictors
    )

    if model_offset_col is not None:
        final_required.append(
            model_offset_col
        )

    complete = complete_ssf_strata(
        complete,
        predictors=final_required,
        id_col=id_col,
        stratum_col=stratum_col,
        used_col=used_col,
        expected_n_choices=expected_n_choices,
    )


    # Important: calculate retained strata AFTER final filtering
    n_strata_after = int(
        complete[
            [id_col, stratum_col]
        ]
        .drop_duplicates()
        .shape[0]
    )


    # ------------------------------------------------------------
    # Constant choice-set size
    # ------------------------------------------------------------

    sizes = (
        complete.groupby(
            [
                id_col,
                stratum_col,
            ],
            sort=False,
        )
        .size()
        .unique()
    )

    if len(sizes) != 1:
        raise ValueError(
            "iSSF design requires a constant choice-set size."
        )

    n_choices = int(
        sizes[0]
    )


    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    diagnostics = {
        "n_strata_before":
            n_strata_before,

        "n_strata_retained":
            n_strata_after,

        "n_strata_excluded":
            n_strata_before
            - n_strata_after,

        "retained_fraction":
            n_strata_after
            / max(
                n_strata_before,
                1,
            ),

        "n_choice_rows":
            int(
                len(complete)
            ),

        "n_choices":
            n_choices,

        "n_individuals":
            int(
                complete[
                    id_col
                ].nunique()
            ),

        "proposal_correction":
            proposal_logpdf_col
            is not None,

        "offset_centered_within_stratum":
            bool(
                proposal_logpdf_col
                is not None
                and center_offset
            ),

        "n_endpoint_predictors":
            len(
                endpoint_predictors
            ),

        "n_start_predictors":
            len(
                start_predictors
            ),

        "n_directional_predictors":
            len(
                directional_predictors
            ),
    }


    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    return ISSFDesign(
        data=complete,
        predictors=tuple(
            model_predictors
        ),

        endpoint_predictors=(
            endpoint_predictors
        ),

        start_predictors=(
            start_predictors
        ),

        directional_predictors=(
            directional_predictors
        ),

        movement_terms=(
            movement_terms
        ),

        interaction_terms=(
            interaction_terms
        ),

        interaction_columns=(
            interaction_columns
        ),

        scaling=fitted_scaling,

        offset_col=(
            model_offset_col
        ),

        proposal_logpdf_col=(
            proposal_logpdf_col
        ),

        id_col=id_col,
        stratum_col=stratum_col,
        n_choices=n_choices,
        diagnostics=diagnostics,
    )

def _movement_modifiers(
    design: ISSFDesign,
) -> tuple[str, ...]:
    """Predictors allowed to modify the movement kernel."""

    directional = getattr(
        design,
        "directional_predictors",
        (),
    )

    return tuple(
        dict.fromkeys(
            (
                *design.start_predictors,
                *directional,
            )
        )
    )

def _movement_parameters(
    params: Mapping[str, float] | pd.Series,
    design: ISSFDesign,
    conditions: Mapping[str, float] | None = None,
) -> tuple[float, float]:
    conditions = {} if conditions is None else dict(conditions)
    if "step_length_km" not in design.movement_terms:
        raise ValueError(
            "Expected-step diagnostics require movement term 'step_length_km'."
        )
    if "log_step_length" not in design.movement_terms:
        raise ValueError(
            "Expected-step diagnostics require movement term 'log_step_length'."
        )

    gamma_length = float(params["step_length_km"])
    gamma_log_length = float(params["log_step_length"])

    for predictor in _movement_modifiers(design):
        value = float(conditions.get(predictor, 0.0))
        length_col = design.interaction_columns.get(
            (predictor, "step_length_km")
        )
        log_col = design.interaction_columns.get(
            (predictor, "log_step_length")
        )
        if length_col is not None:
            gamma_length += float(params[length_col]) * value
        if log_col is not None:
            gamma_log_length += float(params[log_col]) * value

    shape = 1.0 + gamma_log_length
    rate = -gamma_length
    return shape, rate

def _turning_parameter(
    params: Mapping[str, float] | pd.Series,
    design: ISSFDesign,
    conditions: Mapping[str, float] | None = None,
) -> float:
    """Return the effective cos(turn-angle) coefficient."""

    conditions = (
        {}
        if conditions is None
        else dict(conditions)
    )

    if (
        "cos_turn_angle"
        not in design.movement_terms
    ):
        raise ValueError(
            "Turning-angle diagnostics require "
            "'cos_turn_angle' in movement_terms."
        )

    kappa = float(
        params["cos_turn_angle"]
    )

    for predictor in _movement_modifiers(
        design
    ):

        value = float(
            conditions.get(
                predictor,
                0.0,
            )
        )

        column = (
            design.interaction_columns.get(
                (
                    predictor,
                    "cos_turn_angle",
                )
            )
        )

        if column is not None:
            kappa += (
                float(params[column])
                * value
            )

    return kappa

def _turning_density(
    theta: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Normalized exp(kappa * cos(theta)) turning kernel."""

    log_density = (
        kappa * np.cos(theta)
    )

    # Numerical stability
    log_density -= np.max(
        log_density
    )

    density = np.exp(
        log_density
    )

    density /= np.trapezoid(
        density,
        theta,
    )

    return density

def _expected_step_gradient(
    params: pd.Series,
    design: ISSFDesign,
    conditions: Mapping[str, float] | None = None,
) -> tuple[float, np.ndarray]:
    conditions = {} if conditions is None else dict(conditions)
    shape, rate = _movement_parameters(params, design, conditions)
    if shape <= 0 or rate <= 0:
        return np.nan, np.full(len(params), np.nan)

    mean = shape / rate
    gradient = np.zeros(len(params), dtype=float)
    lookup = {name: i for i, name in enumerate(params.index)}
    gradient[lookup["step_length_km"]] = shape / rate**2
    gradient[lookup["log_step_length"]] = 1.0 / rate

    for predictor in _movement_modifiers(design):
        value = float(conditions.get(predictor, 0.0))
        length_col = design.interaction_columns.get(
            (predictor, "step_length_km")
        )
        log_col = design.interaction_columns.get(
            (predictor, "log_step_length")
        )
        if length_col is not None:
            gradient[lookup[length_col]] = value * shape / rate**2
        if log_col is not None:
            gradient[lookup[log_col]] = value / rate

    return mean, gradient


@dataclass
class FrequentistISSFFit(SSFFit):
    """Fitted pooled proposal-corrected iSSF."""

    model: Any
    result: Any
    design: ISSFDesign
    id_col: str

    def summary(self, *args, **kwargs):
        return self.result.summary(*args, **kwargs)

    def coefficients(self, *, alpha: float = 0.05) -> pd.DataFrame:
        ci = pd.DataFrame(self.result.conf_int(alpha=alpha))
        rows = []
        for predictor in self.design.predictors:
            rows.append(
                {
                    "predictor": predictor,
                    "beta": float(self.result.params[predictor]),
                    "se": float(self.result.bse[predictor]),
                    "lower": float(ci.loc[predictor].iloc[0]),
                    "upper": float(ci.loc[predictor].iloc[1]),
                    "p": float(self.result.pvalues[predictor]),
                }
            )
        return pd.DataFrame(rows)

    def choice_scores(self):
        arrays = build_ssf_choice_arrays(
            self.design.data,
            id_col=self.id_col,
            predictors=self.design.predictors,
            stratum_col=self.design.stratum_col,
            offset_col=self.design.offset_col,
            dtype="float64",
        )
        beta = self.result.params.to_numpy(dtype=float)
        eta = np.einsum(
            "sjp,p->sj",
            np.asarray(arrays.X, dtype=float),
            beta,
            optimize=True,
        )
        if arrays.offset is not None:
            eta = eta + np.asarray(arrays.offset, dtype=float)
        probability = stable_softmax(eta, axis=1)
        per_stratum, summary = score_choice_probabilities(
            probability,
            arrays.chosen,
        )
        per_stratum = pd.concat(
            [
                arrays.strata[
                    [self.id_col, self.design.stratum_col]
                ].reset_index(drop=True),
                per_stratum.reset_index(drop=True),
            ],
            axis=1,
        )
        return {
            "per_stratum": per_stratum,
            "summary": summary,
            "mean_probability": probability,
        }

    def movement_parameters(
        self,
        **start_conditions_z: float,
    ) -> pd.Series:
        shape, rate = _movement_parameters(
            self.result.params,
            self.design,
            start_conditions_z,
        )
        return pd.Series(
            {
                "shape": shape,
                "rate_per_km": rate,
                "scale_km": np.nan if rate <= 0 else 1.0 / rate,
                "expected_step_km": (
                    np.nan if shape <= 0 or rate <= 0 else shape / rate
                ),
            }
        )

    def expected_step_length(
        self,
        **start_conditions_z: float,
    ) -> float:
        return float(
            self.movement_parameters(
                **start_conditions_z
            )["expected_step_km"]
        )

    def plot_movement_response(
        self,
        predictor: str,
        *,
        moderator: str | None = None,
        moderator_levels: Sequence[float] = (-1.0, 0.0, 1.0),
        grid: Sequence[float] | None = None,
        ci: float = 0.95,
        ax=None,
    ):
        """Plot expected net displacement against one start-condition predictor."""
        modifiers = _movement_modifiers(
            self.design
        )

        if predictor not in modifiers:
            raise KeyError(
                f"{predictor!r} is not a movement-modifying "
                "predictor in this iSSF design."
            )

        if (
            moderator is not None
            and moderator not in modifiers
        ):
            raise KeyError(
                f"{moderator!r} is not a movement-modifying "
                "predictor in this iSSF design."
            )

        if moderator == predictor:
            raise ValueError(
                "moderator must differ from predictor."
            )

        if not 0 < ci < 1:
            raise ValueError(
                "ci must lie in (0, 1)."
            )

        import matplotlib.pyplot as plt
        from scipy.stats import norm

        if grid is None:
            grid_values = np.linspace(-2.0, 2.0, 200)
        else:
            grid_values = np.asarray(grid, dtype=float)
        levels = (0.0,) if moderator is None else tuple(moderator_levels)

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 5.0))

        covariance = self.result.cov_params().loc[
            self.result.params.index,
            self.result.params.index,
        ].to_numpy(dtype=float)
        critical = float(norm.ppf(0.5 + ci / 2.0))

        for level in levels:
            mean = np.empty(len(grid_values), dtype=float)
            se = np.empty(len(grid_values), dtype=float)
            for i, value in enumerate(grid_values):
                conditions = {predictor: float(value)}
                if moderator is not None:
                    conditions[moderator] = float(level)
                mean[i], gradient = _expected_step_gradient(
                    self.result.params,
                    self.design,
                    conditions,
                )
                if np.isfinite(gradient).all():
                    variance = float(gradient @ covariance @ gradient)
                    se[i] = np.sqrt(max(variance, 0.0))
                else:
                    se[i] = np.nan

            label = (
                "Expected displacement"
                if moderator is None
                else f"{moderator} = {level:+g} SD"
            )
            line, = ax.plot(
                grid_values,
                mean,
                linewidth=2,
                label=label,
            )
            ax.fill_between(
                grid_values,
                mean - critical * se,
                mean + critical * se,
                alpha=0.15,
                color=line.get_color(),
            )

        ax.axvline(0.0, linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel(f"{predictor} at step start [SD]")
        ax.set_ylabel("Expected net displacement [km]")
        ax.set_title("Environmental modulation of movement")
        if moderator is not None:
            ax.legend(title="Starting condition", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return ax

    def plot_movement_distributions(
        self,
        conditions: Mapping[str, Mapping[str, float]],
        *,
        max_km: float | None = None,
        n_points: int = 500,
        title = None,
        ax=None,
    ):
        """Plot fitted Gamma-like step-length densities for named conditions."""
        import matplotlib.pyplot as plt
        from scipy.stats import gamma

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 5.0))

        parameter_rows = []
        for label, condition in conditions.items():
            shape, rate = _movement_parameters(
                self.result.params,
                self.design,
                condition,
            )
            if shape <= 0 or rate <= 0:
                continue
            parameter_rows.append((label, shape, rate))

        if not parameter_rows:
            raise ValueError(
                "None of the requested conditions yields a proper Gamma kernel."
            )

        if max_km is None:
            max_km = max(
                float(gamma.ppf(0.99, a=shape, scale=1.0 / rate))
                for _, shape, rate in parameter_rows
            )
        x = np.linspace(max(float(max_km) / n_points, 1e-4), max_km, n_points)

        for label, shape, rate in parameter_rows:
            ax.plot(
                x,
                gamma.pdf(x, a=shape, scale=1.0 / rate),
                linewidth=2,
                label=str(label),
            )

        ax.set_xlabel("Net displacement [km]")
        ax.set_ylabel("Density")
        if title == None:   
            ax.set_title("Environment-dependent movement kernel")
        else:
            ax.set_title(title)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return ax

    def plot_step_length_distribution(
        self,
        predictor: str,
        *,
        levels: Sequence[float] = (
            -1.0,
            0.0,
            1.0,
        ),
        conditions: Mapping[str, float] | None = None,
        max_km: float | None = None,
        n_points: int = 500,
        ax=None,
    ):
        """
        Plot fitted step-length kernels across levels of one
        movement-modifying predictor.

        Predictor levels and conditions are expressed on the
        standardized scale.
        """

        modifiers = _movement_modifiers(
            self.design
        )

        if predictor not in modifiers:
            raise KeyError(
                f"{predictor!r} is not a movement-modifying "
                "predictor in this iSSF design."
            )

        base = (
            {}
            if conditions is None
            else dict(conditions)
        )

        named_conditions = {}

        for level in levels:

            condition = dict(base)

            condition[predictor] = float(
                level
            )

            named_conditions[
                f"{predictor} = {level:+g} SD"
            ] = condition

        return self.plot_movement_distributions(
            named_conditions,
            max_km=max_km,
            n_points=n_points,
            ax=ax,
        )

    def plot_turning_angle_distribution(
        self,
        predictor: str,
        *,
        levels: Sequence[float] = (
            -1.0,
            0.0,
            1.0,
        ),
        conditions: Mapping[str, float] | None = None,
        n_points: int = 500,
        ax=None,
    ):
        """
        Plot conditional turning-angle kernels across levels of
        one movement-modifying predictor.

        Predictor levels and conditions are on the standardized
        scale.
        """

        import matplotlib.pyplot as plt

        modifiers = _movement_modifiers(
            self.design
        )

        if predictor not in modifiers:
            raise KeyError(
                f"{predictor!r} is not a movement-modifying "
                "predictor in this iSSF design."
            )

        if ax is None:
            _, ax = plt.subplots(
                figsize=(7.5, 5.0)
            )

        theta = np.linspace(
            -np.pi,
            np.pi,
            n_points,
        )

        base = (
            {}
            if conditions is None
            else dict(conditions)
        )

        for level in levels:

            condition = dict(base)

            condition[predictor] = float(
                level
            )

            kappa = _turning_parameter(
                self.result.params,
                self.design,
                condition,
            )

            density = _turning_density(
                theta,
                kappa,
            )

            ax.plot(
                np.degrees(theta),
                density,
                linewidth=2,
                label=(
                    f"{predictor} = "
                    f"{level:+g} SD"
                ),
            )

        ax.axvline(
            0.0,
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )

        ax.set_xlabel(
            "Turning angle [degrees]"
        )

        ax.set_ylabel(
            "Density"
        )

        ax.set_title(
            "Environment-dependent turning kernel"
        )

        ax.legend(
            frameon=False
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        return ax

    def plot_turning_distribution(
        self,
        *,
        conditions=None,
        n_points: int = 500,
        ax=None,
    ):
        """
        Plot fitted turning-angle distributions.

        Conditions are given on the standardized scale of
        movement-modifying start or directional predictors.
        """

        import matplotlib.pyplot as plt

        if conditions is None:
            conditions = {
                "Movement kernel": {}
            }

        if ax is None:
            _, ax = plt.subplots(
                figsize=(7.5, 5.0)
            )

        theta = np.linspace(
            -np.pi,
            np.pi,
            n_points,
        )

        for label, condition in conditions.items():

            kappa = _turning_parameter(
                self.result.params,
                self.design,
                condition,
            )

            density = _turning_density(
                theta,
                kappa,
            )

            ax.plot(
                np.degrees(theta),
                density,
                linewidth=2,
                label=str(label),
            )

        ax.axvline(
            0.0,
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )

        ax.set_xlabel(
            "Turning angle [degrees]"
        )

        ax.set_ylabel(
            "Density"
        )

        ax.set_title(
            "Fitted turning-angle kernel"
        )

        if len(conditions) > 1:
            ax.legend(
                frameon=False
            )

        ax.spines["top"].set_visible(
            False
        )

        ax.spines["right"].set_visible(
            False
        )

        return ax


@dataclass
class IndividualISSFFits:
    """No-pooling individual iSSFs on one shared environmental scale."""

    summary: pd.DataFrame
    fits: dict[Any, FrequentistISSFFit]
    design: ISSFDesign
    id_col: str

    def plot_coefficients(
        self,
        predictor: str,
        *,
        alpha: float = 0.05,
        ax=None,
    ):
        """Forest plot of one no-pooling coefficient across individuals."""
        import matplotlib.pyplot as plt

        data = self.summary.loc[
            self.summary["predictor"].eq(predictor)
        ].copy()
        if data.empty:
            raise KeyError(f"{predictor!r} not found in individual fits.")
        data = data.sort_values("beta").reset_index(drop=True)

        if ax is None:
            _, ax = plt.subplots(
                figsize=(7.0, max(3.0, 0.55 * len(data) + 1.5))
            )

        y = np.arange(len(data))
        ax.errorbar(
            data["beta"],
            y,
            xerr=[
                data["beta"] - data["lower"],
                data["upper"] - data["beta"],
            ],
            fmt="o",
            capsize=3,
        )
        ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(data[self.id_col].astype(str))
        ax.set_xlabel("Coefficient")
        ax.set_title(predictor)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return ax


def build_hierarchical_issf_model(
    arrays: SSFChoiceArrays,
    *,
    mu_sigma: float = 1.0,
    heterogeneity_sigma: float = 0.5,
):
    """Build a non-centred hierarchical proposal-corrected iSSF in PyMC."""
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyMC is required; install hsa[bayesian].") from exc

    coords = {
        "individual": list(arrays.individuals),
        "predictor": list(arrays.predictors),
        "stratum": np.arange(arrays.X.shape[0]),
        "choice": np.arange(arrays.n_choices),
    }
    offset = (
        np.zeros(
            (arrays.X.shape[0], arrays.n_choices),
            dtype=np.asarray(arrays.X).dtype,
        )
        if arrays.offset is None
        else arrays.offset
    )

    with pm.Model(coords=coords) as model:
        X_data = pm.Data(
            "X",
            arrays.X,
            dims=("stratum", "choice", "predictor"),
        )
        offset_data = pm.Data(
            "offset",
            offset,
            dims=("stratum", "choice"),
        )
        individual_idx = pm.Data(
            "individual_idx",
            arrays.individual_idx,
            dims="stratum",
        )

        mu_beta = pm.Normal(
            "mu_beta",
            mu=0.0,
            sigma=float(mu_sigma),
            dims="predictor",
        )
        sigma_beta = pm.HalfNormal(
            "sigma_beta",
            sigma=float(heterogeneity_sigma),
            dims="predictor",
        )
        z_beta = pm.Normal(
            "z_beta",
            mu=0.0,
            sigma=1.0,
            dims=("individual", "predictor"),
        )
        beta = pm.Deterministic(
            "beta",
            mu_beta[None, :] + sigma_beta[None, :] * z_beta,
            dims=("individual", "predictor"),
        )

        beta_s = beta[individual_idx, :]
        eta = (
            (X_data * beta_s[:, None, :]).sum(axis=-1)
            + offset_data
        )
        pm.Categorical(
            "y",
            logit_p=eta,
            observed=arrays.chosen,
            dims="stratum",
        )
    return model


def _equal_tail_interval(
    values: np.ndarray,
    prob: float,
) -> tuple[float, float]:
    if not 0 < prob < 1:
        raise ValueError("ci_prob must lie in (0, 1).")
    tail = (1.0 - prob) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)


@dataclass
class BayesianISSFFit(SSFFit):
    """Fitted hierarchical Bayesian iSSF."""

    model: Any
    idata: Any
    arrays: SSFChoiceArrays
    design: ISSFDesign
    id_col: str

    def summary(
        self,
        *,
        ci_prob: float = 0.89,
        include_individual: bool = False,
    ):
        try:
            import arviz as az
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ArviZ is required; install hsa[bayesian].") from exc

        names = ["mu_beta", "sigma_beta"]
        if include_individual:
            names.append("beta")
        try:
            return az.summary(
                self.idata,
                var_names=names,
                ci_prob=ci_prob,
            )
        except TypeError:
            return az.summary(
                self.idata,
                var_names=names,
                hdi_prob=ci_prob,
            )

    def coefficients(
        self,
        *,
        ci_prob: float = 0.89,
    ) -> pd.DataFrame:
        rows = []
        for predictor in self.design.predictors:
            values = np.asarray(
                self.idata.posterior["mu_beta"]
                .sel(predictor=predictor)
                .values,
                dtype=float,
            ).ravel()
            lower, upper = _equal_tail_interval(values, ci_prob)
            rows.append(
                {
                    "predictor": predictor,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "lower": lower,
                    "upper": upper,
                    "p_gt_zero": float(np.mean(values > 0)),
                    "p_lt_zero": float(np.mean(values < 0)),
                }
            )
        return pd.DataFrame(rows)

    def individual_coefficients(
        self,
        *,
        ci_prob: float = 0.89,
    ) -> pd.DataFrame:
        rows = []
        beta = self.idata.posterior["beta"]
        for individual in self.arrays.individuals:
            for predictor in self.design.predictors:
                values = np.asarray(
                    beta.sel(
                        individual=individual,
                        predictor=predictor,
                    ).values,
                    dtype=float,
                ).ravel()
                lower, upper = _equal_tail_interval(values, ci_prob)
                rows.append(
                    {
                        self.id_col: individual,
                        "predictor": predictor,
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "median": float(np.median(values)),
                        "lower": lower,
                        "upper": upper,
                    }
                )
        return pd.DataFrame(rows)

    def heterogeneity(
        self,
        *,
        ci_prob: float = 0.89,
    ) -> pd.DataFrame:
        rows = []
        for predictor in self.design.predictors:
            values = np.asarray(
                self.idata.posterior["sigma_beta"]
                .sel(predictor=predictor)
                .values,
                dtype=float,
            ).ravel()
            lower, upper = _equal_tail_interval(values, ci_prob)
            rows.append(
                {
                    "predictor": predictor,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "lower": lower,
                    "upper": upper,
                }
            )
        return pd.DataFrame(rows)

    def choice_scores(
        self,
        *,
        batch_size: int = 250,
    ):
        beta = (
            self.idata.posterior["beta"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "individual", "predictor")
            .values
        )
        X = np.asarray(self.arrays.X, dtype=float)
        offset = (
            np.zeros((X.shape[0], X.shape[1]), dtype=float)
            if self.arrays.offset is None
            else np.asarray(self.arrays.offset, dtype=float)
        )
        probability = np.empty((X.shape[0], X.shape[1]), dtype=float)

        for start in range(0, X.shape[0], batch_size):
            stop = min(start + batch_size, X.shape[0])
            idx = self.arrays.individual_idx[start:stop]
            beta_batch = beta[:, idx, :]
            eta = np.einsum(
                "bjk,sbk->sbj",
                X[start:stop],
                beta_batch,
                optimize=True,
            )
            eta += offset[start:stop][None, :, :]
            eta -= eta.max(axis=-1, keepdims=True)
            p = np.exp(eta)
            p /= p.sum(axis=-1, keepdims=True)
            probability[start:stop] = p.mean(axis=0)

        per, summary = score_choice_probabilities(
            probability,
            self.arrays.chosen,
        )
        meta = self.arrays.strata.reset_index(drop=True)
        per[[self.id_col, self.design.stratum_col]] = meta[
            [self.id_col, self.design.stratum_col]
        ].to_numpy()
        return {
            "per_stratum": per,
            "summary": summary,
            "mean_probability": probability,
        }

    def plot_trace(
        self,
        *,
        include_individual: bool = False,
        individuals: Sequence[str] | None = None,
    ):
        return plot_bayesian_ssf_trace(
            self.idata,
            include_individual=include_individual,
            individuals=individuals,
        )

    def plot_forest(
        self,
        *,
        predictors: Sequence[str] | None = None,
        ci_prob: float = 0.95,
        include_population: bool = True,
        include_individual: bool = True,
        include_heterogeneity: bool = True,
    ):
        return plot_bayesian_ssf_forests(
            self.idata,
            predictors=None if predictors is None else list(predictors),
            ci_prob=ci_prob,
            include_population=include_population,
            include_individual=include_individual,
            include_heterogeneity=include_heterogeneity,
        )

    def plot_diagnostics(
        self,
        *,
        forest_predictors: Sequence[str] | None = None,
        ci_prob: float = 0.95,
        include_population_forest: bool = True,
        include_heterogeneity_forest: bool = True,
        include_individual_forests: bool = True,
        include_individual_trace: bool = False,
        trace_individuals: Sequence[str] | None = None,
    ):
        return plot_bayesian_ssf_diagnostics(
            self.idata,
            forest_predictors=(
                None if forest_predictors is None else list(forest_predictors)
            ),
            ci_prob=ci_prob,
            include_population_forest=include_population_forest,
            include_heterogeneity_forest=include_heterogeneity_forest,
            include_individual_forests=include_individual_forests,
            include_individual_trace=include_individual_trace,
            trace_individuals=trace_individuals,
        )

    def attach_fit(self, idata, *, validate: bool = True) -> BayesianISSFFit:
        """
        Attach externally sampled InferenceData to this iSSF analysis.

        The current analysis supplies the existing iSSF design, scaling,
        choice arrays, and model configuration. Only the posterior sampling
        result is loaded from disk.
        """

        import arviz as az


        # --------------------------------------------------------
        # Recover the design already belonging to this analysis
        # --------------------------------------------------------

        if self.issf_design_ is None:
            design = self.prepare_design()
        else:
            design = self.issf_design_


        # --------------------------------------------------------
        # Recreate the cheap dense representation
        # --------------------------------------------------------

        arrays = build_ssf_choice_arrays(
            design.data,
            id_col=self.id_col,
            predictors=design.predictors,
            stratum_col=design.stratum_col,
            offset_col=design.offset_col,
            dtype="float32",
        )


        # --------------------------------------------------------
        # Safety checks
        # --------------------------------------------------------

        if validate:

            posterior = idata.posterior

            if "predictor" in posterior.coords:

                stored_predictors = tuple(
                    map(
                        str,
                        posterior.coords[
                            "predictor"
                        ].values,
                    )
                )

                expected_predictors = tuple(
                    map(
                        str,
                        design.predictors,
                    )
                )

                if stored_predictors != expected_predictors:
                    raise ValueError(
                        "Posterior predictors do not match the "
                        "current iSSF design.\n"
                        f"Posterior: {stored_predictors}\n"
                        f"Current:   {expected_predictors}"
                    )


            if "individual" in posterior.coords:

                stored_individuals = tuple(
                    map(
                        str,
                        posterior.coords[
                            "individual"
                        ].values,
                    )
                )

                expected_individuals = tuple(
                    map(
                        str,
                        arrays.individuals,
                    )
                )

                if stored_individuals != expected_individuals:
                    raise ValueError(
                        "Posterior individuals do not match the "
                        "current iSSF analysis.\n"
                        f"Posterior: {stored_individuals}\n"
                        f"Current:   {expected_individuals}"
                    )


        # --------------------------------------------------------
        # Rebuild PyMC graph -- no sampling occurs
        # --------------------------------------------------------

        model = build_hierarchical_issf_model(
            arrays,
            **self.model_kwargs,
        )


        # --------------------------------------------------------
        # Construct normal fitted-object wrapper
        # --------------------------------------------------------

        fit = BayesianISSFFit(
            model=model,
            idata=idata,
            arrays=arrays,
            design=design,
            id_col=self.id_col,
        )

        self.fit_ = fit

        return fit

    def export_fit(self, path, *, overwrite: bool = False,
        scaling=None, center_offset: bool = True):
        """
        Export a frozen Bayesian iSSF fit specification.

        The export contains only the model-ready numeric choice table and
        metadata required to reconstruct the hierarchical PyMC model.
        Spatial objects, rasters, and relocation data are intentionally
        excluded.

        Parameters
        ----------
        path
            Directory in which to write the fit bundle.
        overwrite
            Allow replacement of existing exported files.
        scaling
            Optional externally supplied iSSF scaling. If omitted, use the
            currently prepared design or prepare a new one.
        center_offset
            Whether to centre the proposal offset within strata when a new
            design must be prepared.

        Returns
        -------
        pathlib.Path
            Export directory.
        """

        from pathlib import Path
        import json

        import numpy as np
        import pandas as pd


        path = Path(path)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        data_path = (
            path / "issf_design.parquet"
        )

        config_path = (
            path / "fit_config.json"
        )


        # --------------------------------------------------------
        # Protect existing exports
        # --------------------------------------------------------

        if not overwrite:

            existing = [
                p
                for p in (
                    data_path,
                    config_path,
                )
                if p.exists()
            ]

            if existing:
                raise FileExistsError(
                    "Fit export already exists: "
                    + ", ".join(
                        str(p)
                        for p in existing
                    )
                )


        # --------------------------------------------------------
        # Obtain the exact model design
        # --------------------------------------------------------

        if (
            self.issf_design_ is not None
            and scaling is None
        ):
            design = self.issf_design_

        else:
            design = self.prepare_design(
                scaling=scaling,
                center_offset=center_offset,
            )


        # --------------------------------------------------------
        # Export ONLY statistical columns
        # --------------------------------------------------------

        model_columns = [
            design.id_col,
            design.stratum_col,
            "candidate_id",
            "used",
            *design.predictors,
        ]

        if design.offset_col is not None:
            model_columns.append(
                design.offset_col
            )

        model_columns = list(
            dict.fromkeys(
                model_columns
            )
        )

        missing = [
            column
            for column in model_columns
            if column not in design.data
        ]

        if missing:
            raise KeyError(
                "Cannot export iSSF fit; "
                f"missing columns: {missing}"
            )


        model_data = pd.DataFrame(
            design.data[
                model_columns
            ].copy()
        )


        # --------------------------------------------------------
        # Final numeric / structural checks
        # --------------------------------------------------------

        numerical_columns = [
            *design.predictors,
        ]

        if design.offset_col is not None:
            numerical_columns.append(
                design.offset_col
            )

        if not np.isfinite(
            model_data[
                numerical_columns
            ].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Cannot export iSSF fit because the "
                "model design contains non-finite values."
            )


        check = (
            model_data
            .groupby(
                [
                    design.id_col,
                    design.stratum_col,
                ],
                sort=False,
            )
            .agg(
                n_choices=(
                    "candidate_id",
                    "size",
                ),
                n_used=(
                    "used",
                    "sum",
                ),
            )
        )

        if not (
            check["n_choices"]
            == design.n_choices
        ).all():
            raise ValueError(
                "Export contains incomplete choice sets."
            )

        if not (
            check["n_used"] == 1
        ).all():
            raise ValueError(
                "Every exported stratum must contain "
                "exactly one observed choice."
            )


        # --------------------------------------------------------
        # Write frozen design
        # --------------------------------------------------------

        model_data.to_parquet(
            data_path,
            index=False,
        )


        # --------------------------------------------------------
        # JSON-safe helper
        # --------------------------------------------------------

        def json_default(value):

            if isinstance(
                value,
                np.integer,
            ):
                return int(value)

            if isinstance(
                value,
                np.floating,
            ):
                return float(value)

            if isinstance(
                value,
                np.ndarray,
            ):
                return value.tolist()

            raise TypeError(
                f"{type(value).__name__} "
                "is not JSON serializable"
            )


        # --------------------------------------------------------
        # Model metadata
        # --------------------------------------------------------

        config = {
            "model": "BayesianISSF",

            "id_col":
                design.id_col,

            "stratum_col":
                design.stratum_col,

            "candidate_col":
                "candidate_id",

            "used_col":
                "used",

            "predictors":
                list(
                    design.predictors
                ),

            "endpoint_predictors":
                list(
                    design.endpoint_predictors
                ),

            "start_predictors":
                list(
                    design.start_predictors
                ),

            "movement_terms":
                list(
                    design.movement_terms
                ),

            "interaction_terms":
                list(
                    design.interaction_terms
                ),

            "interaction_columns": {
                f"{start}|{movement}":
                    column

                for (
                    start,
                    movement,
                ), column
                in design.interaction_columns.items()
            },

            "offset_col":
                design.offset_col,

            "proposal_logpdf_col":
                design.proposal_logpdf_col,

            "scaling":
                design.scaling,

            "n_choices":
                int(
                    design.n_choices
                ),

            "n_individuals":
                int(
                    model_data[
                        design.id_col
                    ].nunique()
                ),

            "n_strata":
                int(
                    len(check)
                ),

            "n_choice_rows":
                int(
                    len(model_data)
                ),

            "model_kwargs":
                dict(
                    self.model_kwargs
                ),
        }


        with config_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                config,
                f,
                indent=2,
                default=json_default,
            )


        # --------------------------------------------------------
        # Keep export provenance on the live object too
        # --------------------------------------------------------

        self.fit_export_ = {
            "path": str(path),
            "data": str(data_path),
            "config": str(config_path),
            "n_strata": len(check),
            "n_choice_rows":
                len(model_data),
        }


        return path

    def load_fit(self, path, *, validate=True):
        import arviz as az

        idata = az.from_netcdf(path)

        return self.attach_fit(
            idata,
            validate=validate,
        )

    def _posterior_parameter_draws(
        self,
        *,
        individual: str | None = None,
    ) -> pd.DataFrame:
        if individual is None:
            values = (
                self.idata.posterior["mu_beta"]
                .stack(sample=("chain", "draw"))
                .transpose("sample", "predictor")
            )
        else:
            values = (
                self.idata.posterior["beta"]
                .sel(individual=individual)
                .stack(sample=("chain", "draw"))
                .transpose("sample", "predictor")
            )
        return pd.DataFrame(
            np.asarray(values.values, dtype=float),
            columns=list(self.design.predictors),
        )

    def plot_movement_response(
        self,
        predictor: str,
        *,
        moderator: str | None = None,
        moderator_levels: Sequence[float] = (-1.0, 0.0, 1.0),
        grid: Sequence[float] | None = None,
        ci_prob: float = 0.95,
        individual: str | None = None,
        ax=None,
    ):
        """Posterior movement-response curves for population or one individual."""
        modifiers = _movement_modifiers(
            self.design
        )

        if predictor not in modifiers:
            raise KeyError(
                f"{predictor!r} is not a movement-modifying "
                "predictor in this iSSF design."
            )

        if (
            moderator is not None
            and moderator not in modifiers
        ):
            raise KeyError(
                f"{moderator!r} is not a movement-modifying "
                "predictor in this iSSF design."
            )

        if moderator == predictor:
            raise ValueError(
                "moderator must differ from predictor."
            )
        
        if moderator == predictor:
            raise ValueError("moderator must differ from predictor.")
        if not 0 < ci_prob < 1:
            raise ValueError("ci_prob must lie in (0, 1).")

        import matplotlib.pyplot as plt

        grid_values = (
            np.linspace(-2.0, 2.0, 200)
            if grid is None
            else np.asarray(grid, dtype=float)
        )
        levels = (0.0,) if moderator is None else tuple(moderator_levels)
        draws = self._posterior_parameter_draws(individual=individual)

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 5.0))

        tail = (1.0 - ci_prob) / 2.0
        for level in levels:
            response = np.full(
                (len(draws), len(grid_values)),
                np.nan,
                dtype=float,
            )
            for j, value in enumerate(grid_values):
                conditions = {predictor: float(value)}
                if moderator is not None:
                    conditions[moderator] = float(level)

                gamma_length = draws["step_length_km"].to_numpy(dtype=float).copy()
                gamma_log = draws["log_step_length"].to_numpy(dtype=float).copy()
                for start_predictor in self.design.start_predictors:
                    z = float(conditions.get(start_predictor, 0.0))
                    length_col = self.design.interaction_columns.get(
                        (start_predictor, "step_length_km")
                    )
                    log_col = self.design.interaction_columns.get(
                        (start_predictor, "log_step_length")
                    )
                    if length_col is not None:
                        gamma_length += draws[length_col].to_numpy(dtype=float) * z
                    if log_col is not None:
                        gamma_log += draws[log_col].to_numpy(dtype=float) * z

                shape = 1.0 + gamma_log
                rate = -gamma_length
                valid = (shape > 0) & (rate > 0)
                response[valid, j] = shape[valid] / rate[valid]

            median = np.nanmedian(response, axis=0)
            lower = np.nanquantile(response, tail, axis=0)
            upper = np.nanquantile(response, 1.0 - tail, axis=0)

            label = (
                "Population" if moderator is None and individual is None
                else (
                    str(individual)
                    if moderator is None
                    else f"{moderator} = {level:+g} SD"
                )
            )
            line, = ax.plot(
                grid_values,
                median,
                linewidth=2,
                label=label,
            )
            ax.fill_between(
                grid_values,
                lower,
                upper,
                alpha=0.15,
                color=line.get_color(),
            )

        ax.axvline(0.0, linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel(f"{predictor} at step start [SD]")
        ax.set_ylabel("Expected net displacement [km]")
        title_subject = "population" if individual is None else str(individual)
        ax.set_title(f"Environmental modulation of movement: {title_subject}")
        if moderator is not None:
            ax.legend(title="Starting condition", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return ax


class ISSFAnalysis(SSFAnalysis):
    """Shared stateful workflow for frequentist and Bayesian iSSFs."""

    def __init__(
        self,
        reloc: gpd.GeoDataFrame,
        env=None,
        *,
        endpoint_predictors: Sequence[str],
        start_predictors: Sequence[str],
        directional_predictors=(),
        movement_terms: Sequence[str] = (
            "step_length_km",
            "log_step_length",
            "cos_turn_angle",
        ),
        interaction_terms: Sequence[str] = (
            "step_length_km",
            "log_step_length",
        ),
        proposal_logpdf_col: str | None = "proposal_logpdf",
        id_col: str = "individual-local-identifier",
        timestamp_col: str = "Timestamp",
        expected_interval_min: float | None = 60,
        tolerance_min: float = 10,
        round_freq: str | None = "h",
        n_available: int = 20,
        speed_margin: float = 1.05,
        burst_gap: str = "2h",
        choices: gpd.GeoDataFrame | None = None,
    ):
        self.endpoint_predictors = _as_tuple(endpoint_predictors)
        self.start_predictors = _as_tuple(start_predictors)
        self.directional_predictors = _as_tuple(directional_predictors)
        self.movement_terms = _as_tuple(movement_terms)
        self.interaction_terms = _as_tuple(interaction_terms)
        self.proposal_logpdf_col = proposal_logpdf_col
        self.issf_design_: ISSFDesign | None = None
        self.start_annotation_diagnostics_: dict[str, Any] = {}
        super().__init__(
            reloc,
            env,
            predictors=self.endpoint_predictors,
            id_col=id_col,
            timestamp_col=timestamp_col,
            expected_interval_min=expected_interval_min,
            tolerance_min=tolerance_min,
            round_freq=round_freq,
            n_available=n_available,
            speed_margin=speed_margin,
            burst_gap=burst_gap,
            choices=choices,
        )

    @classmethod
    def from_ssf(
        cls,
        analysis: SSFAnalysis,
        *,
        endpoint_predictors: Sequence[str],
        start_predictors: Sequence[str],
        **kwargs,
    ):
        """Reuse an existing SSF's relocations, environment, and choice sets."""
        options = {
            "id_col": analysis.id_col,
            "timestamp_col": analysis.timestamp_col,
            "expected_interval_min": analysis.expected_interval_min,
            "tolerance_min": analysis.tolerance_min,
            "round_freq": analysis.round_freq,
            "n_available": analysis.n_available,
            "speed_margin": analysis.speed_margin,
            "burst_gap": analysis.burst_gap,
            "choices": analysis.choices,
        }
        options.update(kwargs)
        return cls(
            analysis.reloc,
            analysis.env,
            endpoint_predictors=endpoint_predictors,
            start_predictors=start_predictors,
            **options,
        )

    def _unique_starts(self) -> gpd.GeoDataFrame:
        columns = [
            self.id_col,
            "stratum_id",
            "start_time",
            "start_geometry",
        ]
        missing = [column for column in columns if column not in self.choices]
        if missing:
            raise KeyError(f"Missing start columns: {missing}")
        starts = (
            self.choices[columns]
            .drop_duplicates([self.id_col, "stratum_id"])
            .copy()
        )
        return gpd.GeoDataFrame(
            starts,
            geometry="start_geometry",
            crs=self.choices.crs,
        )

    def set_predictors(
        self,
        *,
        endpoint_predictors=_UNSET,
        start_predictors=_UNSET,
        directional_predictors=_UNSET,
        movement_terms=_UNSET,
        interaction_terms=_UNSET,
    ):
        """
        Update the iSSF model specification.

        Omitted arguments retain their current values.
        ``None`` or an empty sequence clears that component.
        """

        def resolve(value, current):
            if value is _UNSET:
                return current

            if value is None:
                return ()

            return _as_tuple(value)


        endpoint = resolve(
            endpoint_predictors,
            self.endpoint_predictors,
        )

        start = resolve(
            start_predictors,
            self.start_predictors,
        )

        directional = resolve(
            directional_predictors,
            self.directional_predictors,
        )

        movement = resolve(
            movement_terms,
            self.movement_terms,
        )

        interactions = resolve(
            interaction_terms,
            self.interaction_terms,
        )


        # --------------------------------------------------------
        # Empty names should never enter the specification
        # --------------------------------------------------------

        groups = {
            "endpoint_predictors": endpoint,
            "start_predictors": start,
            "directional_predictors": directional,
            "movement_terms": movement,
            "interaction_terms": interactions,
        }

        for name, values in groups.items():

            empty = [
                value
                for value in values
                if not str(value).strip()
            ]

            if empty:
                raise ValueError(
                    f"{name} contains an empty predictor name."
                )


        # --------------------------------------------------------
        # Requested movement interactions must exist as movement
        # terms
        # --------------------------------------------------------

        unknown = sorted(
            set(interactions)
            - set(movement)
        )

        if unknown:
            raise ValueError(
                "interaction_terms must also be present in "
                f"movement_terms: {unknown}"
            )


        # It makes no sense to request interactions without
        # any environmental condition at the step start.
        if interactions and not start:
            raise ValueError(
                "interaction_terms were supplied but "
                "start_predictors is empty."
            )


        # --------------------------------------------------------
        # Store synchronized specification
        # --------------------------------------------------------

        self.endpoint_predictors = endpoint
        self.start_predictors = start
        self.directional_predictors = directional
        self.movement_terms = movement
        self.interaction_terms = interactions

        self.predictors = list(endpoint)

        self.issf_design_ = None
        self.fit_ = None

        if hasattr(self, "validation_"):
            self.validation_ = None

        return self
    
    def _merge_start_values(
        self,
        starts: pd.DataFrame,
        columns: Sequence[str],
    ) -> None:
        drop_existing = [
            column for column in columns
            if column in self.choices.columns
        ]
        base = self.choices.drop(columns=drop_existing)
        values = starts[
            [self.id_col, "stratum_id", *columns]
        ].copy()
        merged = base.merge(
            values,
            on=[self.id_col, "stratum_id"],
            how="left",
            validate="many_to_one",
        )
        self.choices_ = gpd.GeoDataFrame(
            merged,
            geometry="geometry",
            crs=self.choices.crs,
        )
        self.issf_design_ = None

    def annotate_start_static(
        self,
        *,
        bands: Sequence[str] | Mapping[str, str] | str,
        batch_size: int = 100_000,
        outside: str = "nan",
    ) -> "ISSFAnalysis":
        """Sample static covariates once at the shared start of each stratum.

        ``outside='nan'`` keeps unsupported starts as missing so the subsequent
        design preparation can exclude the whole stratum. ``outside='raise'``
        instead stops immediately.
        """
        if self.env is None:
            raise ValueError("No environmental raster is stored on this analysis.")
        if outside not in {"nan", "raise"}:
            raise ValueError("outside must be 'nan' or 'raise'.")

        mapping = _normalize_start_band_mapping(bands)
        starts = self._unique_starts()
        inside = check_raster_coverage(starts, self.env)

        if outside == "raise" and not inside.all():
            n_outside = int((~inside).sum())
            raise ValueError(
                f"{n_outside:,} iSSF start locations fall outside the raster extent."
            )

        for output in mapping.values():
            starts[output] = np.nan

        supported = starts.loc[inside].copy()
        if not supported.empty:
            sampled = sample_static_covariates_batched(
                supported,
                self.env,
                bands=list(mapping),
                batch_size=batch_size,
                require_inside=True,
            )
            for source, output in mapping.items():
                starts.loc[supported.index, output] = (
                    sampled[source].to_numpy()
                )

        self._merge_start_values(
            starts,
            list(mapping.values()),
        )
        self.start_annotation_diagnostics_["static"] = {
            "n_starts": int(len(starts)),
            "n_inside": int(inside.sum()),
            "n_outside": int((~inside).sum()),
            "outside_policy": outside,
            "variables": dict(mapping),
        }
        return self

    def annotate_start_dynamic(
        self,
        field,
        *,
        variables,
        time_col: str = "start_time",
        **kwargs,
    ) -> "ISSFAnalysis":
        """Sample dynamic covariates once at each stratum's start location/time."""
        starts = self._unique_starts()
        sampled = sample_dynamic_covariates_at_points(
            starts,
            field,
            variables=variables,
            time_col=time_col,
            **kwargs,
        )

        if isinstance(variables, str):
            outputs = [variables]
        elif isinstance(variables, Mapping):
            outputs = list(map(str, variables.values()))
        else:
            outputs = list(map(str, variables))

        self._merge_start_values(sampled, outputs)
        self.start_annotation_diagnostics_["dynamic"] = {
            "n_starts": int(len(sampled)),
            "variables": tuple(outputs),
            "time_col": time_col,
        }
        return self

    def prepare_design(
        self,
        *,
        scaling: Mapping[str, Mapping[str, float]] | None = None,
        center_offset: bool = True,
    ) -> ISSFDesign:
        """Prepare and retain the model-ready iSSF design table."""
        design = prepare_issf_design(
            self.choices,
            endpoint_predictors=self.endpoint_predictors,
            start_predictors=self.start_predictors,
            directional_predictors=self.directional_predictors,
            id_col=self.id_col,
            expected_n_choices=self.n_available + 1,
            movement_terms=self.movement_terms,
            interaction_terms=self.interaction_terms,
            proposal_logpdf_col=self.proposal_logpdf_col,
            center_offset=center_offset,
            scaling=scaling,
        )
        self.issf_design_ = design
        return design

    def validate(self, scheme, **kwargs):
        raise NotImplementedError(
            "iSSF cross-validation is not yet exposed because start-condition "
            "scaling and proposal correction must be fitted within each training fold."
        )
    
    def subset(self, *, n_strata_per_id: int | None = None, fraction: float | None = None,
               individuals=None, seed: int = 42):
        """
        Return a smaller iSSF analysis containing complete choice strata.

        Exactly one of ``n_strata_per_id`` or ``fraction`` may be supplied.
        Sampling is performed independently within each individual so that
        hierarchical structure is retained.

        The returned object has the same concrete class as ``self`` and
        reuses the already prepared/annotated choice sets. Model fits and
        prepared iSSF designs are intentionally reset.

        Parameters
        ----------
        n_strata_per_id
            Maximum number of complete strata sampled per individual.
        fraction
            Fraction of each individual's strata to retain, in (0, 1].
        individuals
            Optional sequence of individuals to retain. By default all
            individuals are retained.
        seed
            Random seed for reproducible stratum sampling.

        Returns
        -------
        ISSFAnalysis
            Independent smaller analysis object of the same concrete class.
        """

        if (
            n_strata_per_id is not None
            and fraction is not None
        ):
            raise ValueError(
                "Specify either n_strata_per_id or fraction, not both."
            )

        if n_strata_per_id is None and fraction is None:
            raise ValueError(
                "Specify n_strata_per_id or fraction."
            )

        if (
            n_strata_per_id is not None
            and n_strata_per_id <= 0
        ):
            raise ValueError(
                "n_strata_per_id must be positive."
            )

        if (
            fraction is not None
            and not 0 < fraction <= 1
        ):
            raise ValueError(
                "fraction must lie in (0, 1]."
            )

        choices = self.choices

        # --------------------------------------------------------
        # Optional individual restriction
        # --------------------------------------------------------

        if individuals is None:
            selected_individuals = list(
                choices[self.id_col].drop_duplicates()
            )
        else:
            selected_individuals = list(individuals)

            unknown = sorted(
                set(selected_individuals)
                - set(choices[self.id_col].unique())
            )

            if unknown:
                raise KeyError(
                    f"Unknown individuals: {unknown}"
                )

        choices = choices.loc[
            choices[self.id_col].isin(
                selected_individuals
            )
        ].copy()


        # --------------------------------------------------------
        # One row per complete stratum
        # --------------------------------------------------------

        strata = (
            choices[
                [
                    self.id_col,
                    "stratum_id",
                ]
            ]
            .drop_duplicates()
        )


        # --------------------------------------------------------
        # Sample strata independently within individuals
        # --------------------------------------------------------

        rng = np.random.default_rng(seed)

        selected_parts = []

        for _, group in strata.groupby(
            self.id_col,
            sort=False,
        ):

            n_available = len(group)

            if n_strata_per_id is not None:

                n_keep = min(
                    int(n_strata_per_id),
                    n_available,
                )

            else:

                n_keep = max(
                    1,
                    int(
                        np.ceil(
                            n_available
                            * float(fraction)
                        )
                    ),
                )

            selected_index = rng.choice(
                group.index.to_numpy(),
                size=n_keep,
                replace=False,
            )

            selected_parts.append(
                group.loc[selected_index]
            )


        selected_strata = pd.concat(
            selected_parts,
            ignore_index=True,
        )


        # --------------------------------------------------------
        # Retain ALL candidates belonging to sampled strata
        # --------------------------------------------------------

        choice_keys = pd.MultiIndex.from_frame(
            choices[
                [
                    self.id_col,
                    "stratum_id",
                ]
            ]
        )

        selected_keys = pd.MultiIndex.from_frame(
            selected_strata[
                [
                    self.id_col,
                    "stratum_id",
                ]
            ]
        )

        subset_choices = (
            choices.loc[
                choice_keys.isin(
                    selected_keys
                )
            ]
            .copy()
        )


        # --------------------------------------------------------
        # Structural safety check
        # --------------------------------------------------------

        check = (
            subset_choices
            .groupby(
                [
                    self.id_col,
                    "stratum_id",
                ],
                sort=False,
            )
            .agg(
                n_choices=(
                    "candidate_id",
                    "size",
                ),
                n_used=(
                    "used",
                    "sum",
                ),
            )
        )

        expected_n_choices = (
            self.n_available + 1
        )

        if not (
            check["n_choices"]
            == expected_n_choices
        ).all():
            raise RuntimeError(
                "Subsetting produced incomplete iSSF strata."
            )

        if not (
            check["n_used"] == 1
        ).all():
            raise RuntimeError(
                "Subsetting produced strata without exactly one "
                "observed choice."
            )


        # --------------------------------------------------------
        # Keep relocation metadata for retained individuals
        # --------------------------------------------------------

        subset_reloc = (
            self.reloc.loc[
                self.reloc[self.id_col].isin(
                    selected_individuals
                )
            ]
            .copy()
        )


        # --------------------------------------------------------
        # Reconstruct same concrete iSSF class
        # --------------------------------------------------------

        kwargs = dict(
            endpoint_predictors=(
                self.endpoint_predictors
            ),
            start_predictors=(
                self.start_predictors
            ),
            movement_terms=(
                self.movement_terms
            ),
            interaction_terms=(
                self.interaction_terms
            ),
            proposal_logpdf_col=(
                self.proposal_logpdf_col
            ),
            id_col=self.id_col,
            timestamp_col=self.timestamp_col,
            expected_interval_min=(
                self.expected_interval_min
            ),
            tolerance_min=self.tolerance_min,
            round_freq=self.round_freq,
            n_available=self.n_available,
            speed_margin=self.speed_margin,
            burst_gap=self.burst_gap,
            choices=subset_choices,
        )

        # BayesianISSF has an additional constructor argument.
        if hasattr(self, "model_kwargs"):
            kwargs["model_kwargs"] = dict(
                self.model_kwargs
            )


        out = self.__class__(
            subset_reloc,
            self.env,
            **kwargs,
        )


        # --------------------------------------------------------
        # Record provenance
        # --------------------------------------------------------

        out.subset_diagnostics_ = {
            "source_n_individuals":
                int(self.choices[self.id_col].nunique()),

            "source_n_strata":
                int(
                    self.choices[
                        [
                            self.id_col,
                            "stratum_id",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),

            "n_individuals":
                int(
                    subset_choices[
                        self.id_col
                    ].nunique()
                ),

            "n_strata":
                int(len(check)),

            "n_choice_rows":
                int(len(subset_choices)),

            "n_strata_per_id":
                n_strata_per_id,

            "fraction":
                fraction,

            "seed":
                seed,
        }

        return out

class FrequentistISSF(ISSFAnalysis):
    """Pooled frequentist integrated step-selection analysis."""

    def fit(
        self,
        *,
        scaling: Mapping[str, Mapping[str, float]] | None = None,
        center_offset: bool = True,
        engine: str = "fast",
        method: str = "lbfgs",
        maxiter: int = 1000,
        disp: bool = False,
    ) -> FrequentistISSFFit:
        design = self.prepare_design(
            scaling=scaling,
            center_offset=center_offset,
        )
        model, result = fit_conditional_ssf(
            design.data,
            predictors=design.predictors,
            id_col=self.id_col,
            stratum_col=design.stratum_col,
            offset_col=design.offset_col,
            engine=engine,
            method=method,
            maxiter=maxiter,
            disp=disp,
        )
        fit = FrequentistISSFFit(
            model=model,
            result=result,
            design=design,
            id_col=self.id_col,
        )
        self.fit_ = fit
        return fit

    def fit_individuals(
        self,
        *,
        scaling: Mapping[str, Mapping[str, float]] | None = None,
        center_offset: bool = True,
        engine: str = "fast",
        method: str = "lbfgs",
        maxiter: int = 1000,
    ) -> IndividualISSFFits:
        design = self.prepare_design(
            scaling=scaling,
            center_offset=center_offset,
        )
        fits: dict[Any, FrequentistISSFFit] = {}
        frames = []

        for individual, group in design.data.groupby(
            self.id_col,
            sort=False,
        ):
            model, result = fit_conditional_ssf(
                group,
                predictors=design.predictors,
                id_col=self.id_col,
                stratum_col=design.stratum_col,
                offset_col=design.offset_col,
                engine=engine,
                method=method,
                maxiter=maxiter,
            )
            fit = FrequentistISSFFit(
                model=model,
                result=result,
                design=ISSFDesign(
                    data=group.copy(),
                    predictors=design.predictors,
                    endpoint_predictors=design.endpoint_predictors,
                    start_predictors=design.start_predictors,
                    movement_terms=design.movement_terms,
                    interaction_terms=design.interaction_terms,
                    interaction_columns=design.interaction_columns,
                    scaling=design.scaling,
                    offset_col=design.offset_col,
                    proposal_logpdf_col=design.proposal_logpdf_col,
                    id_col=design.id_col,
                    stratum_col=design.stratum_col,
                    n_choices=design.n_choices,
                    diagnostics={
                        **design.diagnostics,
                        "n_individuals": 1,
                        "n_strata_retained": int(
                            group[design.stratum_col].nunique()
                        ),
                        "n_choice_rows": int(len(group)),
                    },
                ),
                id_col=self.id_col,
            )
            fits[individual] = fit
            coefficients = fit.coefficients()
            coefficients[self.id_col] = individual
            coefficients["n_strata"] = int(
                group[design.stratum_col].nunique()
            )
            frames.append(coefficients)

        return IndividualISSFFits(
            summary=pd.concat(frames, ignore_index=True),
            fits=fits,
            design=design,
            id_col=self.id_col,
        )


class BayesianISSF(ISSFAnalysis):
    """Hierarchical Bayesian integrated step-selection analysis."""

    def __init__(
        self,
        *args,
        model_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_kwargs = (
            {} if model_kwargs is None else dict(model_kwargs)
        )

    def fit(
        self,
        *,
        scaling: Mapping[str, Mapping[str, float]] | None = None,
        center_offset: bool = True,
        seed: int = 42,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> BayesianISSFFit:
        try:
            import pymc as pm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyMC is required; install hsa[bayesian].") from exc

        design = self.prepare_design(
            scaling=scaling,
            center_offset=center_offset,
        )
        arrays = build_ssf_choice_arrays(
            design.data,
            id_col=self.id_col,
            predictors=design.predictors,
            stratum_col=design.stratum_col,
            offset_col=design.offset_col,
            dtype="float32",
        )
        model = build_hierarchical_issf_model(
            arrays,
            **self.model_kwargs,
        )

        sampling = {
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.95,
            "return_inferencedata": True,
            "random_seed": seed,
        }
        if sample_kwargs is not None:
            sampling.update(dict(sample_kwargs))
            sampling.setdefault("random_seed", seed)

        with model:
            idata = pm.sample(**sampling)

        fit = BayesianISSFFit(
            model=model,
            idata=idata,
            arrays=arrays,
            design=design,
            id_col=self.id_col,
        )
        self.fit_ = fit
        return fit


__all__ = [
    "ISSFDesign",
    "ISSFAnalysis",
    "FrequentistISSF",
    "FrequentistISSFFit",
    "IndividualISSFFits",
    "BayesianISSF",
    "BayesianISSFFit",
    "prepare_issf_design",
    "issf_interaction_name",
    "build_hierarchical_issf_model",
]
