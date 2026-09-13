"""Ergonomic public iSSF workflow.

This module is a thin public facade over :mod:`hsa.ssf.issf`, which retains the
statistical core and backwards-compatible low-level API. The facade separates
trajectory/choice-set state, environmental annotation, and model
specification::

    analysis -> sample availability -> annotate -> set_model -> fit

Environmental fields are passed when they are sampled rather than stored on
the analysis object. Model modifiers are classified automatically as
stratum-constant start conditions or candidate-varying directional conditions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from hsa.ssf.environment import (
    add_vector_support_covariates,
    sample_dynamic_covariates_at_points,
    sample_static_covariates_batched,
)
from hsa.ssf.frequentist import fit_conditional_ssf
from hsa.ssf.issf import (
    BayesianISSF as _LegacyBayesianISSF,
    BayesianISSFFit as _LegacyBayesianISSFFit,
    FrequentistISSF as _LegacyFrequentistISSF,
    FrequentistISSFFit,
    ISSFDesign,
    IndividualISSFFits,
    _UNSET as _LEGACY_UNSET,
    _movement_modifiers,
)


DEFAULT_MOVEMENT_TERMS = (
    "step_length_km",
    "log_step_length",
    "cos_turn_angle",
)
DEFAULT_MODIFIER_TERMS = (
    "step_length_km",
    "log_step_length",
)

_MOVEMENT_ALIASES = {
    "default": DEFAULT_MOVEMENT_TERMS,
    "step_length": DEFAULT_MODIFIER_TERMS,
    "length": DEFAULT_MODIFIER_TERMS,
    "turning": ("cos_turn_angle",),
    "turn_angle": ("cos_turn_angle",),
    "step_length_km": ("step_length_km",),
    "log_step_length": ("log_step_length",),
    "cos_turn_angle": ("cos_turn_angle",),
}


@dataclass(frozen=True)
class ISSFModelSpec:
    """Human-facing iSSF model specification."""

    selection: tuple[str, ...]
    movement_terms: tuple[str, ...]
    modifiers: dict[str, tuple[str, ...]]
    start_modifiers: tuple[str, ...]
    directional_modifiers: tuple[str, ...]
    proposal_correction: bool

    def summary(self) -> pd.Series:
        """Return a compact, readable representation of the specification."""
        return pd.Series(
            {
                "selection": self.selection,
                "movement_terms": self.movement_terms,
                "start_modifiers": self.start_modifiers,
                "directional_modifiers": self.directional_modifiers,
                "proposal_correction": self.proposal_correction,
            }
        )


def _as_names(values) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    names = tuple(dict.fromkeys(map(str, values)))
    if any(not name.strip() for name in names):
        raise ValueError("Predictor names may not be empty.")
    return names


def _normalize_mapping(values, *, default_suffix: str = "") -> dict[str, str]:
    """Normalize source names or ``source -> output`` mappings."""
    if values is None:
        return {}
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, Mapping):
        mapping = {str(source): str(output) for source, output in values.items()}
    else:
        mapping = {
            str(source): f"{source}{default_suffix}"
            for source in values
        }
    if not mapping:
        return {}
    if any(
        not source.strip() or not output.strip()
        for source, output in mapping.items()
    ):
        raise ValueError("Annotation names may not be empty.")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Annotation output names must be unique.")
    return mapping


def _resolve_movement_terms(value) -> tuple[str, ...]:
    if value is False or value is None:
        return ()
    if value is True:
        return DEFAULT_MOVEMENT_TERMS
    if isinstance(value, str):
        if value not in _MOVEMENT_ALIASES:
            raise ValueError(
                f"Unknown movement shorthand {value!r}. "
                f"Use one of {sorted(_MOVEMENT_ALIASES)} or explicit terms."
            )
        return tuple(_MOVEMENT_ALIASES[value])

    resolved: list[str] = []
    for item in value:
        item = str(item)
        terms = _MOVEMENT_ALIASES.get(item, (item,))
        for term in terms:
            if term not in resolved:
                resolved.append(term)
    return tuple(resolved)


def _resolve_modifier_terms(value) -> tuple[str, ...]:
    if value is True or value is None:
        return DEFAULT_MODIFIER_TERMS
    if value is False:
        return ()
    return _resolve_movement_terms(value)


class _ISSFFacadeMixin:
    """Shared ergonomic workflow layered over the established iSSF core."""

    model_spec_: ISSFModelSpec | None
    modifier_terms_: dict[str, tuple[str, ...]]

    def _finish_facade_init(self) -> None:
        # The legacy base couples construction to a non-empty endpoint predictor
        # list. The public facade uses a temporary internal placeholder and then
        # clears it so construction is independent of model specification.
        self.endpoint_predictors = ()
        self.start_predictors = ()
        self.directional_predictors = ()
        self.movement_terms = DEFAULT_MOVEMENT_TERMS
        self.interaction_terms = ()
        self.predictors = []
        self.model_spec_ = None
        self.modifier_terms_ = {}
        self.issf_design_ = None
        self.fit_ = None

    def _invalidate_fit(self) -> None:
        self.issf_design_ = None
        self.fit_ = None
        if hasattr(self, "validation_"):
            self.validation_ = None

    @classmethod
    def from_ssf(cls, analysis, *, copy_model: bool = True, **kwargs):
        """Reuse existing relocations and choice sets without redrawing availability."""
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
        out = cls(analysis.reloc, **options)

        has_model = (
            getattr(analysis, "model_spec_", None) is not None
            or bool(getattr(analysis, "endpoint_predictors", ()))
            or bool(getattr(analysis, "start_predictors", ()))
            or bool(getattr(analysis, "directional_predictors", ()))
        )
        if copy_model and has_model:
            out.endpoint_predictors = tuple(
                getattr(analysis, "endpoint_predictors", ())
            )
            out.start_predictors = tuple(
                getattr(analysis, "start_predictors", ())
            )
            out.directional_predictors = tuple(
                getattr(analysis, "directional_predictors", ())
            )
            out.movement_terms = tuple(
                getattr(analysis, "movement_terms", DEFAULT_MOVEMENT_TERMS)
            )
            out.interaction_terms = tuple(
                getattr(analysis, "interaction_terms", ())
            )
            out.predictors = list(out.endpoint_predictors)
            out.proposal_logpdf_col = getattr(
                analysis,
                "proposal_logpdf_col",
                "proposal_logpdf",
            )
            out.modifier_terms_ = dict(
                getattr(
                    analysis,
                    "modifier_terms_",
                    {
                        name: out.interaction_terms
                        for name in (
                            *out.start_predictors,
                            *out.directional_predictors,
                        )
                    },
                )
            )
            out.model_spec_ = getattr(analysis, "model_spec_", None)
            if out.model_spec_ is None:
                out.model_spec_ = ISSFModelSpec(
                    selection=out.endpoint_predictors,
                    movement_terms=out.movement_terms,
                    modifiers=dict(out.modifier_terms_),
                    start_modifiers=out.start_predictors,
                    directional_modifiers=out.directional_predictors,
                    proposal_correction=out.proposal_logpdf_col is not None,
                )
        return out

    def sample(
        self,
        *,
        n_available: int | None = None,
        seed: int = 42,
        domain=None,
        observed_outside: str = "raise",
        step_cutoff: float = float("inf"),
        max_domain_redraws: int = 20,
        force: bool = False,
    ):
        """Generate movement-informed choice sets.

        ``domain`` is an optional static raster used only to constrain candidate
        availability. It is not retained on the analysis object.
        """
        if n_available is not None:
            if int(n_available) <= 0:
                raise ValueError("n_available must be positive.")
            self.n_available = int(n_available)

        previous_env = self.env
        try:
            self.env = domain
            super().prepare_choice_sets(
                seed=seed,
                step_cutoff=step_cutoff,
                restrict_to_env=domain is not None,
                observed_outside=observed_outside,
                max_domain_redraws=max_domain_redraws,
                force=force,
            )
        finally:
            self.env = previous_env

        self._invalidate_fit()
        return self

    def annotate_static(
        self,
        field,
        *,
        endpoint=None,
        start=None,
        batch_size: int = 100_000,
        require_inside: bool = True,
        outside: str = "nan",
    ):
        """Sample static endpoint and/or start covariates in one call."""
        endpoint_map = _normalize_mapping(endpoint)
        start_map = _normalize_mapping(start, default_suffix="_start")
        if not endpoint_map and not start_map:
            raise ValueError("Specify endpoint= and/or start= static variables.")

        if endpoint_map:
            sampled = sample_static_covariates_batched(
                self.choices,
                field,
                bands=list(endpoint_map),
                batch_size=batch_size,
                require_inside=require_inside,
            )
            out = self.choices.copy()
            for source, output in endpoint_map.items():
                out[output] = sampled[source].to_numpy()
            self.choices_ = gpd.GeoDataFrame(
                out,
                geometry="geometry",
                crs=self.choices.crs,
            )

        if start_map:
            previous_env = self.env
            try:
                self.env = field
                super().annotate_start_static(
                    bands=start_map,
                    batch_size=batch_size,
                    outside=outside,
                )
            finally:
                self.env = previous_env

        self._invalidate_fit()
        return self

    def annotate_dynamic(
        self,
        field,
        *,
        endpoint=None,
        start=None,
        endpoint_time_col: str = "end_time",
        start_time_col: str = "start_time",
        method: str = "linear",
        **kwargs,
    ):
        """Sample dynamic endpoint and/or start covariates in one call."""
        endpoint_map = _normalize_mapping(endpoint)
        start_map = _normalize_mapping(start, default_suffix="_start")
        if not endpoint_map and not start_map:
            raise ValueError("Specify endpoint= and/or start= dynamic variables.")

        if endpoint_map:
            self.choices_ = sample_dynamic_covariates_at_points(
                self.choices,
                field,
                variables=endpoint_map,
                time_col=endpoint_time_col,
                method=method,
                **kwargs,
            )

        if start_map:
            starts = self._unique_starts()
            sampled = sample_dynamic_covariates_at_points(
                starts,
                field,
                variables=start_map,
                time_col=start_time_col,
                method=method,
                **kwargs,
            )
            self._merge_start_values(sampled, list(start_map.values()))
            self.start_annotation_diagnostics_["dynamic"] = {
                "n_starts": int(len(sampled)),
                "variables": tuple(start_map.values()),
                "time_col": start_time_col,
            }

        self._invalidate_fit()
        return self

    def annotate_vector(
        self,
        field,
        *,
        u: str,
        v: str,
        prefix: str = "wind",
        at: Sequence[str] | str = ("endpoint", "start"),
        support: bool = True,
        endpoint_time_col: str = "end_time",
        start_time_col: str = "start_time",
        method: str = "linear",
        **kwargs,
    ):
        """Sample an east/north vector field and optionally derive support.

        For ``prefix='wind'`` this produces ``wind_support`` at endpoints and
        ``wind_start_support`` from the single start wind vector projected onto
        each candidate bearing.
        """
        locations = _as_names(at)
        if not locations:
            raise ValueError("at must contain 'endpoint' and/or 'start'.")
        unknown = sorted(set(locations).difference({"endpoint", "start"}))
        if unknown:
            raise ValueError(f"at contains unsupported locations: {unknown}")

        if "endpoint" in locations:
            u_col = f"{prefix}_u"
            v_col = f"{prefix}_v"
            speed_col = f"{prefix}_speed"
            self.choices_ = sample_dynamic_covariates_at_points(
                self.choices,
                field,
                variables={u: u_col, v: v_col},
                time_col=endpoint_time_col,
                method=method,
                **kwargs,
            )
            self.choices_[speed_col] = np.hypot(
                self.choices_[u_col].to_numpy(dtype=float),
                self.choices_[v_col].to_numpy(dtype=float),
            ).astype("float32")
            if support:
                self.choices_ = add_vector_support_covariates(
                    self.choices_,
                    u_col=u_col,
                    v_col=v_col,
                    speed_col=speed_col,
                    prefix=prefix,
                )

        if "start" in locations:
            u_col = f"{prefix}_u_start"
            v_col = f"{prefix}_v_start"
            speed_col = f"{prefix}_speed_start"
            starts = self._unique_starts()
            sampled = sample_dynamic_covariates_at_points(
                starts,
                field,
                variables={u: u_col, v: v_col},
                time_col=start_time_col,
                method=method,
                **kwargs,
            )
            sampled[speed_col] = np.hypot(
                sampled[u_col].to_numpy(dtype=float),
                sampled[v_col].to_numpy(dtype=float),
            ).astype("float32")
            self._merge_start_values(sampled, [u_col, v_col, speed_col])
            if support:
                self.choices_ = add_vector_support_covariates(
                    self.choices,
                    u_col=u_col,
                    v_col=v_col,
                    speed_col=speed_col,
                    prefix=f"{prefix}_start",
                )

        self._invalidate_fit()
        return self

    def _classify_modifier(self, predictor: str) -> str:
        if predictor not in self.choices:
            raise KeyError(
                f"{predictor!r} is not present in the choice table. "
                "Annotate or derive it before set_model()."
            )
        variation = self.choices.groupby(
            [self.id_col, "stratum_id"],
            sort=False,
        )[predictor].nunique(dropna=False)
        return "start" if variation.le(1).all() else "directional"

    def set_model(
        self,
        *,
        selection=(),
        movement="default",
        modifiers=None,
        proposal_correction: bool = True,
    ):
        """Declare the ecological iSSF model.

        ``selection`` contains ordinary candidate-level effects. ``movement``
        may be ``'default'`` (``L + log(L) + cos(theta)``), ``False``, or an
        explicit sequence. ``modifiers`` may be a sequence, or a mapping from
        an environmental condition to the movement terms it modifies. Modifier
        type is inferred from within-stratum variation.
        """
        selection_names = _as_names(selection)
        movement_terms = _resolve_movement_terms(movement)

        if modifiers is None:
            modifier_map: dict[str, tuple[str, ...]] = {}
        elif isinstance(modifiers, Mapping):
            modifier_map = {
                str(name): _resolve_modifier_terms(terms)
                for name, terms in modifiers.items()
            }
        else:
            modifier_map = {
                name: DEFAULT_MODIFIER_TERMS
                for name in _as_names(modifiers)
            }

        if modifier_map and not movement_terms:
            raise ValueError("Movement modifiers require movement terms.")
        for name, terms in modifier_map.items():
            unknown = sorted(set(terms).difference(movement_terms))
            if unknown:
                raise ValueError(
                    f"Modifier {name!r} requests movement terms not present "
                    f"in the model: {unknown}"
                )

        start: list[str] = []
        directional: list[str] = []
        for name in modifier_map:
            kind = self._classify_modifier(name)
            (start if kind == "start" else directional).append(name)

        # Directional modifiers already receive a candidate-level main effect.
        selection_names = tuple(
            name for name in selection_names if name not in directional
        )

        for name in selection_names:
            if name not in self.choices:
                raise KeyError(
                    f"{name!r} is not present in the choice table. "
                    "Annotate it before set_model()."
                )
            variation = self.choices.groupby(
                [self.id_col, "stratum_id"],
                sort=False,
            )[name].nunique(dropna=False)
            if variation.le(1).all():
                raise ValueError(
                    f"{name!r} is constant within every choice stratum and "
                    "cannot have an identifiable conditional-choice main effect. "
                    "Use it as a movement modifier instead."
                )

        if not selection_names and not directional and not movement_terms:
            raise ValueError("The iSSF model contains no estimable terms.")

        interaction_terms: list[str] = []
        for terms in modifier_map.values():
            for term in terms:
                if term not in interaction_terms:
                    interaction_terms.append(term)

        self.endpoint_predictors = tuple(selection_names)
        self.start_predictors = tuple(start)
        self.directional_predictors = tuple(directional)
        self.movement_terms = tuple(movement_terms)
        self.interaction_terms = tuple(interaction_terms)
        self.modifier_terms_ = dict(modifier_map)
        self.predictors = list(self.endpoint_predictors)
        self.proposal_logpdf_col = (
            "proposal_logpdf" if proposal_correction else None
        )
        self.model_spec_ = ISSFModelSpec(
            selection=self.endpoint_predictors,
            movement_terms=self.movement_terms,
            modifiers=dict(self.modifier_terms_),
            start_modifiers=self.start_predictors,
            directional_modifiers=self.directional_predictors,
            proposal_correction=bool(proposal_correction),
        )
        self._invalidate_fit()
        return self

    def set_predictors(
        self,
        *,
        endpoint_predictors=_LEGACY_UNSET,
        start_predictors=_LEGACY_UNSET,
        directional_predictors=_LEGACY_UNSET,
        movement_terms=_LEGACY_UNSET,
        interaction_terms=_LEGACY_UNSET,
    ):
        """Backwards-compatible wrapper for the pre-refactor model API."""
        warnings.warn(
            "set_predictors() is retained for compatibility; prefer set_model().",
            DeprecationWarning,
            stacklevel=2,
        )

        def resolve(value, current):
            if value is _LEGACY_UNSET:
                return tuple(current)
            if value is None:
                return ()
            return _as_names(value)

        endpoint = resolve(endpoint_predictors, self.endpoint_predictors)
        start = resolve(start_predictors, self.start_predictors)
        directional = resolve(
            directional_predictors,
            self.directional_predictors,
        )
        movement = resolve(movement_terms, self.movement_terms)
        interactions = resolve(interaction_terms, self.interaction_terms)
        unknown = sorted(set(interactions).difference(movement))
        if unknown:
            raise ValueError(
                "interaction_terms must also be present in movement_terms: "
                f"{unknown}"
            )
        if interactions and not (start or directional):
            raise ValueError(
                "interaction_terms require start_predictors and/or "
                "directional_predictors."
            )

        self.endpoint_predictors = endpoint
        self.start_predictors = start
        self.directional_predictors = directional
        self.movement_terms = movement
        self.interaction_terms = interactions
        self.modifier_terms_ = {
            name: interactions
            for name in (*start, *directional)
        }
        self.predictors = list(endpoint)
        self.model_spec_ = ISSFModelSpec(
            selection=endpoint,
            movement_terms=movement,
            modifiers=dict(self.modifier_terms_),
            start_modifiers=start,
            directional_modifiers=directional,
            proposal_correction=self.proposal_logpdf_col is not None,
        )
        self._invalidate_fit()
        return self

    def prepare_design(
        self,
        *,
        scaling: Mapping[str, Mapping[str, float]] | None = None,
        center_offset: bool = True,
    ) -> ISSFDesign:
        """Prepare a model-ready design, respecting per-modifier term choices."""
        if self.model_spec_ is None:
            raise RuntimeError(
                "No iSSF model has been specified. Call set_model() before fit()."
            )
        design = super().prepare_design(
            scaling=scaling,
            center_offset=center_offset,
        )

        allowed_interactions = {
            (modifier, term)
            for modifier, terms in self.modifier_terms_.items()
            for term in terms
        }
        interactions = {
            key: column
            for key, column in design.interaction_columns.items()
            if key in allowed_interactions
        }
        predictors = (
            *(f"{name}_z" for name in design.endpoint_predictors),
            *(f"{name}_z" for name in design.directional_predictors),
            *design.movement_terms,
            *interactions.values(),
        )
        design = replace(
            design,
            predictors=tuple(predictors),
            interaction_columns=interactions,
        )
        self.issf_design_ = design
        return design

    def subset(
        self,
        *,
        n_strata_per_id: int | None = None,
        fraction: float | None = None,
        individuals=None,
        seed: int = 42,
    ):
        """Return an independent analysis containing complete sampled strata."""
        if (n_strata_per_id is None) == (fraction is None):
            raise ValueError("Specify exactly one of n_strata_per_id or fraction.")
        if n_strata_per_id is not None and n_strata_per_id <= 0:
            raise ValueError("n_strata_per_id must be positive.")
        if fraction is not None and not 0 < fraction <= 1:
            raise ValueError("fraction must lie in (0, 1].")

        choices = self.choices
        available_ids = set(choices[self.id_col].unique())
        selected_ids = (
            list(choices[self.id_col].drop_duplicates())
            if individuals is None
            else list(individuals)
        )
        unknown = sorted(set(selected_ids).difference(available_ids))
        if unknown:
            raise KeyError(f"Unknown individuals: {unknown}")

        choices = choices.loc[choices[self.id_col].isin(selected_ids)].copy()
        strata = choices[[self.id_col, "stratum_id"]].drop_duplicates()
        rng = np.random.default_rng(seed)
        selected = []
        for _, group in strata.groupby(self.id_col, sort=False):
            if n_strata_per_id is not None:
                n_keep = min(int(n_strata_per_id), len(group))
            else:
                n_keep = max(1, int(np.ceil(len(group) * float(fraction))))
            idx = rng.choice(group.index.to_numpy(), size=n_keep, replace=False)
            selected.append(group.loc[idx])

        selected_strata = pd.concat(selected, ignore_index=True)
        keys = pd.MultiIndex.from_frame(choices[[self.id_col, "stratum_id"]])
        selected_keys = pd.MultiIndex.from_frame(
            selected_strata[[self.id_col, "stratum_id"]]
        )
        subset_choices = choices.loc[keys.isin(selected_keys)].copy()
        check = subset_choices.groupby(
            [self.id_col, "stratum_id"],
            sort=False,
        ).agg(
            n_choices=("candidate_id", "size"),
            n_used=("used", "sum"),
        )
        expected_n_choices = self.n_available + 1
        if not check["n_choices"].eq(expected_n_choices).all():
            raise RuntimeError("Subsetting produced incomplete iSSF strata.")
        if not check["n_used"].eq(1).all():
            raise RuntimeError(
                "Subsetting produced strata without exactly one observed choice."
            )

        subset_reloc = self.reloc.loc[
            self.reloc[self.id_col].isin(selected_ids)
        ].copy()
        kwargs = {
            "id_col": self.id_col,
            "timestamp_col": self.timestamp_col,
            "expected_interval_min": self.expected_interval_min,
            "tolerance_min": self.tolerance_min,
            "round_freq": self.round_freq,
            "n_available": self.n_available,
            "speed_margin": self.speed_margin,
            "burst_gap": self.burst_gap,
            "choices": subset_choices,
            "proposal_logpdf_col": self.proposal_logpdf_col,
        }
        if hasattr(self, "model_kwargs"):
            kwargs["model_kwargs"] = dict(self.model_kwargs)

        out = self.__class__(subset_reloc, **kwargs)
        out.endpoint_predictors = tuple(self.endpoint_predictors)
        out.start_predictors = tuple(self.start_predictors)
        out.directional_predictors = tuple(self.directional_predictors)
        out.movement_terms = tuple(self.movement_terms)
        out.interaction_terms = tuple(self.interaction_terms)
        out.modifier_terms_ = dict(self.modifier_terms_)
        out.model_spec_ = self.model_spec_
        out.predictors = list(self.predictors)
        out.choice_set_diagnostics_ = dict(self.choice_set_diagnostics_)
        out.start_annotation_diagnostics_ = dict(
            self.start_annotation_diagnostics_
        )
        out.subset_diagnostics_ = {
            "source_n_individuals": int(self.choices[self.id_col].nunique()),
            "source_n_strata": int(
                self.choices[[self.id_col, "stratum_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "n_individuals": int(subset_choices[self.id_col].nunique()),
            "n_strata": int(len(check)),
            "n_choice_rows": int(len(subset_choices)),
            "n_strata_per_id": n_strata_per_id,
            "fraction": fraction,
            "seed": seed,
        }
        return out

    def to_bayesian(self, *, model_kwargs: Mapping[str, Any] | None = None):
        """Create a Bayesian analysis reusing this object's annotated choices."""
        out = BayesianISSF(
            self.reloc,
            id_col=self.id_col,
            timestamp_col=self.timestamp_col,
            expected_interval_min=self.expected_interval_min,
            tolerance_min=self.tolerance_min,
            round_freq=self.round_freq,
            n_available=self.n_available,
            speed_margin=self.speed_margin,
            burst_gap=self.burst_gap,
            choices=self.choices,
            proposal_logpdf_col=self.proposal_logpdf_col,
            model_kwargs=model_kwargs,
        )
        out.endpoint_predictors = tuple(self.endpoint_predictors)
        out.start_predictors = tuple(self.start_predictors)
        out.directional_predictors = tuple(self.directional_predictors)
        out.movement_terms = tuple(self.movement_terms)
        out.interaction_terms = tuple(self.interaction_terms)
        out.modifier_terms_ = dict(self.modifier_terms_)
        out.model_spec_ = self.model_spec_
        out.predictors = list(self.predictors)
        return out


class FrequentistISSF(_ISSFFacadeMixin, _LegacyFrequentistISSF):
    """Pooled frequentist iSSF with the simplified public workflow."""

    def __init__(
        self,
        reloc: gpd.GeoDataFrame,
        *,
        id_col: str = "individual-local-identifier",
        timestamp_col: str = "Timestamp",
        expected_interval_min: float | None = 60,
        tolerance_min: float = 10,
        round_freq: str | None = "h",
        n_available: int = 20,
        speed_margin: float = 1.05,
        burst_gap: str = "2h",
        choices: gpd.GeoDataFrame | None = None,
        proposal_logpdf_col: str | None = "proposal_logpdf",
    ):
        _LegacyFrequentistISSF.__init__(
            self,
            reloc,
            None,
            endpoint_predictors=("__unconfigured__",),
            start_predictors=(),
            directional_predictors=(),
            movement_terms=DEFAULT_MOVEMENT_TERMS,
            interaction_terms=(),
            proposal_logpdf_col=proposal_logpdf_col,
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
        self._finish_facade_init()
        self.proposal_logpdf_col = proposal_logpdf_col

    def fit_individuals(
        self,
        *,
        scaling: Mapping[str, Mapping[str, float]] | None = None,
        center_offset: bool = True,
        engine: str = "fast",
        method: str = "lbfgs",
        maxiter: int = 1000,
    ) -> IndividualISSFFits:
        """Fit independent individual iSSFs on the same prepared scale."""
        design = self.prepare_design(
            scaling=scaling,
            center_offset=center_offset,
        )
        fits: dict[Any, FrequentistISSFFit] = {}
        frames = []

        for individual, group in design.data.groupby(self.id_col, sort=False):
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
            individual_design = replace(
                design,
                data=group.copy(),
                diagnostics={
                    **design.diagnostics,
                    "n_individuals": 1,
                    "n_strata_retained": int(
                        group[design.stratum_col].nunique()
                    ),
                    "n_choice_rows": int(len(group)),
                },
            )
            fit = FrequentistISSFFit(
                model=model,
                result=result,
                design=individual_design,
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


class BayesianISSFFit(_LegacyBayesianISSFFit):
    """Bayesian iSSF fit with directional modifiers honoured in movement plots."""

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
        modifiers = _movement_modifiers(self.design)
        if predictor not in modifiers:
            raise KeyError(
                f"{predictor!r} is not a movement-modifying predictor "
                "in this iSSF design."
            )
        if moderator is not None and moderator not in modifiers:
            raise KeyError(
                f"{moderator!r} is not a movement-modifying predictor "
                "in this iSSF design."
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

                gamma_length = (
                    draws["step_length_km"].to_numpy(dtype=float).copy()
                )
                gamma_log = (
                    draws["log_step_length"].to_numpy(dtype=float).copy()
                )
                for modifier in modifiers:
                    z = float(conditions.get(modifier, 0.0))
                    length_col = self.design.interaction_columns.get(
                        (modifier, "step_length_km")
                    )
                    log_col = self.design.interaction_columns.get(
                        (modifier, "log_step_length")
                    )
                    if length_col is not None:
                        gamma_length += (
                            draws[length_col].to_numpy(dtype=float) * z
                        )
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
                "Population"
                if moderator is None and individual is None
                else (
                    str(individual)
                    if moderator is None
                    else f"{moderator} = {level:+g} SD"
                )
            )
            line, = ax.plot(grid_values, median, linewidth=2, label=label)
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


class BayesianISSF(_ISSFFacadeMixin, _LegacyBayesianISSF):
    """Hierarchical Bayesian iSSF with the simplified public workflow."""

    def __init__(
        self,
        reloc: gpd.GeoDataFrame,
        *,
        id_col: str = "individual-local-identifier",
        timestamp_col: str = "Timestamp",
        expected_interval_min: float | None = 60,
        tolerance_min: float = 10,
        round_freq: str | None = "h",
        n_available: int = 20,
        speed_margin: float = 1.05,
        burst_gap: str = "2h",
        choices: gpd.GeoDataFrame | None = None,
        proposal_logpdf_col: str | None = "proposal_logpdf",
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        _LegacyBayesianISSF.__init__(
            self,
            reloc,
            None,
            endpoint_predictors=("__unconfigured__",),
            start_predictors=(),
            directional_predictors=(),
            movement_terms=DEFAULT_MOVEMENT_TERMS,
            interaction_terms=(),
            proposal_logpdf_col=proposal_logpdf_col,
            id_col=id_col,
            timestamp_col=timestamp_col,
            expected_interval_min=expected_interval_min,
            tolerance_min=tolerance_min,
            round_freq=round_freq,
            n_available=n_available,
            speed_margin=speed_margin,
            burst_gap=burst_gap,
            choices=choices,
            model_kwargs=model_kwargs,
        )
        self._finish_facade_init()
        self.proposal_logpdf_col = proposal_logpdf_col

    def fit(self, *args, **kwargs) -> BayesianISSFFit:
        legacy = super().fit(*args, **kwargs)
        fit = BayesianISSFFit(
            model=legacy.model,
            idata=legacy.idata,
            arrays=legacy.arrays,
            design=legacy.design,
            id_col=legacy.id_col,
        )
        self.fit_ = fit
        return fit

    def attach_fit(self, idata, *, validate: bool = True) -> BayesianISSFFit:
        """Attach externally sampled InferenceData to the current analysis."""
        legacy = _LegacyBayesianISSFFit.attach_fit(
            self,
            idata,
            validate=validate,
        )
        fit = BayesianISSFFit(
            model=legacy.model,
            idata=legacy.idata,
            arrays=legacy.arrays,
            design=legacy.design,
            id_col=legacy.id_col,
        )
        self.fit_ = fit
        return fit

    def load_fit(self, path, *, validate: bool = True) -> BayesianISSFFit:
        """Load a NetCDF InferenceData file and attach it to this analysis."""
        import arviz as az

        return self.attach_fit(az.from_netcdf(path), validate=validate)

    def export_fit(
        self,
        path,
        *,
        overwrite: bool = False,
        scaling=None,
        center_offset: bool = True,
    ):
        """Export the frozen numeric design and Bayesian model metadata."""
        exported = _LegacyBayesianISSFFit.export_fit(
            self,
            path,
            overwrite=overwrite,
            scaling=scaling,
            center_offset=center_offset,
        )
        config_path = Path(exported) / "fit_config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        design = self.issf_design_
        if design is not None:
            config["directional_predictors"] = list(
                design.directional_predictors
            )
        config["modifier_terms"] = {
            name: list(terms)
            for name, terms in self.modifier_terms_.items()
        }
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        return exported


__all__ = [
    "ISSFModelSpec",
    "FrequentistISSF",
    "FrequentistISSFFit",
    "IndividualISSFFits",
    "BayesianISSF",
    "BayesianISSFFit",
]
