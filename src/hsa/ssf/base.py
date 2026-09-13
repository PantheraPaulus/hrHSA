"""Object-oriented orchestration for step-selection analyses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import geopandas as gpd
import pandas as pd

from hsa._time import require_timezone_aware

from hsa.movement import fit_movement_kernel_per_id
from hsa.ssf.choice_sets import build_movement_choice_sets
from hsa.ssf.domain import redraw_available_steps_inside_raster
from hsa.ssf.environment import (
    add_movement_terms,
    add_vector_support_covariates,
    annotate_static_covariates,
    sample_dynamic_covariates_at_points,
    sample_dynamic_vectors_at_points,
)


class SSFFit(ABC):
    """Common interface implemented by fitted frequentist and Bayesian SSFs."""

    @abstractmethod
    def summary(self, *args, **kwargs):
        """Return estimator-specific model summary information."""

    @abstractmethod
    def coefficients(self, *args, **kwargs) -> pd.DataFrame:
        """Return tidy coefficient estimates."""

    @abstractmethod
    def choice_scores(self, *args, **kwargs):
        """Return conditional-choice predictive scores."""


class SSFAnalysis(ABC):
    """Base class for stateful movement-informed SSF workflows.

    Unlike ``RSFAnalysis``, an SSF owns a trajectory-derived choice table. All
    alternatives in one stratum share the same step origin; availability is
    generated from fitted movement kernels rather than a spatial background.
    """

    def __init__(
        self,
        reloc: gpd.GeoDataFrame,
        env=None,
        *,
        predictors: Sequence[str],
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
        predictors = list(dict.fromkeys(predictors))
        if not predictors:
            raise ValueError("At least one predictor is required.")
        if n_available <= 0:
            raise ValueError("n_available must be positive.")
        if speed_margin < 1:
            raise ValueError("speed_margin must be at least 1.")

        self.id_col = id_col
        self.timestamp_col = timestamp_col
        self.predictors = predictors
        self.env = env
        self.expected_interval_min = expected_interval_min
        self.tolerance_min = float(tolerance_min)
        self.round_freq = round_freq
        self.n_available = int(n_available)
        self.speed_margin = float(speed_margin)
        self.burst_gap = burst_gap

        self.reloc = self._prepare_relocations(reloc)
        self.movement_: dict[str, Any] | None = None
        self.choices_: gpd.GeoDataFrame | None = None
        self.choice_set_diagnostics_: dict[str, Any] = {}
        if choices is not None:
            self.set_choices(choices)
        self.fit_: SSFFit | None = None
        self.validation_ = None

    def _prepare_relocations(self, reloc: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if not isinstance(reloc, gpd.GeoDataFrame):
            raise TypeError("reloc must be a geopandas.GeoDataFrame.")
        for col in (self.id_col, self.timestamp_col, "geometry"):
            if col not in reloc:
                raise KeyError(f"{col!r} not found in reloc.")
        if reloc.crs is None:
            raise ValueError("reloc.crs is None; set a projected CRS before SSF analysis.")
        g = reloc.copy()
        g[self.timestamp_col] = require_timezone_aware(
            g[self.timestamp_col],
            name=self.timestamp_col,
        )
        g = g.dropna(subset=[self.id_col, self.timestamp_col, "geometry"]).copy()
        if g.empty:
            raise ValueError("No complete relocation records remain.")
        return g.sort_values([self.id_col, self.timestamp_col]).reset_index(drop=True)

    @property
    def individuals(self) -> list:
        return pd.Index(self.reloc[self.id_col].dropna().unique()).tolist()

    @property
    def choices(self) -> gpd.GeoDataFrame:
        if self.choices_ is None:
            raise RuntimeError(
                "SSF choice sets have not been prepared. Call prepare_choice_sets() "
                "or set_choices() first."
            )
        return self.choices_

    def set_choices(self, choices: gpd.GeoDataFrame) -> "SSFAnalysis":
        """Attach a precomputed canonical SSF choice table."""
        if not isinstance(choices, gpd.GeoDataFrame):
            raise TypeError("choices must be a geopandas.GeoDataFrame.")
        required = {
            self.id_col,
            "stratum_id",
            "candidate_id",
            "used",
            "start_time",
            "end_time",
            "start_geometry",
            "geometry",
        }
        missing = sorted(required.difference(choices.columns))
        if missing:
            raise KeyError(f"Missing canonical SSF columns: {missing}")
        check = choices.groupby("stratum_id", sort=False)["used"].agg(
            n_choices="size", n_used="sum"
        )
        if not check["n_used"].eq(1).all():
            raise ValueError("Every SSF stratum must contain exactly one used alternative.")
        self.choices_ = choices.copy()
        return self

    def prepare_choice_sets(
        self,
        *,
        seed: int = 42,
        step_cutoff: float = float("inf"),
        restrict_to_env: bool = True,
        observed_outside: str = "raise",
        max_domain_redraws: int = 20,
        force: bool = False,
    ) -> "SSFAnalysis":
        """Fit movement kernels and generate movement-informed SSF alternatives.

        When an environmental raster is stored and ``restrict_to_env=True``,
        available alternatives outside raster coverage are rejection-sampled
        individually from the same movement proposal until covered. Already-valid
        alternatives are retained and observed endpoints are never moved.

        ``observed_outside='raise'`` is the conservative default. If the raster
        cannot reasonably be expanded, ``observed_outside='exclude'`` removes
        the entire unsupported observed stratum (used endpoint and all of its
        alternatives) and records the exclusion in ``choice_set_diagnostics_``.
        """
        if self.choices_ is not None and not force:
            raise RuntimeError("Choice sets already exist. Pass force=True to regenerate them.")
        if observed_outside not in {"raise", "exclude"}:
            raise ValueError("observed_outside must be 'raise' or 'exclude'.")

        movement = fit_movement_kernel_per_id(
            self.reloc,
            id_col=self.id_col,
            timestamp_col=self.timestamp_col,
            round_freq=self.round_freq,
            drop_duplicate_fixes=True,
            expected_interval_min=self.expected_interval_min,
            tolerance_min=self.tolerance_min,
            step_cutoff=step_cutoff,
        )
        choices = build_movement_choice_sets(
            movement,
            id_col=self.id_col,
            n_available=self.n_available,
            speed_margin=self.speed_margin,
            burst_gap=self.burst_gap,
            seed=seed,
        )

        domain_diagnostics: dict[str, Any] = {
            "observed_outside_policy": observed_outside,
            "n_outside_initial": 0,
            "n_strata_outside_initial": 0,
            "n_available_outside_initial": 0,
            "n_observed_outside_initial": 0,
            "n_observed_strata_excluded": 0,
            "observed_strata_excluded_by_id": {},
            "excluded_stratum_ids": [],
            "n_strata_redrawn_initial": 0,
            "n_available_replaced_total": 0,
            "redraw_rounds": 0,
            "n_outside_final": 0,
        }
        if restrict_to_env and self.env is not None:
            choices, domain_diagnostics = redraw_available_steps_inside_raster(
                choices,
                movement,
                self.env,
                id_col=self.id_col,
                n_available=self.n_available,
                speed_margin=self.speed_margin,
                seed=seed,
                max_rounds=max_domain_redraws,
                observed_outside=observed_outside,
            )

        self.movement_ = movement
        self.choices_ = choices
        self.choice_set_diagnostics_ = {
            "restrict_to_env": bool(restrict_to_env and self.env is not None),
            **domain_diagnostics,
        }
        return self

    def annotate_static(
        self,
        *,
        bands: Sequence[str],
        batch_size: int = 100_000,
        require_inside: bool = True,
    ) -> "SSFAnalysis":
        """Attach static raster values at every candidate endpoint."""
        if self.env is None:
            raise ValueError("No environmental raster is stored on this analysis.")
        self.choices_ = annotate_static_covariates(
            self.choices,
            self.env,
            bands=bands,
            batch_size=batch_size,
            require_inside=require_inside,
        )
        return self

    def annotate_dynamic(
        self,
        field,
        *,
        variables,
        time_col: str = "end_time",
        **kwargs,
    ) -> "SSFAnalysis":
        """Sample arbitrary time-varying covariates at candidate endpoints.

        ``variables`` may be a name, a sequence of names, or a mapping from
        source variable names to desired choice-table column names. Coordinate
        names, interpolation, batching and optional transformations are passed
        through to :func:`sample_dynamic_covariates_at_points`.
        """
        self.choices_ = sample_dynamic_covariates_at_points(
            self.choices,
            field,
            variables=variables,
            time_col=time_col,
            **kwargs,
        )
        return self

    def annotate_dynamic_vectors(
        self,
        field,
        *,
        time_col: str = "end_time",
        **kwargs,
    ) -> "SSFAnalysis":
        """Sample a dynamic vector field independently at candidate endpoints."""
        self.choices_ = sample_dynamic_vectors_at_points(
            self.choices, field, time_col=time_col, **kwargs
        )
        return self

    def plot_dynamic_conditions(
        self,
        field,
        *,
        variables,
        times,
        domain: str | Sequence[float] = "relocations",
        timezone: str | None = "UTC",
        spatial_margin: float = 0.25,
        resolution: float | None = None,
        max_cells: int | None = None,
        transforms=None,
        derived=None,
        extract_kwargs=None,
        plot_kwargs=None,
    ):
        """Quickly compare gridded environmental conditions across time points.

        By default the dynamic field is cropped to the relocation extent before
        loading. ``domain='choices'`` instead uses all candidate endpoints, while
        an explicit ``(west, south, east, north)`` sequence supplies a custom
        geographic region. Ordinary regions crossing Greenwich are supported
        even when the field stores longitudes as 0..360.

        ``resolution`` optionally decimates the field to approximately that
        geographic spacing in degrees, and ``max_cells`` can impose an additional
        quick-look size cap. These controls use direct index strides rather than
        conservative aggregation, so they are intended for fast visualization.

        Naive requested times are interpreted in ``timezone`` and converted to
        UTC for lookup. The result contains the matplotlib figure/axes, the
        extracted xarray snapshots, and an area-weighted distribution summary.
        """
        from hsa.ssf.dynamic_diagnostics import plot_dynamic_conditions

        if isinstance(domain, str):
            if domain == "relocations":
                geographic = self.reloc.to_crs(4326)
            elif domain == "choices":
                geographic = self.choices.to_crs(4326)
            else:
                raise ValueError(
                    "domain must be 'relocations', 'choices', or explicit "
                    "(west, south, east, north) bounds."
                )
            bounds = tuple(map(float, geographic.total_bounds))
        else:
            bounds = tuple(map(float, domain))
            if len(bounds) != 4:
                raise ValueError("Explicit domain must contain four geographic bounds.")

        options = {} if extract_kwargs is None else dict(extract_kwargs)
        options.setdefault("spatial_margin", spatial_margin)
        options.setdefault("resolution", resolution)
        options.setdefault("max_cells", max_cells)
        return plot_dynamic_conditions(
            field,
            variables=variables,
            times=times,
            bounds=bounds,
            timezone=timezone,
            transforms=transforms,
            derived=derived,
            extract_kwargs=options,
            plot_kwargs=plot_kwargs,
        )

    def add_vector_support(self, **kwargs) -> "SSFAnalysis":
        """Add geodesic support/cross-vector covariates to the choice table."""
        self.choices_ = add_vector_support_covariates(self.choices, **kwargs)
        return self

    def add_movement_terms(self) -> "SSFAnalysis":
        """Add movement terms retained for future iSSF fits."""
        self.choices_ = add_movement_terms(self.choices)
        return self

    def validate_predictors(self) -> None:
        """Check predictor names before whole-stratum completeness filtering."""
        missing = [p for p in self.predictors if p not in self.choices]
        if missing:
            raise ValueError(
                "Fitted SSF predictors are missing from the choice table: "
                f"{missing}. Annotate/derive them before fit()."
            )

    def validate(self, scheme, **kwargs):
        """Run an SSF validation strategy against this analysis."""
        result = scheme.run(self, **kwargs)
        self.validation_ = result
        return result

    @abstractmethod
    def fit(self, *args, **kwargs) -> SSFFit:
        """Fit the estimator to the prepared SSF choice table."""


__all__ = ["SSFAnalysis", "SSFFit"]
