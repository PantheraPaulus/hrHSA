"""Object-oriented orchestration for resource-selection analyses.

The numerical/statistical kernels remain functional and independently testable.
The classes in this module own analysis state and coordinate those kernels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from hsa._time import require_timezone_aware
from pyproj import CRS

from hsa.rsf.cv import _domains_by_id, thin_by_time
from hsa.sampling import sample_available_points, sample_raster_stack


class RSFFit(ABC):
    """Common interface implemented by fitted frequentist and Bayesian RSFs."""

    @abstractmethod
    def summary(self, *args, **kwargs):
        """Return a model summary."""

    @abstractmethod
    def coefficients(self, *args, **kwargs) -> pd.DataFrame:
        """Return a tidy coefficient table."""

    @abstractmethod
    def predict_surface(self, env=None, *args, **kwargs):
        """Project the fitted RSF over an environmental raster."""


class RSFAnalysis(ABC):
    """Base class for stateful RSF workflows.

    Parameters
    ----------
    reloc
        Relocations with an individual identifier, timestamp and geometry.
    env
        Environmental raster stack sampled during fitting and prediction.
    predictors
        Predictor names used by the estimator.
    id_col
        Individual identifier column.
    timestamp_col
        Canonical timestamp column. Current low-level CV functions use
        ``"Timestamp"``; another source column can be supplied here and is
        normalized internally.
    domain
        Optional common availability domain. If omitted, one domain is derived
        for each individual.
    domain_quantile
        Quantile passed to the MCP availability-domain estimator.
    """

    def __init__(
        self,
        reloc: gpd.GeoDataFrame,
        env,
        *,
        predictors: Sequence[str],
        id_col: str = "Individual_ID",
        timestamp_col: str = "Timestamp",
        domain: gpd.GeoDataFrame | None = None,
        domain_quantile: float = 0.95,
    ):
        predictors = list(dict.fromkeys(predictors))
        if not predictors:
            raise ValueError("At least one predictor is required.")
        if not 0 < domain_quantile <= 1:
            raise ValueError("domain_quantile must be in (0, 1].")

        self.id_col = id_col
        self.timestamp_col = timestamp_col
        self.predictors = predictors
        self.env = env
        self.domain = domain
        self.domain_quantile = float(domain_quantile)

        self.reloc = self._prepare_relocations(reloc)
        self._validate_environment_crs()
        self._domains: dict[Any, gpd.GeoDataFrame] | None = None
        self.fit_: RSFFit | None = None
        self.validation_ = None

    def _prepare_relocations(
        self,
        reloc: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        if not isinstance(reloc, gpd.GeoDataFrame):
            raise TypeError("reloc must be a geopandas.GeoDataFrame.")
        if self.id_col not in reloc.columns:
            raise KeyError(f"{self.id_col!r} not found in reloc.")
        if self.timestamp_col not in reloc.columns:
            raise KeyError(f"{self.timestamp_col!r} not found in reloc.")
        if "geometry" not in reloc.columns:
            raise KeyError("'geometry' not found in reloc.")
        if reloc.crs is None:
            raise ValueError("reloc.crs is None; set a CRS before RSF analysis.")

        g = reloc.copy()
        if self.timestamp_col != "Timestamp":
            if "Timestamp" in g.columns:
                raise ValueError(
                    "reloc already contains 'Timestamp' while timestamp_col "
                    f"is {self.timestamp_col!r}; normalize the columns first."
                )
            g = g.rename(columns={self.timestamp_col: "Timestamp"})
            self.timestamp_col = "Timestamp"

        g["Timestamp"] = require_timezone_aware(
            g["Timestamp"],
            name="Timestamp",
        )
        g = g.dropna(
            subset=[self.id_col, "Timestamp", "geometry"],
        ).copy()
        if g.empty:
            raise ValueError("No complete relocation records remain.")
        return g

    def _validate_environment_crs(self) -> None:
        """Fail early when relocations and a georeferenced raster disagree."""
        try:
            env_crs = self.env.rio.crs
        except Exception:
            return

        if env_crs is None:
            return

        reloc_crs = CRS.from_user_input(self.reloc.crs)
        raster_crs = CRS.from_user_input(env_crs)
        if reloc_crs != raster_crs:
            raise ValueError(
                "CRS mismatch between relocations and environmental raster: "
                f"{reloc_crs.to_string()} != {raster_crs.to_string()}. "
                "Reproject relocations or the raster before constructing the analysis."
            )

    @property
    def individuals(self) -> list:
        """Individuals represented in the analysis."""
        return pd.Index(self.reloc[self.id_col].dropna().unique()).tolist()

    @property
    def domains(self) -> dict[Any, gpd.GeoDataFrame]:
        """Lazily construct and cache one availability domain per individual."""
        if self._domains is None:
            self._domains = _domains_by_id(
                self.reloc,
                id_col=self.id_col,
                domain=self.domain,
                quantile=self.domain_quantile,
            )
        return self._domains

    def prepare_domains(self) -> "RSFAnalysis":
        """Materialize availability domains and return ``self``."""
        _ = self.domains
        return self

    def prepare(
        self,
        path,
        *,
        sampling_factor: int = 10,
        thin_dt: str | None = None,
        seed: int = 42,
        execution=None,
        client=None,
        engine: str = "auto",
        overwrite: bool = False,
        dropna: bool = True,
    ):
        """Materialise reusable per-individual RSF samples on disk.

        Prepared datasets separate the expensive geospatial stage from repeated
        statistical fitting. They are especially useful for leave-one-individual-
        out validation because each animal's environmental values are extracted
        once and then reused across folds.
        """
        from hsa.compute.prepared import prepare_rsf_dataset

        return prepare_rsf_dataset(
            self,
            path,
            sampling_factor=sampling_factor,
            thin_dt=thin_dt,
            seed=seed,
            execution=execution,
            client=client,
            engine=engine,
            overwrite=overwrite,
            dropna=dropna,
        )

    def sample_training_data(
        self,
        *,
        sampling_factor: int = 10,
        bands: Sequence[str] | None = None,
        thin_dt: str | None = None,
        id_cols: str | Sequence[str] | None = None,
        seed: int = 42,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """Sample used/available training data inside each animal's domain.

        By default only ``self.predictors`` are sampled from the environmental
        stack. ``bands`` can be supplied explicitly for advanced workflows, but
        every fitted predictor must still be present in the sampled result.
        """
        if sampling_factor <= 0:
            raise ValueError("sampling_factor must be positive.")

        sample_bands = (
            list(self.predictors)
            if bands is None
            else list(dict.fromkeys(bands))
        )
        if not sample_bands:
            raise ValueError("bands must contain at least one raster band.")

        if "band" not in self.env.dims:
            raise ValueError("env must contain a 'band' dimension.")

        available_bands = {str(value) for value in self.env["band"].values}
        missing_bands = [
            band
            for band in sample_bands
            if band not in available_bands
        ]
        if missing_bands:
            raise ValueError(
                "Requested raster bands are missing from env: "
                f"{missing_bands}"
            )

        missing_predictor_bands = [
            predictor
            for predictor in self.predictors
            if predictor not in sample_bands
        ]
        if missing_predictor_bands:
            raise ValueError(
                "bands must include all fitted predictors; missing: "
                f"{missing_predictor_bands}"
            )

        sampled_parts: list[gpd.GeoDataFrame] = []
        grouped = self.reloc.groupby(self.id_col, sort=False)

        for j, (individual_id, used_i) in enumerate(grouped):
            used_i = used_i.copy()
            if thin_dt is not None:
                used_i = thin_by_time(used_i, min_dt=thin_dt)
            if used_i.empty:
                continue

            sampled_i = sample_available_points(
                self.domains[individual_id],
                len(used_i) * sampling_factor,
                used=used_i,
                seed=seed + j,
                timestamp_col="Timestamp",
            )
            sampled_i[self.id_col] = individual_id
            sampled_parts.append(sampled_i)

        if not sampled_parts:
            raise ValueError("No training samples were generated.")

        samples = gpd.GeoDataFrame(
            pd.concat(sampled_parts, ignore_index=True),
            geometry="geometry",
            crs=self.reloc.crs,
        )

        sampled_result = sample_raster_stack(
            samples,
            self.env,
            bands=sample_bands,
            id_cols=id_cols,
        )
        sampled = (
            sampled_result[0]
            if isinstance(sampled_result, tuple)
            else sampled_result
        )
        sampled = sampled.replace([np.inf, -np.inf], np.nan)

        missing_predictors = [
            predictor
            for predictor in self.predictors
            if predictor not in sampled.columns
        ]
        if missing_predictors:
            raise ValueError(
                "Predictors missing after raster sampling: "
                f"{missing_predictors}"
            )

        if dropna:
            sampled = sampled.dropna(subset=self.predictors)

        return sampled.reset_index(drop=True)

    def validate(self, scheme, **kwargs):
        """Run a validation strategy against this analysis."""
        result = scheme.run(self, **kwargs)
        self.validation_ = result
        return result

    @abstractmethod
    def fit(self, *args, **kwargs) -> RSFFit:
        """Fit the estimator to this analysis."""

    @abstractmethod
    def _run_loio(self, scheme, **kwargs):
        """Estimator-specific implementation used by LOIO validation."""
