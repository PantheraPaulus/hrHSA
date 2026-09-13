"""Stateful frequentist RSF workflow built on the functional kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hsa.compute.prepared import PreparedDataset
from hsa.rsf.base import RSFAnalysis, RSFFit
from hsa.rsf.cv import _extract_predictor_cols, leave_one_individual_out_rsf
from hsa.rsf.hpc import leave_one_individual_out_rsf_prepared
from hsa.rsf.model import fit_rsf, predict_rsf_points
from hsa.rsf.schemes import FrequentistLOIOResult
from hsa.rsf.surface import predict_rsf_surface
from hsa.rsf.surface_fast import predict_rsf_surface_chunked
from hsa.types import FeatureSpec


@dataclass
class FrequentistRSFFit(RSFFit):
    """Fitted statsmodels RSF with all prediction metadata kept together."""

    model: Any
    scaler: Any
    spec: FeatureSpec
    meta: dict[str, Any]
    env: Any = None

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def coefficients(self, *args, **kwargs) -> pd.DataFrame:
        params = pd.Series(self.model.params)
        out = pd.DataFrame(
            {
                "term": params.index,
                "estimate": params.to_numpy(dtype=float),
            }
        )

        if hasattr(self.model, "bse"):
            bse = pd.Series(self.model.bse).reindex(params.index)
            out["std_error"] = bse.to_numpy(dtype=float)

        if hasattr(self.model, "pvalues"):
            pvalues = pd.Series(self.model.pvalues).reindex(params.index)
            out["p_value"] = pvalues.to_numpy(dtype=float)

        if hasattr(self.model, "conf_int"):
            alpha = kwargs.pop("alpha", 0.05)
            ci = self.model.conf_int(alpha=alpha)
            ci = pd.DataFrame(ci).reindex(params.index)
            if ci.shape[1] >= 2:
                out["lower"] = ci.iloc[:, 0].to_numpy(dtype=float)
                out["upper"] = ci.iloc[:, 1].to_numpy(dtype=float)

        return out

    def predict_points(
        self,
        df: pd.DataFrame,
        *,
        pred_col: str = "rsf_pred",
    ) -> pd.DataFrame:
        return predict_rsf_points(
            df,
            self.model,
            self.scaler,
            self.spec,
            self.meta,
            pred_col=pred_col,
        )

    def predict_surface(
        self,
        env=None,
        *args,
        engine: str = "reference",
        **kwargs,
    ):
        """Predict an RSF surface using the reference or fused chunked engine."""
        env = self.env if env is None else env
        if env is None:
            raise ValueError("No environmental raster was supplied or stored.")
        if engine == "reference":
            return predict_rsf_surface(
                env,
                self.model,
                self.scaler,
                self.spec,
                self.meta,
                *args,
                **kwargs,
            )
        if engine == "chunked":
            if args:
                raise TypeError(
                    "The chunked surface engine accepts keyword arguments only."
                )
            return predict_rsf_surface_chunked(
                env,
                self.model,
                self.scaler,
                self.spec,
                self.meta,
                **kwargs,
            )
        raise ValueError("engine must be 'reference' or 'chunked'.")


class FrequentistRSF(RSFAnalysis):
    """Frequentist logistic resource-selection analysis.

    The object owns relocations, environmental predictors, availability domains
    and the most recent fit. Numerical fitting remains delegated to ``fit_rsf``.
    """

    def __init__(
        self,
        reloc,
        env,
        *,
        spec: FeatureSpec,
        id_col: str = "Individual_ID",
        timestamp_col: str = "Timestamp",
        domain=None,
        domain_quantile: float = 0.95,
    ):
        self.spec = spec
        predictors = _extract_predictor_cols(spec)
        super().__init__(
            reloc,
            env,
            predictors=predictors,
            id_col=id_col,
            timestamp_col=timestamp_col,
            domain=domain,
            domain_quantile=domain_quantile,
        )

    def fit(
        self,
        *,
        sampling_factor: int = 10,
        thin_dt: str | None = None,
        min_available_proportion: float = 0.0,
        clean: bool = True,
        seed: int = 42,
        prepared: PreparedDataset | str | Path | None = None,
        method: str = "newton",
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> FrequentistRSFFit:
        """Fit a pooled frequentist RSF across the analysis relocations.

        Passing ``prepared`` bypasses availability generation and environmental
        extraction entirely and fits from the cached Parquet partitions.

        ``method`` and ``fit_kwargs`` are passed to :func:`hsa.rsf.fit_rsf`.
        Newton remains the conservative reference default. For very large,
        low-dimensional continuous RSFs, workstation benchmarks found L-BFGS to
        be a useful lower-latency option when coefficient/log-likelihood agreement
        with the reference fit has been verified for the analysis.
        """
        if prepared is None:
            sampled = self.sample_training_data(
                sampling_factor=sampling_factor,
                thin_dt=thin_dt,
                id_cols=None,
                seed=seed,
                bands=self.predictors,
            )
        else:
            if not isinstance(prepared, PreparedDataset):
                prepared = PreparedDataset.open(prepared)
            missing = sorted(set(self.predictors).difference(prepared.predictors))
            if missing:
                raise ValueError(
                    f"Prepared dataset is missing fitted predictors: {missing}"
                )
            sampled = prepared.load()

        model, scaler, fitted_spec, meta = fit_rsf(
            sampled,
            self.spec,
            min_available_proportion=min_available_proportion,
            clean=clean,
            method=method,
            fit_kwargs=fit_kwargs,
        )

        fit = FrequentistRSFFit(
            model=model,
            scaler=scaler,
            spec=fitted_spec,
            meta=meta,
            env=self.env,
        )
        self.fit_ = fit
        return fit

    def _run_loio(self, scheme, **kwargs) -> FrequentistLOIOResult:
        """Run serial reference LOIO or the prepared/distributed execution path."""
        prepared = kwargs.pop("prepared", None)
        client = kwargs.pop("client", None)
        surface_engine = kwargs.pop("surface_engine", "chunked")
        target_chunk_mb = int(kwargs.pop("target_chunk_mb", 256))
        keep_models = bool(kwargs.pop("keep_models", False))
        keep_surfaces = bool(kwargs.pop("keep_surfaces", False))

        if prepared is None:
            result = leave_one_individual_out_rsf(
                self.reloc,
                self.env,
                self.spec,
                id_col=self.id_col,
                heldout=scheme.heldout,
                domain=self.domain,
                domain_quantile=self.domain_quantile,
                thin_train_dt=scheme.thin_train_dt,
                thin_test_dt=scheme.thin_test_dt,
                sampling_factor_train=scheme.sampling_factor_train,
                n_background_boyce=scheme.n_background,
                n_bins=scheme.n_bins,
                seed=scheme.seed,
                **kwargs,
            )
        else:
            fit_method = str(kwargs.pop("fit_method", "newton"))
            fit_kwargs = kwargs.pop("fit_kwargs", None)
            blas_threads = int(kwargs.pop("blas_threads", 1))

            if not isinstance(prepared, PreparedDataset):
                prepared = PreparedDataset.open(prepared)
            prepared_thin = prepared.metadata.get("thin_dt")
            if scheme.thin_train_dt != prepared_thin or scheme.thin_test_dt != prepared_thin:
                raise ValueError(
                    "Prepared LOIO requires the same thinning rule for cached training and "
                    "held-out points. Build the cache with thin_dt matching both "
                    "scheme.thin_train_dt and scheme.thin_test_dt. For the common unthinned "
                    "case, leave all three values as None."
                )
            prepared_factor = int(prepared.metadata.get("sampling_factor", -1))
            if prepared_factor != int(scheme.sampling_factor_train):
                raise ValueError(
                    "Prepared sampling_factor does not match the LOIO scheme: "
                    f"{prepared_factor} != {scheme.sampling_factor_train}."
                )
            result = leave_one_individual_out_rsf_prepared(
                self,
                prepared,
                self.spec,
                heldout=scheme.heldout,
                n_background=scheme.n_background,
                n_bins=scheme.n_bins,
                seed=scheme.seed,
                client=client,
                surface_engine=surface_engine,
                target_chunk_mb=target_chunk_mb,
                keep_models=keep_models,
                keep_surfaces=keep_surfaces,
                fit_method=fit_method,
                fit_kwargs=fit_kwargs,
                blas_threads=blas_threads,
                **kwargs,
            )

        (
            summary,
            params,
            boyce_bins,
            calibration_bins,
            diagnostics,
        ) = result

        return FrequentistLOIOResult(
            summary=summary,
            params=params,
            boyce_bins=boyce_bins,
            calibration_bins=calibration_bins,
            diagnostics=diagnostics,
            scheme=scheme,
            analysis=self,
        )


__all__ = [
    "FrequentistRSF",
    "FrequentistRSFFit",
]
