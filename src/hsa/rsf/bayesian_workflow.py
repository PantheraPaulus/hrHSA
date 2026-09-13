"""Stateful hierarchical Bayesian RSF workflow built on the functional kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hsa.compute.prepared import PreparedDataset
from hsa.rsf.base import RSFAnalysis, RSFFit
from hsa.rsf.bayesian import (
    build_bayesian_rsf_model,
    evaluate_bayesian_rsf,
    plot_bayesian_rsf_diagnostics,
    predict_bayesian_rsf_surface,
    prepare_bayesian_rsf_data,
)
from hsa.rsf.bayesian_cv import leave_one_individual_out_bayesian_rsf
from hsa.rsf.bayesian_hpc import leave_one_individual_out_bayesian_rsf_prepared
from hsa.rsf.bayesian_shrinkage import (
    collinearity_diagnostics,
    posterior_beta_correlation,
    regularized_horseshoe_summary,
)
from hsa.rsf.schemes import BayesianLOIOResult


@dataclass
class BayesianRSFFit(RSFFit):
    """Fitted hierarchical Bayesian RSF with inference state kept together."""

    model: Any
    idata: Any
    data: dict[str, Any]
    meta: Mapping[str, Any]
    predictors: tuple[str, ...]
    env: Any = None

    def summary(
        self,
        *,
        ci_prob: float = 0.95,
        include_random_effects: bool = True,
    ):
        return evaluate_bayesian_rsf(
            self.idata,
            ci_prob=ci_prob,
            include_random_effects=include_random_effects,
        )

    def coefficients(
        self,
        *,
        ci_prob: float = 0.95,
        include_intercept: bool = True,
    ) -> pd.DataFrame:
        """Return tidy population-level posterior coefficients."""
        try:
            import arviz as az
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ArviZ is required; install hsa[bayesian].") from exc

        rows: list[dict[str, Any]] = []
        posterior = self.idata.posterior

        if include_intercept and "alpha" in posterior:
            values = np.asarray(posterior["alpha"].values, dtype=float).ravel()
            hdi = np.asarray(az.hdi(values, prob=ci_prob), dtype=float)
            rows.append(
                {
                    "term": "alpha",
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "lower": float(hdi[0]),
                    "upper": float(hdi[1]),
                    "p_gt_zero": float(np.mean(values > 0)),
                }
            )

        beta = posterior["beta"]
        for predictor in self.predictors:
            values = np.asarray(
                beta.sel(predictor=predictor).values,
                dtype=float,
            ).ravel()
            hdi = np.asarray(az.hdi(values, prob=ci_prob), dtype=float)
            rows.append(
                {
                    "term": predictor,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "lower": float(hdi[0]),
                    "upper": float(hdi[1]),
                    "p_gt_zero": float(np.mean(values > 0)),
                    "odds_ratio_median": float(np.exp(np.median(values))),
                }
            )

        return pd.DataFrame(rows)

    def posterior_correlation(self) -> pd.DataFrame:
        """Return posterior correlations among population-level coefficients."""
        return posterior_beta_correlation(
            self.idata,
            predictors=self.predictors,
        )

    def collinearity_diagnostics(
        self,
        *,
        min_abs_predictor_corr: float = 0.5,
    ) -> pd.DataFrame:
        """Compare fitted-design and posterior-beta correlations."""
        return collinearity_diagnostics(
            self.idata,
            self.data,
            predictors=self.predictors,
            min_abs_predictor_corr=min_abs_predictor_corr,
        )

    def shrinkage_summary(
        self,
        *,
        ci_prob: float = 0.95,
    ) -> pd.DataFrame:
        """Summarize a fitted regularized-horseshoe population prior."""
        return regularized_horseshoe_summary(
            self.idata,
            ci_prob=ci_prob,
        )

    def plot_diagnostics(
        self,
        *,
        forest_predictors: Sequence[str] | None = None,
        ci_prob: float = 0.95,
    ):
        return plot_bayesian_rsf_diagnostics(
            self.idata,
            forest_predictors=forest_predictors,
            ci_prob=ci_prob,
        )

    def predict_surface(
        self,
        env=None,
        *,
        ci_prob: float = 0.95,
        threshold: float = 1.0,
        n_draws: int | None = 500,
        random_seed: int = 42,
        **kwargs,
    ):
        env = self.env if env is None else env
        if env is None:
            raise ValueError("No environmental raster was supplied or stored.")
        return predict_bayesian_rsf_surface(
            self.idata.posterior["beta"],
            env,
            self.meta,
            predictors=self.predictors,
            ci_prob=ci_prob,
            threshold=threshold,
            n_draws=n_draws,
            random_seed=random_seed,
            **kwargs,
        )


class BayesianRSF(RSFAnalysis):
    """Hierarchical Bayesian resource-selection analysis.

    Model configuration that defines the estimator belongs to the analysis
    object. Sampling controls that define a particular inference run belong to
    ``fit`` or ``validate``.
    """

    def __init__(
        self,
        reloc,
        env,
        *,
        predictors: Sequence[str],
        binning: Mapping[str, float | None] | None = None,
        id_col: str = "individual-local-identifier",
        timestamp_col: str = "Timestamp",
        domain=None,
        domain_quantile: float = 0.95,
        random_intercept: bool = True,
        random_slopes: Sequence[str] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            reloc,
            env,
            predictors=predictors,
            id_col=id_col,
            timestamp_col=timestamp_col,
            domain=domain,
            domain_quantile=domain_quantile,
        )

        self.binning = {} if binning is None else dict(binning)
        unknown_binning = sorted(set(self.binning).difference(self.predictors))
        if unknown_binning:
            raise ValueError(
                f"Binning specified for unknown predictors: {unknown_binning}"
            )

        self.random_intercept = bool(random_intercept)
        self.random_slopes = (
            []
            if random_slopes is None
            else list(dict.fromkeys(random_slopes))
        )
        missing_random = sorted(set(self.random_slopes).difference(self.predictors))
        if missing_random:
            raise ValueError(
                f"Random slopes are not fitted predictors: {missing_random}"
            )

        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    def fit(
        self,
        *,
        sampling_factor: int = 50,
        thin_dt: str | None = None,
        seed: int = 42,
        sample_kwargs: Mapping[str, Any] | None = None,
        prepared: PreparedDataset | str | Path | None = None,
        nuts_sampler: str | None = None,
    ) -> BayesianRSFFit:
        """Fit one hierarchical Bayesian model to all analysis individuals.

        ``prepared`` reuses cached environmental extraction. ``nuts_sampler`` is
        forwarded to :func:`pymc.sample`, allowing explicit benchmarking of the
        native PyMC sampler and optional ``nutpie``, ``numpyro`` or ``blackjax``
        backends without changing the model definition.
        """
        try:
            import pymc as pm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyMC is required; install hsa[bayesian].") from exc

        if prepared is None:
            sampled = self.sample_training_data(
                sampling_factor=sampling_factor,
                thin_dt=thin_dt,
                id_cols=self.id_col,
                seed=seed,
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
            if self.id_col not in sampled:
                raise ValueError(
                    f"Prepared dataset must retain Bayesian id column {self.id_col!r}."
                )

        bayes_data = prepare_bayesian_rsf_data(
            sampled,
            self.predictors,
            id_col=self.id_col,
            binning=self.binning,
        )

        model = build_bayesian_rsf_model(
            bayes_data,
            predictors=self.predictors,
            random_intercept=self.random_intercept,
            random_slopes=self.random_slopes,
            **self.model_kwargs,
        )

        sampling = {
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.9,
            "return_inferencedata": True,
        }
        if sample_kwargs is not None:
            sampling.update(dict(sample_kwargs))
        if nuts_sampler is not None:
            sampling.setdefault("nuts_sampler", nuts_sampler)
        sampling.setdefault("random_seed", seed)

        with model:
            idata = pm.sample(**sampling)

        fit = BayesianRSFFit(
            model=model,
            idata=idata,
            data=bayes_data,
            meta=bayes_data["meta"],
            predictors=tuple(self.predictors),
            env=self.env,
        )
        self.fit_ = fit
        return fit

    def _run_loio(self, scheme, **kwargs) -> BayesianLOIOResult:
        """Run serial reference LOIO or prepared fold-parallel Bayesian LOIO."""
        prepared = kwargs.pop("prepared", None)
        client = kwargs.pop("client", None)
        nuts_sampler = kwargs.pop("nuts_sampler", None)
        sample_kwargs = kwargs.pop("sample_kwargs", None)
        sampling = {} if sample_kwargs is None else dict(sample_kwargs)
        if nuts_sampler is not None:
            sampling.setdefault("nuts_sampler", nuts_sampler)

        # Estimator-defining model_kwargs live on the analysis object; a caller
        # may add/override run-specific values explicitly for a validation run.
        model_kwargs = dict(self.model_kwargs)
        override_model_kwargs = kwargs.pop("model_kwargs", None)
        if override_model_kwargs is not None:
            model_kwargs.update(dict(override_model_kwargs))

        if prepared is None:
            if client is not None:
                raise ValueError(
                    "Parallel Bayesian LOIO requires a PreparedDataset. Build one "
                    "with analysis.prepare(...) and pass prepared=... together with client=."
                )
            summary, params, boyce_bins, diagnostics = (
                leave_one_individual_out_bayesian_rsf(
                    self.reloc,
                    self.env,
                    predictors=self.predictors,
                    binning=self.binning,
                    id_col=self.id_col,
                    heldout=scheme.heldout,
                    domain=self.domain,
                    domain_quantile=self.domain_quantile,
                    thin_train_dt=scheme.thin_train_dt,
                    thin_test_dt=scheme.thin_test_dt,
                    sampling_factor_train=scheme.sampling_factor_train,
                    random_intercept=self.random_intercept,
                    random_slopes=self.random_slopes,
                    n_background_boyce=scheme.n_background,
                    n_bins=scheme.n_bins,
                    model_kwargs=model_kwargs,
                    sample_kwargs=sampling,
                    seed=scheme.seed,
                    **kwargs,
                )
            )
        else:
            store_scores = bool(kwargs.pop("store_scores", False))
            keep_idata = bool(kwargs.pop("keep_idata", False))
            keep_model = bool(kwargs.pop("keep_model", False))
            keep_bayes_data = bool(kwargs.pop("keep_bayes_data", False))

            if not isinstance(prepared, PreparedDataset):
                prepared = PreparedDataset.open(prepared)
            prepared_thin = prepared.metadata.get("thin_dt")
            if scheme.thin_train_dt != prepared_thin:
                raise ValueError(
                    "Prepared Bayesian LOIO requires cached training thinning to match "
                    "scheme.thin_train_dt. Rebuild the cache with the same thin_dt."
                )
            prepared_factor = int(prepared.metadata.get("sampling_factor", -1))
            if prepared_factor != int(scheme.sampling_factor_train):
                raise ValueError(
                    "Prepared sampling_factor does not match the LOIO scheme: "
                    f"{prepared_factor} != {scheme.sampling_factor_train}."
                )

            summary, params, boyce_bins, diagnostics = (
                leave_one_individual_out_bayesian_rsf_prepared(
                    self,
                    prepared,
                    predictors=self.predictors,
                    binning=self.binning,
                    heldout=scheme.heldout,
                    thin_test_dt=scheme.thin_test_dt,
                    random_intercept=self.random_intercept,
                    random_slopes=self.random_slopes,
                    n_background=scheme.n_background,
                    n_bins=scheme.n_bins,
                    model_kwargs=model_kwargs,
                    sample_kwargs=sampling,
                    client=client,
                    store_scores=store_scores,
                    keep_idata=keep_idata,
                    keep_model=keep_model,
                    keep_bayes_data=keep_bayes_data,
                    seed=scheme.seed,
                    **kwargs,
                )
            )

        return BayesianLOIOResult(
            summary=summary,
            params=params,
            boyce_bins=boyce_bins,
            diagnostics=diagnostics,
            scheme=scheme,
            analysis=self,
        )


__all__ = [
    "BayesianRSF",
    "BayesianRSFFit",
]
