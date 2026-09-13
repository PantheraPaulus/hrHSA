"""Hierarchical Bayesian categorical/softmax step-selection workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
)


def build_hierarchical_ssf_model(
    arrays: SSFChoiceArrays,
    *,
    mu_sigma: float = 1.0,
    heterogeneity_sigma: float = 0.5,
):
    """Build a non-centred hierarchical categorical SSF in PyMC."""
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
    with pm.Model(coords=coords) as model:
        X_data = pm.Data(
            "X",
            arrays.X,
            dims=("stratum", "choice", "predictor"),
        )

        if arrays.offset is None:
            offset_data = 0.0

        else:
            offset_data = pm.Data(
                "offset",
                arrays.offset,
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
            (
                X_data
                * beta_s[:, None, :]
            ).sum(axis=-1)
            + offset_data
        )
        
        pm.Categorical(
            "y",
            logit_p=eta,
            observed=arrays.chosen,
            dims="stratum",
        )
    return model


def posterior_choice_probabilities_known_individual(
    idata,
    arrays: SSFChoiceArrays,
    *,
    beta_var: str = "beta",
    batch_size: int = 250,
) -> np.ndarray:
    """Average conditional choice probabilities over individual posterior betas."""
    beta = (
        idata.posterior[beta_var]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "individual", "predictor")
        .values
    )
    n_strata, n_choices, _ = arrays.X.shape
    mean_probability = np.empty((n_strata, n_choices), dtype=float)

    for start in range(0, n_strata, batch_size):
        stop = min(start + batch_size, n_strata)
        X_batch = arrays.X[start:stop]
        idx_batch = arrays.individual_idx[start:stop]
        beta_batch = beta[:, idx_batch, :]
        eta = np.einsum(
            "bjk,sbk->sbj",
            X_batch,
            beta_batch,
            optimize=True,
        )
        eta -= eta.max(axis=-1, keepdims=True)
        probability = np.exp(eta)
        probability /= probability.sum(axis=-1, keepdims=True)
        mean_probability[start:stop] = probability.mean(axis=0)

    return mean_probability


def posterior_choice_probabilities_new_individual(
    idata,
    X: np.ndarray,
    *,
    mode: str = "new_individual",
    mu_var: str = "mu_beta",
    sigma_var: str = "sigma_beta",
    batch_size: int = 250,
    random_seed: int = 42,
) -> np.ndarray:
    """Predict choice sets for an individual absent from model fitting.

    For ``mode='new_individual'`` one latent coefficient vector is drawn for
    each posterior draw and reused across all strata of the held-out animal.
    """
    rng = np.random.default_rng(random_seed)
    mu = (
        idata.posterior[mu_var]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "predictor")
        .values
    )

    if mode == "population_mean":
        beta_draws = mu
    elif mode == "new_individual":
        sigma = (
            idata.posterior[sigma_var]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "predictor")
            .values
        )
        beta_draws = mu + sigma * rng.normal(size=mu.shape)
    else:
        raise ValueError(
            "mode must be 'population_mean' or 'new_individual'."
        )

    X = np.asarray(X, dtype=float)
    n_strata, n_choices, _ = X.shape
    mean_probability = np.empty((n_strata, n_choices), dtype=float)

    for start in range(0, n_strata, batch_size):
        stop = min(start + batch_size, n_strata)
        eta = np.einsum(
            "bjk,sk->sbj",
            X[start:stop],
            beta_draws,
            optimize=True,
        )
        eta -= eta.max(axis=-1, keepdims=True)
        probability = np.exp(eta)
        probability /= probability.sum(axis=-1, keepdims=True)
        mean_probability[start:stop] = probability.mean(axis=0)

    return mean_probability


def _equal_tail_interval(values: np.ndarray, prob: float) -> tuple[float, float]:
    """Version-independent equal-tail credible interval."""
    if not 0 < prob < 1:
        raise ValueError("ci_prob must be in (0, 1).")
    tail = (1.0 - prob) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _arviz_summary(idata, *, var_names, ci_prob: float):
    """Support both current arviz-stats and legacy ArviZ summary keywords."""
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ArviZ is required; install hsa[bayesian].") from exc

    try:
        return az.summary(
            idata,
            var_names=var_names,
            ci_prob=ci_prob,
        )
    except TypeError:
        return az.summary(
            idata,
            var_names=var_names,
            hdi_prob=ci_prob,
        )


@dataclass
class BayesianSSFFit(SSFFit):
    """Fitted hierarchical Bayesian SSF with posterior and design metadata."""

    model: Any
    idata: Any
    arrays: SSFChoiceArrays
    data: pd.DataFrame
    raw_predictors: tuple[str, ...]
    predictors: tuple[str, ...]
    scaling: dict[str, dict[str, float]]
    id_col: str

    def summary(
        self,
        *,
        ci_prob: float = 0.89,
        include_individual: bool = False,
    ):
        """Return population/heterogeneity MCMC diagnostics and intervals."""
        names = ["mu_beta", "sigma_beta"]
        if include_individual:
            names.append("beta")
        return _arviz_summary(
            self.idata,
            var_names=names,
            ci_prob=ci_prob,
        )

    def coefficients(self, *, ci_prob: float = 0.89) -> pd.DataFrame:
        """Return tidy population-level posterior coefficients."""
        rows = []
        for predictor in self.predictors:
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
                    "relative_selection_median": float(
                        np.exp(np.median(values))
                    ),
                }
            )
        return pd.DataFrame(rows)

    def individual_coefficients(
        self,
        *,
        ci_prob: float = 0.89,
    ) -> pd.DataFrame:
        """Return tidy partially pooled individual posterior coefficients."""
        rows = []
        beta = self.idata.posterior["beta"]
        for individual in self.arrays.individuals:
            for predictor in self.predictors:
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

    def heterogeneity(self, *, ci_prob: float = 0.89) -> pd.DataFrame:
        """Summarize posterior between-individual slope heterogeneity."""
        rows = []
        for predictor in self.predictors:
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

    def _resolve_plot_predictors(
        self,
        predictors: Sequence[str] | None,
    ) -> list[str] | None:
        """Accept either raw predictor names or their standardized model names."""
        if predictors is None:
            return None
        mapping = dict(zip(self.raw_predictors, self.predictors))
        return [mapping.get(str(predictor), str(predictor)) for predictor in predictors]

    def plot_trace(
        self,
        *,
        include_individual: bool = False,
        individuals: Sequence[str] | None = None,
    ):
        """Plot posterior distributions and chain traces for SSF parameters."""
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
    ) -> dict[str, Any]:
        """Plot population, individual and heterogeneity posterior forests.

        Predictor names may be given either as raw names such as ``elevation``
        or as standardized model names such as ``elevation_z``.
        """
        return plot_bayesian_ssf_forests(
            self.idata,
            predictors=self._resolve_plot_predictors(predictors),
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
    ) -> dict[str, Any]:
        """Plot compact trace diagnostics plus hierarchical SSF forest plots.

        This mirrors ``BayesianRSFFit.plot_diagnostics()``. By default the trace
        is kept compact at the population/heterogeneity level, while partially
        pooled individual coefficients are displayed as one forest per predictor.
        """
        return plot_bayesian_ssf_diagnostics(
            self.idata,
            forest_predictors=self._resolve_plot_predictors(forest_predictors),
            ci_prob=ci_prob,
            include_population_forest=include_population_forest,
            include_heterogeneity_forest=include_heterogeneity_forest,
            include_individual_forests=include_individual_forests,
            include_individual_trace=include_individual_trace,
            trace_individuals=trace_individuals,
        )

    def choice_scores(self, *, batch_size: int = 250):
        """Score observed choices after integrating over the posterior."""
        probability = posterior_choice_probabilities_known_individual(
            self.idata,
            self.arrays,
            batch_size=batch_size,
        )
        per, summary = score_choice_probabilities(
            probability,
            self.arrays.chosen,
        )
        meta = self.arrays.strata.reset_index(drop=True)
        per[[self.id_col, "stratum_id"]] = meta[
            [self.id_col, "stratum_id"]
        ].to_numpy()
        return {
            "per_stratum": per,
            "summary": summary,
            "mean_probability": probability,
        }

    def divergences(self) -> int:
        """Return the total number of divergent posterior transitions."""
        if "diverging" not in self.idata.sample_stats:
            return 0
        return int(
            np.asarray(self.idata.sample_stats["diverging"]).sum()
        )

    def psis_loo(self, *, pointwise: bool = True):
        """Run stratum-wise PSIS-LOO and express ELPD on the SSF gain scale.

        This is a local new-stratum/known-individual diagnostic. Use exact
        temporal-block CV for temporal transfer and LOIO for a new animal.
        """
        try:
            import pymc as pm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyMC is required; install hsa[bayesian].") from exc

        if not hasattr(self.idata, "log_likelihood"):
            with self.model:
                pm.compute_log_likelihood(
                    self.idata,
                    var_names=["y"],
                    model=self.model,
                    extend_inferencedata=True,
                )

        loo = pm.stats.loo(
            self.idata,
            var_name="y",
            pointwise=pointwise,
        )
        if not pointwise:
            return {"loo": loo}

        elpd_i = getattr(loo, "elpd_i", None)
        if elpd_i is None:
            elpd_i = getattr(loo, "loo_i")
        elpd_i = np.asarray(elpd_i, dtype=float).reshape(-1)
        gain = elpd_i + np.log(self.arrays.n_choices)

        pareto_k = np.asarray(loo.pareto_k, dtype=float).reshape(-1)
        good_k = float(getattr(loo, "good_k", 0.7))
        summary = pd.Series(
            {
                "n_strata": len(gain),
                "mean_log_score_gain": float(gain.mean()),
                "median_log_score_gain": float(np.median(gain)),
                "fraction_gain_positive": float(np.mean(gain > 0)),
                "total_elpd_gain": float(gain.sum()),
                "predictive_advantage": float(np.exp(gain.mean())),
                "good_k_threshold": good_k,
                "median_k": float(np.median(pareto_k)),
                "q95_k": float(np.quantile(pareto_k, 0.95)),
                "q99_k": float(np.quantile(pareto_k, 0.99)),
                "max_k": float(np.max(pareto_k)),
                "fraction_above_good_k": float(
                    np.mean(pareto_k > good_k)
                ),
                "fraction_above_1": float(np.mean(pareto_k > 1.0)),
            }
        )
        per_stratum = self.arrays.strata.copy()
        per_stratum["elpd_loo"] = elpd_i
        per_stratum["log_score_gain"] = gain
        per_stratum["pareto_k"] = pareto_k
        return {
            "loo": loo,
            "summary": summary,
            "per_stratum": per_stratum,
        }


class BayesianSSF(SSFAnalysis):
    """Hierarchical Bayesian SSF with individual random slopes for all predictors."""

    def __init__(
        self,
        *args,
        model_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_kwargs = (
            {}
            if model_kwargs is None
            else dict(model_kwargs)
        )

    def fit(
        self,
        *,
        scaling: Mapping[str, Mapping[str, float]] | None = None,
        seed: int = 42,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> BayesianSSFFit:
        """Fit one hierarchical Bayesian categorical SSF.

        ``scaling`` may be supplied to make a pilot fit directly comparable to
        a reference fit on the same standardized predictor scale. Validation
        schemes always estimate scaling from their training partition instead.
        """
        try:
            import pymc as pm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyMC is required; install hsa[bayesian].") from exc

        self.validate_predictors()
        complete = complete_ssf_strata(
            self.choices,
            predictors=self.predictors,
            id_col=self.id_col,
            expected_n_choices=self.n_available + 1,
        )
        if scaling is None:
            fitted_scaling = fit_ssf_scaling(
                complete,
                self.predictors,
            )
        else:
            fitted_scaling = {
                predictor: {
                    "mean": float(scaling[predictor]["mean"]),
                    "sd": float(scaling[predictor]["sd"]),
                }
                for predictor in self.predictors
            }

        scaled = apply_ssf_scaling(
            complete,
            self.predictors,
            fitted_scaling,
        )
        model_predictors = tuple(
            f"{predictor}_z"
            for predictor in self.predictors
        )
        arrays = build_ssf_choice_arrays(
            scaled,
            id_col=self.id_col,
            predictors=model_predictors,
        )
        model = build_hierarchical_ssf_model(
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

        fit = BayesianSSFFit(
            model=model,
            idata=idata,
            arrays=arrays,
            data=scaled,
            raw_predictors=tuple(self.predictors),
            predictors=model_predictors,
            scaling=fitted_scaling,
            id_col=self.id_col,
        )
        self.fit_ = fit
        return fit


__all__ = [
    "BayesianSSF",
    "BayesianSSFFit",
    "build_hierarchical_ssf_model",
    "posterior_choice_probabilities_known_individual",
    "posterior_choice_probabilities_new_individual",
    "plot_bayesian_ssf_trace",
    "plot_bayesian_ssf_forests",
    "plot_bayesian_ssf_diagnostics",
]
