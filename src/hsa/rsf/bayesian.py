"""Hierarchical Bayesian resource-selection models.

The functions in this module are intentionally independent of the vulture
notebooks. They prepare aggregated use--availability data, build the PyMC
model, summarize inference, plot sampling diagnostics, and project population
posterior coefficients over an environmental raster.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from hsa.rsf.bayesian_shrinkage import build_population_beta_prior


def prepare_bayesian_rsf_data(
    df: pd.DataFrame,
    predictors: list[str],
    *,
    id_col: str = "individual-local-identifier",
    response_col: str = "used",
    binning: dict[str, float | None] | None = None,
    standardize: bool = True,
) -> dict[str, Any]:
    """Prepare aggregated Binomial data for a hierarchical Bayesian RSF.

    Continuous predictors can be binned in their original ecological units
    before aggregation. Standardization always uses the mean and standard
    deviation of the original, unbinned observations so that coefficients are
    interpretable per one standard deviation of the original predictor.
    """

    predictors = list(predictors)
    required = [id_col, response_col, *predictors]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not predictors:
        raise ValueError("At least one predictor is required.")

    model_df = df[required].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    if model_df.empty:
        raise ValueError("No complete rows remain for Bayesian RSF preparation.")

    model_df[response_col] = model_df[response_col].astype(int)
    invalid_response = ~model_df[response_col].isin([0, 1])
    if invalid_response.any():
        raise ValueError(f"{response_col!r} must contain only binary 0/1 values.")

    binning = {} if binning is None else dict(binning)
    unknown_binning = sorted(set(binning).difference(predictors))
    if unknown_binning:
        raise ValueError(f"Binning specified for unknown predictors: {unknown_binning}")

    meta: dict[str, Any] = {}
    aggregation_cols: list[str] = []

    for predictor in predictors:
        values = pd.to_numeric(model_df[predictor], errors="coerce")
        x_mean = float(values.mean())
        x_sd = float(values.std(ddof=0))
        if not np.isfinite(x_mean):
            raise ValueError(f"Predictor {predictor!r} has no finite mean.")
        if standardize and (not np.isfinite(x_sd) or x_sd <= 0):
            raise ValueError(f"Predictor {predictor!r} has zero/invalid standard deviation.")

        bin_width = binning.get(predictor)
        if bin_width is not None:
            if bin_width <= 0:
                raise ValueError(f"Bin width for {predictor!r} must be > 0.")
            aggregate_col = f"{predictor}__bin"
            model_df[aggregate_col] = np.round(model_df[predictor] / bin_width) * bin_width
        else:
            aggregate_col = predictor

        aggregation_cols.append(aggregate_col)
        meta[predictor] = {
            "mean": x_mean,
            "sd": x_sd,
            "bin_width": bin_width,
            "aggregation_col": aggregate_col,
        }

    aggregated = (
        model_df.groupby(
            [id_col, *aggregation_cols],
            observed=True,
            as_index=False,
        )
        .agg(
            used_count=(response_col, "sum"),
            n_trials=(response_col, "size"),
        )
    )

    if aggregated.empty:
        raise ValueError("Aggregation produced no Bayesian RSF observations.")

    individuals = pd.Index(aggregated[id_col].dropna().unique()).tolist()
    id_lookup = {individual: i for i, individual in enumerate(individuals)}
    id_idx = aggregated[id_col].map(id_lookup).to_numpy(dtype=np.int32)

    x: dict[str, np.ndarray] = {}
    columns: list[np.ndarray] = []
    for predictor in predictors:
        aggregate_col = meta[predictor]["aggregation_col"]
        values = aggregated[aggregate_col].to_numpy(dtype=float)
        if standardize:
            values = (values - meta[predictor]["mean"]) / meta[predictor]["sd"]
        x[predictor] = values
        columns.append(values)

    X = np.column_stack(columns).astype(np.float64, copy=False)
    used_count = aggregated["used_count"].to_numpy(dtype=np.int64)
    n_trials = aggregated["n_trials"].to_numpy(dtype=np.int64)

    coords = {
        "obs": np.arange(len(aggregated)),
        "individual": individuals,
        "predictor": predictors,
    }

    meta["_model"] = {
        "id_col": id_col,
        "response_col": response_col,
        "predictors": predictors,
        "standardize": standardize,
        "n_raw": int(len(model_df)),
        "n_aggregated": int(len(aggregated)),
        "compression_ratio": float(len(model_df) / len(aggregated)),
    }

    return {
        "X": X,
        "x": x,
        "used_count": used_count,
        "n_trials": n_trials,
        "id_idx": id_idx,
        "individuals": individuals,
        "predictors": predictors,
        "coords": coords,
        "data": aggregated,
        "meta": meta,
    }


def build_bayesian_rsf_model(
    bayes_data: dict,
    *,
    predictors: Sequence[str] | None = None,
    random_intercept: bool = True,
    random_slopes: Sequence[str] | None = None,
    alpha_prior_sigma: float = 2.5,
    beta_prior: str = "normal",
    beta_prior_sigma: float = 1.0,
    shrinkage: Mapping[str, Any] | None = None,
    sigma_prior_rate: float = 1.0,
    store_eta: bool = False,
):
    """Build a non-centred hierarchical Bayesian RSF in PyMC.

    Population slopes use independent Normal priors by default. Setting
    ``beta_prior='regularized_horseshoe'`` enables optional sparse global-local
    shrinkage; ``shrinkage`` then configures the expected sparsity or explicit
    global scale and finite slab.

    ``store_eta`` controls whether the observation-level linear predictor is
    registered as a :class:`pymc.Deterministic`. It defaults to ``False`` because
    storing one value per aggregated observation for every posterior draw can
    dominate memory, serialization, and inference time in large RSFs. The
    likelihood always uses the same symbolic ``eta`` expression; set
    ``store_eta=True`` only when posterior draws of the full observation-level
    linear predictor are explicitly required.
    """

    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyMC is required; install hsa[bayesian].") from exc

    available_predictors = list(bayes_data["predictors"])
    if predictors is None:
        predictors = available_predictors
    else:
        predictors = list(predictors)
        missing = sorted(set(predictors).difference(available_predictors))
        if missing:
            raise ValueError(f"Predictors not present in bayes_data: {missing}")

    random_slopes = [] if random_slopes is None else list(random_slopes)
    missing_random = sorted(set(random_slopes).difference(predictors))
    if missing_random:
        raise ValueError(f"Random slopes are not fitted predictors: {missing_random}")

    normalized_beta_prior = str(beta_prior).strip().lower().replace("-", "_")
    if normalized_beta_prior == "regularized_horseshoe":
        standardized = bool(
            bayes_data.get("meta", {})
            .get("_model", {})
            .get("standardize", False)
        )
        if not standardized:
            raise ValueError(
                "regularized_horseshoe requires standardized predictors; "
                "prepare Bayesian data with standardize=True."
            )

    predictor_idx = [available_predictors.index(p) for p in predictors]
    X = np.asarray(bayes_data["X"][:, predictor_idx], dtype=float)
    id_idx = np.asarray(bayes_data["id_idx"], dtype=np.int32)
    used_count = np.asarray(bayes_data["used_count"], dtype=np.int64)
    n_trials = np.asarray(bayes_data["n_trials"], dtype=np.int64)

    coords = {
        "obs": np.arange(X.shape[0]),
        "individual": list(bayes_data["individuals"]),
        "predictor": list(predictors),
    }
    if random_slopes:
        coords["random_slope"] = random_slopes

    random_slope_idx = np.array([predictors.index(p) for p in random_slopes], dtype=int)

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X, dims=("obs", "predictor"))
        id_data = pm.Data("id_idx", id_idx, dims="obs")
        n_data = pm.Data("n_trials", n_trials, dims="obs")

        alpha = pm.Normal("alpha", mu=0.0, sigma=alpha_prior_sigma)
        beta = build_population_beta_prior(
            predictors=predictors,
            used_count=used_count,
            beta_prior=beta_prior,
            beta_prior_sigma=beta_prior_sigma,
            shrinkage=shrinkage,
        )

        if random_intercept:
            sigma_alpha = pm.Exponential("sigma_alpha", lam=sigma_prior_rate)
            z_alpha = pm.Normal("z_alpha", mu=0.0, sigma=1.0, dims="individual")
            alpha_ind = pm.Deterministic("alpha_ind", z_alpha * sigma_alpha, dims="individual")
            alpha_total = pm.Deterministic("alpha_total", alpha + alpha_ind, dims="individual")
            eta = alpha_total[id_data] + (X_data * beta).sum(axis=1)
        else:
            eta = alpha + (X_data * beta).sum(axis=1)

        if random_slopes:
            sigma_beta = pm.Exponential(
                "sigma_beta", lam=sigma_prior_rate, dims="random_slope"
            )
            z_beta = pm.Normal(
                "z_beta", mu=0.0, sigma=1.0, dims=("individual", "random_slope")
            )
            beta_ind = pm.Deterministic(
                "beta_ind",
                z_beta * sigma_beta,
                dims=("individual", "random_slope"),
            )
            pm.Deterministic(
                "beta_total",
                beta[random_slope_idx] + beta_ind,
                dims=("individual", "random_slope"),
            )
            X_random = X_data[:, random_slope_idx]
            eta = eta + (X_random * beta_ind[id_data]).sum(axis=1)

        if store_eta:
            pm.Deterministic("eta", eta, dims="obs")
        pm.Binomial(
            "used",
            n=n_data,
            logit_p=eta,
            observed=used_count,
            dims="obs",
        )

    return model


def evaluate_bayesian_rsf(
    idata,
    *,
    ci_prob: float = 0.95,
    include_random_effects: bool = True,
) -> dict[str, Any]:
    """Return compact convergence, coefficient, and sampler diagnostics."""

    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ArviZ is required; install hsa[bayesian].") from exc

    posterior = idata.posterior
    coefficient_vars = [
        v for v in ("alpha", "beta", "sigma_alpha", "sigma_beta") if v in posterior
    ]
    if include_random_effects:
        coefficient_vars += [v for v in ("alpha_total", "beta_total") if v in posterior]

    summary = az.summary(
        idata,
        var_names=coefficient_vars,
        ci_prob=ci_prob,
        ci_kind="hdi",
        kind="all",
    )

    diagnostics_vars = [
        v
        for v in (
            "alpha",
            "beta",
            "sigma_alpha",
            "sigma_beta",
            "hs_tau",
            "hs_lambda",
            "hs_c2",
            "hs_beta_raw",
        )
        if v in posterior
    ]
    diagnostics = az.summary(idata, var_names=diagnostics_vars, kind="diagnostics")

    n_divergences = 0
    max_tree_depth = np.nan
    mean_n_steps = np.nan
    if hasattr(idata, "sample_stats"):
        stats = idata.sample_stats
        if "diverging" in stats:
            n_divergences = int(stats["diverging"].sum().item())
        if "tree_depth" in stats:
            max_tree_depth = int(stats["tree_depth"].max().item())
        if "n_steps" in stats:
            mean_n_steps = float(stats["n_steps"].mean().item())

    return {
        "summary": summary,
        "diagnostics": diagnostics,
        "max_rhat": float(diagnostics["r_hat"].max()) if "r_hat" in diagnostics else np.nan,
        "min_ess_bulk": (
            float(diagnostics["ess_bulk"].min()) if "ess_bulk" in diagnostics else np.nan
        ),
        "min_ess_tail": (
            float(diagnostics["ess_tail"].min()) if "ess_tail" in diagnostics else np.nan
        ),
        "n_divergences": n_divergences,
        "max_tree_depth": max_tree_depth,
        "mean_n_steps": mean_n_steps,
    }


def plot_bayesian_rsf_diagnostics(
    idata,
    *,
    forest_predictors: Sequence[str] | None = None,
    ci_prob: float = 0.95,
    legend: bool = True,
):
    """Plot compact posterior/trace diagnostics and optional individual slopes."""

    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ArviZ is required; install hsa[bayesian].") from exc

    posterior = idata.posterior
    trace_vars = [
        v
        for v in (
            "alpha",
            "beta",
            "sigma_alpha",
            "sigma_beta",
            "hs_tau",
            "hs_c2",
        )
        if v in posterior
    ]
    trace_plot = az.plot_trace_dist(idata, var_names=trace_vars, combined=False)

    forests: dict[str, Any] = {}
    if "beta_total" in posterior:
        slope_dim = next(
            (d for d in ("random_slope", "predictor") if d in posterior["beta_total"].dims),
            None,
        )
        if slope_dim is not None:
            available = [str(x) for x in posterior["beta_total"].coords[slope_dim].values]
            selected = available if forest_predictors is None else list(forest_predictors)
            missing = sorted(set(selected).difference(available))
            if missing:
                raise ValueError(f"Forest predictors not present in beta_total: {missing}")
            for predictor in selected:
                forests[predictor] = az.plot_forest(
                    idata,
                    var_names=["beta_total"],
                    coords={slope_dim: [predictor]},
                    combined=True,
                    ci_kind="hdi",
                    ci_probs=(0.5, ci_prob),
                )

    if legend and hasattr(trace_plot, "add_legend"):
        # PlotCollection legend behavior differs slightly among ArviZ releases;
        # leave the native legend intact rather than assuming an aesthetic.
        pass

    return {"trace": trace_plot, "individual_forests": forests}


def predict_bayesian_rsf_surface(
    beta_draws: xr.DataArray,
    env: xr.DataArray,
    meta: Mapping,
    *,
    predictors: Sequence[str] | None = None,
    env_predictor_dim: str = "band",
    ci_prob: float = 0.95,
    threshold: float = 1.0,
    n_draws: int | None = 500,
    random_seed: int = 42,
) -> xr.Dataset:
    """Project population posterior RSF coefficients over a raster stack.

    The intercept is intentionally omitted: returned values are relative
    selection scores referenced to an environment at the training means.
    """

    if env_predictor_dim not in env.dims:
        raise ValueError(f"{env_predictor_dim!r} is not a dimension of env.")

    if predictors is None:
        if "predictor" in beta_draws.dims:
            predictors = [str(v) for v in beta_draws["predictor"].values]
        else:
            predictors = list(meta.get("_model", {}).get("predictors", []))
    predictors = list(predictors)
    if not predictors:
        raise ValueError("No predictors supplied for Bayesian surface prediction.")

    env_values = [str(v) for v in env[env_predictor_dim].values]
    missing = sorted(set(predictors).difference(env_values))
    if missing:
        raise ValueError(f"Predictors missing from environmental stack: {missing}")

    beta = beta_draws.sel(predictor=predictors)
    if "sample" not in beta.dims:
        if {"chain", "draw"}.issubset(beta.dims):
            beta = beta.stack(sample=("chain", "draw")).transpose("sample", "predictor")
        else:
            raise ValueError("beta_draws must contain 'sample' or ('chain', 'draw') dimensions.")

    if n_draws is not None and beta.sizes["sample"] > n_draws:
        rng = np.random.default_rng(random_seed)
        idx = np.sort(rng.choice(beta.sizes["sample"], size=n_draws, replace=False))
        beta = beta.isel(sample=idx)

    X = (
        env.sel({env_predictor_dim: predictors})
        .rename({env_predictor_dim: "predictor"})
        .astype("float32")
    )
    standardized = []
    for predictor in predictors:
        if predictor not in meta:
            raise KeyError(f"No training metadata stored for predictor {predictor!r}.")
        mean = float(meta[predictor]["mean"])
        sd = float(meta[predictor]["sd"])
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid training standard deviation for {predictor!r}: {sd}")
        standardized.append((X.sel(predictor=predictor) - mean) / sd)

    X_std = xr.concat(standardized, dim=pd.Index(predictors, name="predictor"))
    eta = xr.dot(beta, X_std, dim="predictor")
    w = np.exp(eta)

    alpha = (1.0 - ci_prob) / 2.0
    quantiles = w.quantile([alpha, 0.5, 1.0 - alpha], dim="sample")

    ds = xr.Dataset(
        {
            "mean": w.mean("sample"),
            "median": quantiles.sel(quantile=0.5, drop=True),
            "sd": w.std("sample"),
            "lower": quantiles.sel(quantile=alpha, drop=True),
            "upper": quantiles.sel(quantile=1.0 - alpha, drop=True),
            "exceedance": (w > threshold).mean("sample"),
        }
    )
    ds["interval_width"] = ds["upper"] - ds["lower"]
    ds.attrs.update(
        {
            "ci_prob": ci_prob,
            "threshold": threshold,
            "n_posterior_draws": int(beta.sizes["sample"]),
            "interpretation": "relative selection score; intercept omitted",
        }
    )

    try:
        if env.rio.crs is not None:
            ds = ds.rio.write_crs(env.rio.crs)
    except Exception:
        pass

    return ds


__all__ = [
    "prepare_bayesian_rsf_data",
    "build_bayesian_rsf_model",
    "evaluate_bayesian_rsf",
    "plot_bayesian_rsf_diagnostics",
    "predict_bayesian_rsf_surface",
]
