"""Posterior diagnostic and coefficient plots for Bayesian SSFs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _arviz():
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ArviZ is required; install hsa[bayesian].") from exc
    return az


def _validate_predictors(idata, predictors: Sequence[str] | None) -> list[str]:
    posterior = idata.posterior
    if "predictor" not in posterior.coords:
        raise ValueError("Posterior does not contain a 'predictor' coordinate.")
    available = [str(value) for value in posterior.coords["predictor"].values]
    selected = available if predictors is None else [str(value) for value in predictors]
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(
            "Forest predictors are not present in the Bayesian SSF posterior: "
            f"{missing}. Available predictors: {available}"
        )
    if not selected:
        raise ValueError("At least one predictor is required for a forest plot.")
    return selected


def plot_bayesian_ssf_trace(
    idata,
    *,
    include_individual: bool = False,
    individuals: Sequence[str] | None = None,
):
    """Plot posterior distributions and chain traces for SSF parameters.

    Population coefficients and between-individual standard deviations are shown
    by default. Individual partially pooled coefficients can be added explicitly;
    this is disabled by default because it can create a very large diagnostic
    grid when many animals are fitted.
    """
    az = _arviz()
    posterior = idata.posterior
    var_names = [name for name in ("mu_beta", "sigma_beta") if name in posterior]
    coords: dict[str, list[str]] = {}

    if include_individual:
        if "beta" not in posterior:
            raise ValueError("Posterior does not contain individual coefficients 'beta'.")
        var_names.append("beta")
        if individuals is not None:
            if "individual" not in posterior["beta"].coords:
                raise ValueError("Posterior beta does not contain an 'individual' coordinate.")
            available = [str(value) for value in posterior["beta"].coords["individual"].values]
            selected = [str(value) for value in individuals]
            missing = sorted(set(selected).difference(available))
            if missing:
                raise ValueError(
                    f"Trace individuals are not present in posterior beta: {missing}"
                )
            coords["individual"] = selected

    if not var_names:
        raise ValueError("No Bayesian SSF parameters are available for trace diagnostics.")

    kwargs: dict[str, Any] = {
        "var_names": var_names,
        "combined": False,
    }
    if coords:
        kwargs["coords"] = coords
    return az.plot_trace_dist(idata, **kwargs)


def plot_bayesian_ssf_forests(
    idata,
    *,
    predictors: Sequence[str] | None = None,
    ci_prob: float = 0.95,
    include_population: bool = True,
    include_individual: bool = True,
    include_heterogeneity: bool = True,
) -> dict[str, Any]:
    """Plot population, individual, and heterogeneity posterior forests.

    ``mu_beta`` is the population-average selection coefficient, ``beta`` is the
    partially pooled individual coefficient, and ``sigma_beta`` is the posterior
    between-individual standard deviation. The latter is a heterogeneity measure,
    not a signed selection effect, and is therefore returned separately.
    """
    if not 0 < ci_prob < 1:
        raise ValueError("ci_prob must be in (0, 1).")
    az = _arviz()
    posterior = idata.posterior
    selected = _validate_predictors(idata, predictors)

    out: dict[str, Any] = {
        "population": None,
        "heterogeneity": None,
        "individual": {},
    }

    forest_kwargs = {
        "combined": True,
        "ci_kind": "hdi",
        "ci_probs": (0.5, float(ci_prob)),
    }

    if include_population:
        if "mu_beta" not in posterior:
            raise ValueError("Posterior does not contain population coefficients 'mu_beta'.")
        out["population"] = az.plot_forest(
            idata,
            var_names=["mu_beta"],
            coords={"predictor": selected},
            **forest_kwargs,
        )

    if include_heterogeneity:
        if "sigma_beta" not in posterior:
            raise ValueError("Posterior does not contain heterogeneity 'sigma_beta'.")
        out["heterogeneity"] = az.plot_forest(
            idata,
            var_names=["sigma_beta"],
            coords={"predictor": selected},
            **forest_kwargs,
        )

    if include_individual:
        if "beta" not in posterior:
            raise ValueError("Posterior does not contain individual coefficients 'beta'.")
        for predictor in selected:
            out["individual"][predictor] = az.plot_forest(
                idata,
                var_names=["beta"],
                coords={"predictor": [predictor]},
                **forest_kwargs,
            )

    return out


def plot_bayesian_ssf_diagnostics(
    idata,
    *,
    forest_predictors: Sequence[str] | None = None,
    ci_prob: float = 0.95,
    include_population_forest: bool = True,
    include_heterogeneity_forest: bool = True,
    include_individual_forests: bool = True,
    include_individual_trace: bool = False,
    trace_individuals: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Plot compact trace diagnostics and SSF coefficient forests.

    This mirrors the Bayesian RSF diagnostic workflow while respecting the SSF
    hierarchy: population means, between-individual SDs, and partially pooled
    individual slopes are kept visually distinct.
    """
    trace = plot_bayesian_ssf_trace(
        idata,
        include_individual=include_individual_trace,
        individuals=trace_individuals,
    )
    forests = plot_bayesian_ssf_forests(
        idata,
        predictors=forest_predictors,
        ci_prob=ci_prob,
        include_population=include_population_forest,
        include_individual=include_individual_forests,
        include_heterogeneity=include_heterogeneity_forest,
    )
    return {
        "trace": trace,
        "population_forest": forests["population"],
        "heterogeneity_forest": forests["heterogeneity"],
        "individual_forests": forests["individual"],
    }


__all__ = [
    "plot_bayesian_ssf_trace",
    "plot_bayesian_ssf_forests",
    "plot_bayesian_ssf_diagnostics",
]
