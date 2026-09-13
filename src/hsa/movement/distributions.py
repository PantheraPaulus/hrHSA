from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import i0


def _aic(nll: float, k: int) -> float:
    return 2 * k + 2 * nll


def _fit_model(nll_func, data: np.ndarray, *, init: list[float], bounds, name: str) -> dict:
    res = minimize(nll_func, init, args=(data,), bounds=bounds)
    if not res.success:
        return {"distribution": name, "params": None, "nll": np.inf, "AIC": np.inf, "success": False}
    return {
        "distribution": name,
        "params": tuple(float(v) for v in res.x),
        "nll": float(res.fun),
        "AIC": _aic(float(res.fun), len(res.x)),
        "success": True,
    }


def _exp_nll(params, x):
    (scale,) = params
    return -np.sum(-np.log(scale) - x / scale)


def _gamma_nll(params, x):
    shape, scale = params
    from scipy.stats import gamma

    return -np.sum(gamma.logpdf(x, a=shape, scale=scale))


def _weibull_nll(params, x):
    shape, scale = params
    from scipy.stats import weibull_min

    return -np.sum(weibull_min.logpdf(x, c=shape, scale=scale))


def _lognorm_nll(params, x):
    sigma, scale = params
    from scipy.stats import lognorm

    return -np.sum(lognorm.logpdf(x, s=sigma, scale=scale))


def fit_step_distribution(
    steps,
    *,
    candidates: tuple[str, ...] = ("exp", "gamma", "weibull", "lognorm"),
    cutoff: float = np.inf,
) -> dict:
    """Fit candidate step-length distributions by maximum likelihood."""

    values = pd.Series(steps).dropna().astype(float)
    values = values[(values > 0) & (values <= cutoff)].to_numpy()
    if values.size == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

    fits = []
    if "exp" in candidates:
        fits.append(_fit_model(_exp_nll, values, init=[np.mean(values)], bounds=[(1e-6, None)], name="exp"))
    if "gamma" in candidates:
        fits.append(_fit_model(_gamma_nll, values, init=[1.0, np.mean(values)], bounds=[(1e-6, None), (1e-6, None)], name="gamma"))
    if "weibull" in candidates:
        fits.append(_fit_model(_weibull_nll, values, init=[1.0, np.mean(values)], bounds=[(1e-6, None), (1e-6, None)], name="weibull"))
    if "lognorm" in candidates:
        fits.append(_fit_model(_lognorm_nll, values, init=[1.0, np.mean(values)], bounds=[(1e-6, None), (1e-6, None)], name="lognorm"))

    table = pd.DataFrame(fits).sort_values("AIC").reset_index(drop=True)
    winner = table.iloc[0]
    return {
        "n": int(values.size),
        "q25": float(np.nanquantile(values, 0.25)),
        "median": float(np.nanmedian(values)),
        "mean": float(np.nanmean(values)),
        "q75": float(np.nanquantile(values, 0.75)),
        "max": float(np.nanmax(values)),
        "distribution": winner["distribution"],
        "params": winner["params"],
        "model_table": table,
    }


def _vonmises_nll(params, angles):
    kappa, mu = params
    log_density = kappa * np.cos(angles - mu) - np.log(2 * np.pi * i0(kappa))
    return -np.sum(log_density)


def _vm_uniform_nll(params, angles):
    kappa, weight = params
    vm = np.exp(kappa * np.cos(angles)) / (2 * np.pi * i0(kappa))
    uni = np.full_like(angles, 1.0 / (2 * np.pi), dtype=float)
    density = weight * vm + (1 - weight) * uni
    return -np.sum(np.log(density + 1e-12))


def fit_turn_angle_distribution(
    angles,
    *,
    candidates: tuple[str, ...] = ("vonmises", "vm_uniform"),
) -> dict:
    """Fit candidate turn-angle distributions by maximum likelihood."""

    values = pd.Series(angles).dropna().astype(float).to_numpy()
    if values.size == 0:
        raise ValueError("No turning angles available after filtering.")

    rows = []
    if "vonmises" in candidates:
        res = minimize(_vonmises_nll, x0=[1.0, 0.0], args=(values,), bounds=[(1e-6, None), (-np.pi, np.pi)])
        rows.append({"distribution": "vonmises", "params": tuple(res.x), "nll": float(res.fun), "AIC": _aic(float(res.fun), 2), "success": bool(res.success)})
    if "vm_uniform" in candidates:
        res = minimize(_vm_uniform_nll, x0=[0.5, 0.5], args=(values,), bounds=[(1e-6, None), (0, 1)])
        rows.append({"distribution": "vm_uniform", "params": tuple(res.x), "nll": float(res.fun), "AIC": _aic(float(res.fun), 2), "success": bool(res.success)})

    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    winner = table.iloc[0]
    return {
        "n": int(values.size),
        "distribution": winner["distribution"],
        "params": winner["params"],
        "model_table": table,
    }
