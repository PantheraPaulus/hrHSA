"""Vectorized conditional-logit engine for fixed-size SSF choice sets.

This module exploits the canonical SSF invariant that every stratum contains
exactly one chosen alternative. Under that invariant the conditional-logit
likelihood is exactly a categorical/softmax likelihood, so likelihood, score,
and Hessian can be evaluated over the complete ``(stratum, choice, predictor)``
tensor without Python-level iteration over strata.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp
from scipy.stats import norm

from hsa.ssf.data import SSFChoiceArrays, build_ssf_choice_arrays


def _softmax(eta: np.ndarray) -> np.ndarray:
    shifted = eta - np.max(eta, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


class FastConditionalLogitModel:
    """Conditional logit specialized to one chosen alternative per stratum."""

    def __init__(self, arrays: SSFChoiceArrays):

        X = np.asarray(
            arrays.X,
            dtype=np.float64,
        )

        chosen = np.asarray(
            arrays.chosen,
            dtype=np.int64,
        )

        if X.ndim != 3:
            raise ValueError(
                "X must have shape "
                "(stratum, choice, predictor)."
            )

        if chosen.shape != (X.shape[0],):
            raise ValueError(
                "chosen must contain one index per stratum."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Fast conditional-logit design "
                "contains non-finite values."
            )

        self.X = X
        self.chosen = chosen
        self.predictor_names = tuple(
            arrays.predictors
        )

        self.n_strata, self.n_choices, self.n_predictors = (
            X.shape
        )

        # --------------------------------------------------------
        # Fixed offset
        # --------------------------------------------------------

        if arrays.offset is None:
            # Keep the common SSF path genuinely offset-free. Allocating and
            # adding a dense zero tensor here costs memory bandwidth on every
            # optimizer evaluation without changing the likelihood.
            self.offset = None
        else:

            offset = np.asarray(
                arrays.offset,
                dtype=np.float64,
            )

            expected = (
                self.n_strata,
                self.n_choices,
            )

            if offset.shape != expected:
                raise ValueError(
                    f"offset must have shape {expected}; "
                    f"got {offset.shape}."
                )

            if not np.isfinite(offset).all():
                raise ValueError(
                    "Fast conditional-logit offset "
                    "contains non-finite values."
                )

            self.offset = offset

        self.offset_name = arrays.offset_name

        self._rows = np.arange(
            self.n_strata
        )

        self._X_chosen = self.X[
            self._rows,
            self.chosen,
        ]
        # Constant across optimizer evaluations. Keeping the reduction here avoids
        # re-summing the chosen design matrix on every score evaluation.
        self._X_chosen_sum = self._X_chosen.sum(axis=0)

    def linear_predictor(self, params) -> np.ndarray:

        beta = np.asarray(
            params,
            dtype=np.float64,
        )

        eta = np.einsum(
            "sjp,p->sj",
            self.X,
            beta,
            optimize=True,
        )
        if self.offset is not None:
            eta = eta + self.offset
        return eta

    def probabilities(self, params) -> np.ndarray:
        """Return conditional probabilities for every stratum and alternative."""
        return _softmax(self.linear_predictor(params))

    def loglike(self, params) -> float:
        """Return the exact conditional log likelihood."""
        eta = self.linear_predictor(params)
        return float(
            eta[self._rows, self.chosen].sum()
            - logsumexp(eta, axis=1).sum()
        )

    def score(self, params) -> np.ndarray:
        """Return the analytic gradient of the conditional log likelihood."""
        probability = self.probabilities(params)
        expected = np.einsum(
            "sj,sjp->sp",
            probability,
            self.X,
            optimize=True,
        )
        return self._X_chosen_sum - expected.sum(axis=0)

    def loglike_and_score(self, params) -> tuple[float, np.ndarray]:
        """Evaluate log likelihood and gradient from one linear-predictor pass.

        SciPy optimizers accepting ``jac=True`` request the objective and gradient
        together. Computing them through :meth:`loglike` and :meth:`score`
        separately would evaluate the full ``X @ beta`` tensor twice. This fused
        kernel shares ``eta`` and its log-normalizer between both quantities.
        """
        eta = self.linear_predictor(params)
        log_normalizer = logsumexp(eta, axis=1)
        loglike = float(
            eta[self._rows, self.chosen].sum()
            - log_normalizer.sum()
        )
        probability = np.exp(eta - log_normalizer[:, None])
        expected = np.einsum(
            "sj,sjp->sp",
            probability,
            self.X,
            optimize=True,
        )
        score = self._X_chosen_sum - expected.sum(axis=0)
        return loglike, score

    def information(self, params) -> np.ndarray:
        """Return the conditional Fisher/observed information matrix."""
        probability = self.probabilities(params)
        mean = np.einsum(
            "sj,sjp->sp",
            probability,
            self.X,
            optimize=True,
        )
        centered = self.X - mean[:, None, :]
        return np.einsum(
            "sj,sjp,sjq->pq",
            probability,
            centered,
            centered,
            optimize=True,
        )

    def hessian(self, params) -> np.ndarray:
        """Return the Hessian of the conditional log likelihood."""
        return -self.information(params)


@dataclass
class FastConditionalLogitResult:
    """Small statsmodels-like result object returned by the vectorized engine."""

    model: FastConditionalLogitModel
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    covariance: pd.DataFrame
    llf: float
    optimize_result: OptimizeResult

    @property
    def converged(self) -> bool:
        return bool(self.optimize_result.success)

    @property
    def mle_retvals(self) -> dict:
        return {
            "converged": self.converged,
            "iterations": getattr(self.optimize_result, "nit", None),
            "message": str(self.optimize_result.message),
        }

    def cov_params(self) -> pd.DataFrame:
        return self.covariance.copy()

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        critical = float(norm.ppf(1.0 - alpha / 2.0))
        lower = self.params - critical * self.bse
        upper = self.params + critical * self.bse
        return pd.DataFrame({0: lower, 1: upper})

    def summary(self, alpha: float = 0.05, *args, **kwargs) -> pd.DataFrame:
        """Return a compact coefficient summary for the fast engine."""
        ci = self.conf_int(alpha=alpha)
        z = self.params / self.bse
        out = pd.DataFrame(
            {
                "coef": self.params,
                "std err": self.bse,
                "z": z,
                "P>|z|": self.pvalues,
                f"[{alpha / 2:.3f}": ci[0],
                f"{1 - alpha / 2:.3f}]": ci[1],
            }
        )
        out.attrs["log_likelihood"] = self.llf
        out.attrs["converged"] = self.converged
        out.attrs["iterations"] = getattr(self.optimize_result, "nit", None)
        return out


def fit_fast_conditional_ssf(
    df: pd.DataFrame,
    *,
    predictors: list[str] | tuple[str, ...],
    id_col: str,
    stratum_col: str = "stratum_id",
    candidate_col: str = "candidate_id",
    used_col: str = "used",
    method: str = "bfgs",
    maxiter: int = 300,
    disp: bool = False,
    gtol: float = 1e-5,
    start_params=None,
    offset_col: str | None = None,
) -> tuple[FastConditionalLogitModel, FastConditionalLogitResult]:
    """Fit a vectorized conditional logit to fixed-size SSF choice sets.

    The estimator is mathematically identical to ordinary conditional logistic
    regression when each stratum has exactly one chosen alternative. The fast
    engine only changes how the likelihood is evaluated.
    """
    predictors = tuple(predictors)
    arrays = build_ssf_choice_arrays(
        df,
        id_col=id_col,
        predictors=predictors,
        stratum_col=stratum_col,
        candidate_col=candidate_col,
        used_col=used_col,
        offset_col=offset_col,
        dtype="float64",
    )
    model = FastConditionalLogitModel(arrays)

    if start_params is None:
        start = np.zeros(model.n_predictors, dtype=np.float64)
    else:
        start = np.asarray(start_params, dtype=np.float64)
        if start.shape != (model.n_predictors,):
            raise ValueError("start_params must contain one value per predictor.")

    method_key = method.lower().replace("_", "-")
    scipy_methods = {
        "bfgs": "BFGS",
        "lbfgs": "L-BFGS-B",
        "l-bfgs": "L-BFGS-B",
        "l-bfgs-b": "L-BFGS-B",
    }
    if method_key not in scipy_methods:
        raise ValueError(
            "Fast SSF engine supports method='bfgs' or 'lbfgs'. "
            "Use engine='statsmodels' for other optimizers."
        )

    def objective(beta):
        loglike, score = model.loglike_and_score(beta)
        return -loglike, -score

    optimization = minimize(
        objective,
        start,
        method=scipy_methods[method_key],
        jac=True,
        options={
            "maxiter": int(maxiter),
            "gtol": float(gtol),
        },
    )
    if not optimization.success:
        warnings.warn(
            "Fast conditional-logit optimization did not report convergence: "
            f"{optimization.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    beta = np.asarray(optimization.x, dtype=np.float64)
    information = model.information(beta)
    covariance_array = np.linalg.pinv(information)
    se = np.sqrt(np.maximum(np.diag(covariance_array), 0.0))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvalue = 2.0 * norm.sf(np.abs(z))

    index = pd.Index(predictors, name="predictor")
    covariance = pd.DataFrame(
        covariance_array,
        index=index,
        columns=index,
    )
    result = FastConditionalLogitResult(
        model=model,
        params=pd.Series(beta, index=index, name="coef"),
        bse=pd.Series(se, index=index, name="std err"),
        pvalues=pd.Series(pvalue, index=index, name="P>|z|"),
        covariance=covariance,
        llf=model.loglike(beta),
        optimize_result=optimization,
    )
    return model, result


__all__ = [
    "FastConditionalLogitModel",
    "FastConditionalLogitResult",
    "fit_fast_conditional_ssf",
]
