"""Frequentist conditional-logit step-selection workflow."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from threadpoolctl import threadpool_limits

from hsa.ssf.base import SSFAnalysis, SSFFit
from hsa.ssf.data import (
    apply_ssf_scaling,
    build_ssf_choice_arrays,
    complete_ssf_strata,
    fit_ssf_scaling,
    score_choice_probabilities,
    stable_softmax,
)
from hsa.ssf.diagnostics import (
    canonical_ciif,
    conditional_information,
    merge_opportunity_diagnostics,
    plot_selection_opportunity,
    summarize_selection_opportunity,
)
from hsa.ssf.fast_conditional import fit_fast_conditional_ssf


@dataclass
class FrequentistSSFFit(SSFFit):
    """Fitted pooled conditional-logit SSF."""

    model: Any
    result: Any
    data: pd.DataFrame
    raw_predictors: tuple[str, ...]
    predictors: tuple[str, ...]
    scaling: dict[str, dict[str, float]]
    id_col: str
    stratum_col: str = "stratum_id"
    engine: str = "fast"

    def summary(self, *args, **kwargs):
        return self.result.summary(*args, **kwargs)

    def coefficients(self, *, alpha: float = 0.05) -> pd.DataFrame:
        """Return standardized conditional-selection coefficients."""
        ci = pd.DataFrame(self.result.conf_int(alpha=alpha))
        rows = []
        for predictor in self.predictors:
            beta = float(self.result.params[predictor])
            rows.append(
                {
                    "predictor": predictor,
                    "beta": beta,
                    "se": float(self.result.bse[predictor]),
                    "lower": float(ci.loc[predictor].iloc[0]),
                    "upper": float(ci.loc[predictor].iloc[1]),
                    "p": float(self.result.pvalues[predictor]),
                    "relative_selection": float(np.exp(beta)),
                }
            )
        return pd.DataFrame(rows)

    def choice_probabilities(self, df: pd.DataFrame | None = None) -> np.ndarray:
        """Return fitted conditional probabilities in dataframe row order.

        Canonical SSF tables are evaluated as one dense tensor, avoiding a
        Python-level groupby loop over tens of thousands of strata.
        """
        d = self.data if df is None else df
        required = [
            self.id_col,
            self.stratum_col,
            "candidate_id",
            *self.predictors,
        ]
        missing = [column for column in required if column not in d]
        if missing:
            raise KeyError(f"Missing choice-probability columns: {missing}")

        work = d[required].copy()
        work["_row_order"] = np.arange(len(work), dtype=np.int64)
        work = work.sort_values(
            [self.id_col, self.stratum_col, "candidate_id"]
        )
        sizes = work.groupby(
            [self.id_col, self.stratum_col],
            sort=False,
        ).size().to_numpy()
        if sizes.size == 0 or not np.all(sizes == sizes[0]):
            raise ValueError(
                "Vectorized SSF probabilities require constant choice-set size."
            )

        n_choices = int(sizes[0])
        n_strata = len(sizes)
        X = (
            work[list(self.predictors)]
            .to_numpy(dtype=np.float64)
            .reshape(n_strata, n_choices, len(self.predictors))
        )
        beta = self.result.params.to_numpy(dtype=np.float64)
        eta = np.einsum("sjp,p->sj", X, beta, optimize=True)
        probability_sorted = stable_softmax(eta, axis=1).reshape(-1)

        probability = np.empty(len(work), dtype=np.float64)
        probability[work["_row_order"].to_numpy(dtype=np.int64)] = probability_sorted
        return probability

    def choice_scores(self, df: pd.DataFrame | None = None):
        """Score actual choices using probability, rank and log-score gain."""
        d = self.data if df is None else df
        arrays = build_ssf_choice_arrays(
            d,
            id_col=self.id_col,
            predictors=self.predictors,
            stratum_col=self.stratum_col,
        )
        beta = self.result.params.to_numpy(dtype=np.float64)
        eta = np.einsum(
            "sjp,p->sj",
            np.asarray(arrays.X, dtype=np.float64),
            beta,
            optimize=True,
        )
        probabilities = stable_softmax(eta, axis=1)
        per_stratum, summary = score_choice_probabilities(
            probabilities,
            arrays.chosen,
        )
        per_stratum = pd.concat(
            [
                arrays.strata[[self.id_col, self.stratum_col]].reset_index(drop=True),
                per_stratum.reset_index(drop=True),
            ],
            axis=1,
        )
        return {
            "per_stratum": per_stratum,
            "summary": summary,
        }

    def ciif(self) -> pd.DataFrame:
        """Compute canonical CIIF from the conditional-logit information matrix."""
        return canonical_ciif(
            self.model,
            self.result.params,
            predictor_names=self.predictors,
        )

    def selection_opportunity(self) -> pd.DataFrame:
        """Summarize conditional information using this fit as the reference beta."""
        information = conditional_information(
            self.data,
            self.result.params.to_numpy(dtype=float),
            predictors=self.predictors,
            id_col=self.id_col,
            stratum_col=self.stratum_col,
        )
        return summarize_selection_opportunity(
            information,
            id_col=self.id_col,
        )


@dataclass
class IndividualSSFFits:
    """No-pooling individual fits plus shared-reference opportunity diagnostics."""

    summary: pd.DataFrame
    fits: dict[Any, FrequentistSSFFit]
    opportunity: pd.DataFrame
    diagnostic: pd.DataFrame
    id_col: str
    reference_beta: pd.Series | None = None

    def plot_selection_opportunity(self, predictor: str, **kwargs):
        """Plot effect magnitude and precision against selection opportunity."""
        return plot_selection_opportunity(
            self.diagnostic,
            predictor,
            id_col=self.id_col,
            **kwargs,
        )


def fit_conditional_ssf(
    df: pd.DataFrame,
    *,
    predictors: list[str] | tuple[str, ...],
    id_col: str,
    stratum_col: str = "stratum_id",
    candidate_col: str = "candidate_id",
    used_col: str = "used",
    engine: str = "fast",
    method: str = "bfgs",
    maxiter: int = 300,
    disp: bool = False,
    offset_col: str | None = None
):
    """Fit conditional logistic regression to an already scaled SSF table.

    Parameters
    ----------
    engine:
        ``"fast"`` (default) uses the exact one-choice-per-stratum softmax
        likelihood evaluated over a dense NumPy tensor. ``"statsmodels"`` keeps
        the general-purpose :class:`statsmodels` implementation as a reference
        engine and compatibility fallback.
    """
    engine = engine.lower()
    if engine == "fast":
        return fit_fast_conditional_ssf(
            df,
            predictors=predictors,
            id_col=id_col,
            stratum_col=stratum_col,
            candidate_col=candidate_col,
            used_col=used_col,
            method=method,
            maxiter=maxiter,
            disp=disp,
            offset_col=offset_col
        )
    if offset_col is not None:
        raise NotImplementedError(
            "Offsets are currently implemented only "
            "for engine='fast'."
        )
    if engine != "statsmodels":
        raise ValueError("engine must be 'fast' or 'statsmodels'.")

    groups = pd.factorize(
        pd.MultiIndex.from_frame(df[[id_col, stratum_col]])
    )[0]
    X = df[list(predictors)].astype(float)
    y = df[used_col].astype(int)
    model = ConditionalLogit(
        endog=y,
        exog=X,
        groups=groups,
    )
    result = model.fit(
        method=method,
        maxiter=maxiter,
        disp=disp,
    )
    return model, result


def _fit_individual_group(
    task: tuple[
        Any,
        pd.DataFrame,
        tuple[str, ...],
        str,
        str,
        str,
        int,
        str | None,
        int,
    ],
):
    """Fit one individual in an isolated native-thread budget.

    This helper is module-level so it is safely picklable by
    :class:`ProcessPoolExecutor`.  Native BLAS/OpenMP pools are explicitly
    bounded inside each worker to avoid nested oversubscription.
    """
    (
        animal_id,
        group,
        predictors,
        id_col,
        stratum_col,
        engine,
        maxiter,
        offset_col,
        native_threads,
    ) = task
    with threadpool_limits(limits=native_threads):
        model, result = fit_conditional_ssf(
            group,
            predictors=predictors,
            id_col=id_col,
            stratum_col=stratum_col,
            engine=engine,
            method="lbfgs",
            maxiter=maxiter,
            offset_col=offset_col,
        )
    return animal_id, model, result


def _parallel_individual_results(
    groups: list[tuple[Any, pd.DataFrame]],
    *,
    predictors: tuple[str, ...],
    id_col: str,
    stratum_col: str,
    engine: str,
    method: str,
    maxiter: int,
    offset_col: str | None,
    workers: int,
    executor: Literal["process", "thread"],
    native_threads: int,
):
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if native_threads <= 0:
        raise ValueError("native_threads must be positive.")
    if executor not in {"process", "thread"}:
        raise ValueError("executor must be 'process' or 'thread'.")
    if workers > 1 and engine != "fast":
        raise ValueError(
            "Parallel individual fitting currently supports engine='fast' only."
        )

    # Keep the historical serial path exact, including arbitrary Statsmodels
    # optimizer names.  Parallel execution is deliberately narrow and measured.
    if workers == 1:
        out = []
        with threadpool_limits(limits=native_threads):
            for animal_id, group in groups:
                model, result = fit_conditional_ssf(
                    group,
                    predictors=predictors,
                    id_col=id_col,
                    stratum_col=stratum_col,
                    engine=engine,
                    method=method,
                    maxiter=maxiter,
                    offset_col=offset_col,
                )
                out.append((animal_id, model, result))
        return out

    # The worker helper currently standardizes the parallel production path on
    # L-BFGS, the empirically preferred large-SSF optimizer.  Rejecting another
    # method is safer than silently changing optimizer semantics.
    method_key = method.lower().replace("_", "-")
    if method_key not in {"lbfgs", "l-bfgs", "l-bfgs-b"}:
        raise ValueError(
            "Parallel individual fitting currently requires method='lbfgs'."
        )

    bounded_workers = min(int(workers), len(groups))
    tasks = [
        (
            animal_id,
            group,
            predictors,
            id_col,
            stratum_col,
            engine,
            int(maxiter),
            offset_col,
            int(native_threads),
        )
        for animal_id, group in groups
    ]
    pool_class = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    with pool_class(max_workers=bounded_workers) as pool:
        return list(pool.map(_fit_individual_group, tasks))


def fit_ssf_per_id(
    df: pd.DataFrame,
    *,
    id_col: str,
    predictors: list[str] | tuple[str, ...],
    raw_predictors: list[str] | tuple[str, ...] | None = None,
    scaling: dict[str, dict[str, float]] | None = None,
    reference_beta: pd.Series | np.ndarray | None = None,
    stratum_col: str = "stratum_id",
    engine: str = "fast",
    method: str = "bfgs",
    maxiter: int = 300,
    offset_col: str | None = None,
    workers: int = 1,
    executor: Literal["process", "thread"] = "process",
    native_threads: int = 1,
) -> IndividualSSFFits:
    """Fit independent individual SSFs on one common predictor scale.

    Selection opportunity is evaluated with one common ``reference_beta`` when
    supplied. This keeps cross-individual information differences focused on
    the alternatives encountered rather than allowing each animal's estimated
    effect size to change the probability weights used in the diagnostic.

    ``workers=1`` preserves the historical serial execution path.  Set
    ``workers>1`` with ``executor='process'`` for bounded per-individual
    parallelism.  Each worker constrains native BLAS/OpenMP execution to
    ``native_threads`` (default 1), preventing nested oversubscription.  Do not
    nest this local process pool inside an outer Dask/Slurm model-distribution
    layer; use ``workers=1`` there and distribute individuals at the outer layer.
    """
    predictors = tuple(predictors)
    raw_predictors = tuple(raw_predictors or predictors)
    groups = [
        (animal_id, group.copy())
        for animal_id, group in df.groupby(id_col, sort=False)
    ]
    if not groups:
        raise ValueError("No individuals are available for SSF fitting.")

    fitted = _parallel_individual_results(
        groups,
        predictors=predictors,
        id_col=id_col,
        stratum_col=stratum_col,
        engine=engine,
        method=method,
        maxiter=maxiter,
        offset_col=offset_col,
        workers=workers,
        executor=executor,
        native_threads=native_threads,
    )
    group_lookup = dict(groups)
    fits: dict[Any, FrequentistSSFFit] = {}
    coefficient_parts = []
    ciif_by_id = {}

    for animal_id, model, result in fitted:
        group = group_lookup[animal_id]
        fit = FrequentistSSFFit(
            model=model,
            result=result,
            data=group,
            raw_predictors=raw_predictors,
            predictors=predictors,
            scaling={} if scaling is None else scaling,
            id_col=id_col,
            stratum_col=stratum_col,
            engine=engine,
        )
        fits[animal_id] = fit
        coefficients = fit.coefficients()
        coefficients[id_col] = animal_id
        coefficients["n_strata"] = int(group[stratum_col].nunique())
        coefficient_parts.append(coefficients)
        ciif_by_id[animal_id] = fit.ciif()

    summary = pd.concat(coefficient_parts, ignore_index=True)

    if reference_beta is None:
        reference_model, reference_result = fit_conditional_ssf(
            df,
            predictors=predictors,
            id_col=id_col,
            stratum_col=stratum_col,
            engine=engine,
            method=method,
            maxiter=maxiter,
            offset_col=offset_col,
        )
        del reference_model
        reference_beta = reference_result.params

    reference_beta = pd.Series(
        np.asarray(reference_beta, dtype=float),
        index=predictors,
        name="reference_beta",
    )
    information = conditional_information(
        df,
        reference_beta.to_numpy(dtype=float),
        predictors=predictors,
        id_col=id_col,
        stratum_col=stratum_col,
    )
    opportunity = summarize_selection_opportunity(
        information,
        id_col=id_col,
    )
    diagnostic = merge_opportunity_diagnostics(
        summary,
        opportunity,
        id_col=id_col,
        ciif_by_id=ciif_by_id,
    )
    return IndividualSSFFits(
        summary=summary,
        fits=fits,
        opportunity=opportunity,
        diagnostic=diagnostic,
        id_col=id_col,
        reference_beta=reference_beta,
    )


class FrequentistSSF(SSFAnalysis):
    """Pooled conditional-logit SSF with movement-informed availability."""

    def _scaled_complete_data(self):
        self.validate_predictors()
        complete = complete_ssf_strata(
            self.choices,
            predictors=self.predictors,
            id_col=self.id_col,
            expected_n_choices=self.n_available + 1,
        )
        scaling = fit_ssf_scaling(complete, self.predictors)
        scaled = apply_ssf_scaling(
            complete,
            self.predictors,
            scaling,
        )
        model_predictors = tuple(f"{predictor}_z" for predictor in self.predictors)
        return scaled, scaling, model_predictors

    def fit(
        self,
        *,
        engine: str = "fast",
        method: str = "bfgs",
        maxiter: int = 300,
        disp: bool = False,
    ) -> FrequentistSSFFit:
        """Fit the pooled conditional-logit reference model.

        ``engine='fast'`` is the default and is exactly equivalent to the
        conditional-logit likelihood for canonical SSF strata with one chosen
        endpoint. Use ``engine='statsmodels'`` to reproduce the previous general
        implementation for validation or benchmarking.
        """
        scaled, scaling, model_predictors = self._scaled_complete_data()
        model, result = fit_conditional_ssf(
            scaled,
            predictors=model_predictors,
            id_col=self.id_col,
            engine=engine,
            method=method,
            maxiter=maxiter,
            disp=disp,
        )
        fit = FrequentistSSFFit(
            model=model,
            result=result,
            data=scaled,
            raw_predictors=tuple(self.predictors),
            predictors=model_predictors,
            scaling=scaling,
            id_col=self.id_col,
            engine=engine,
        )
        self.fit_ = fit
        return fit

    def fit_individuals(
        self,
        *,
        engine: str = "fast",
        method: str = "bfgs",
        maxiter: int = 300,
        workers: int = 1,
        executor: Literal["process", "thread"] = "process",
        native_threads: int = 1,
    ) -> IndividualSSFFits:
        """Fit no-pooling individual models using one scale and reference beta.

        Use ``workers>1, executor='process', method='lbfgs'`` for the bounded
        process-parallel production path established by the inference benchmark.
        ``workers=1`` remains fully backward compatible.
        """
        scaled, scaling, model_predictors = self._scaled_complete_data()
        _, pooled = fit_conditional_ssf(
            scaled,
            predictors=model_predictors,
            id_col=self.id_col,
            engine=engine,
            method=method,
            maxiter=maxiter,
        )
        return fit_ssf_per_id(
            scaled,
            id_col=self.id_col,
            predictors=model_predictors,
            raw_predictors=self.predictors,
            scaling=scaling,
            reference_beta=pooled.params,
            engine=engine,
            method=method,
            maxiter=maxiter,
            workers=workers,
            executor=executor,
            native_threads=native_threads,
        )


__all__ = [
    "FrequentistSSF",
    "FrequentistSSFFit",
    "IndividualSSFFits",
    "fit_conditional_ssf",
    "fit_ssf_per_id",
]
