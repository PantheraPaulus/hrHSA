"""Bounded outer parallelism for independent SSF/iSSF model fits.

The statistical kernels themselves are intentionally kept single-threaded by
native BLAS/OpenMP libraries.  Parallelism here is across independent animals,
which is the scaling axis supported by the workstation inference benchmarks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

import pandas as pd

from hsa.ssf.frequentist import _parallel_individual_results


def fit_issf_per_id(
    analysis,
    *,
    scaling: Mapping[str, Mapping[str, float]] | None = None,
    center_offset: bool = True,
    engine: str = "fast",
    method: str = "lbfgs",
    maxiter: int = 1000,
    workers: int = 1,
    executor: Literal["process", "thread"] = "process",
    native_threads: int = 1,
):
    """Fit independent iSSFs with bounded outer parallelism.

    ``analysis`` may be either the public iSSF facade or the legacy
    :class:`hsa.ssf.issf.FrequentistISSF`; both expose ``prepare_design`` and
    ``id_col``.  The returned object is the normal ``IndividualISSFFits`` type,
    including the individual fitted model objects.

    Parameters
    ----------
    workers:
        Number of concurrent individual fits.  ``1`` preserves serial
        execution.  Workstation benchmarking supports process-level outer
        parallelism with about 6--8 workers for sufficiently large data.
    executor:
        ``"process"`` (recommended for measured large workloads) or
        ``"thread"``.  Parallel execution is currently restricted to the fast
        engine and L-BFGS.
    native_threads:
        Native BLAS/OpenMP threads allowed inside each individual fit.  Keep
        this at ``1`` when running multiple fits concurrently.

    Notes
    -----
    Do not nest this local executor inside an outer Dask/Slurm layer that is
    already distributing independent fits.  In that situation use
    ``workers=1`` and let the outer scheduler provide model-level parallelism.
    """
    # Local imports avoid introducing a circular import during hsa.ssf package
    # initialization.
    from hsa.ssf.issf import FrequentistISSFFit, IndividualISSFFits

    design = analysis.prepare_design(
        scaling=scaling,
        center_offset=center_offset,
    )
    id_col = analysis.id_col
    groups = [
        (individual, group.copy())
        for individual, group in design.data.groupby(id_col, sort=False)
    ]
    if not groups:
        raise ValueError("No individuals are available for iSSF fitting.")

    fitted = _parallel_individual_results(
        groups,
        predictors=tuple(design.predictors),
        id_col=id_col,
        stratum_col=design.stratum_col,
        engine=engine,
        method=method,
        maxiter=maxiter,
        offset_col=design.offset_col,
        workers=workers,
        executor=executor,
        native_threads=native_threads,
    )
    group_lookup = dict(groups)
    fits: dict[Any, FrequentistISSFFit] = {}
    frames = []

    for individual, model, result in fitted:
        group = group_lookup[individual]
        individual_design = replace(
            design,
            data=group,
            diagnostics={
                **design.diagnostics,
                "n_individuals": 1,
                "n_strata_retained": int(
                    group[design.stratum_col].nunique()
                ),
                "n_choice_rows": int(len(group)),
            },
        )
        fit = FrequentistISSFFit(
            model=model,
            result=result,
            design=individual_design,
            id_col=id_col,
        )
        fits[individual] = fit
        coefficients = fit.coefficients()
        coefficients[id_col] = individual
        coefficients["n_strata"] = int(
            group[design.stratum_col].nunique()
        )
        frames.append(coefficients)

    return IndividualISSFFits(
        summary=pd.concat(frames, ignore_index=True),
        fits=fits,
        design=design,
        id_col=id_col,
    )


__all__ = ["fit_issf_per_id"]
