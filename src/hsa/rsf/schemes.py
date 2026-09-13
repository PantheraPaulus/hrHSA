"""Composable validation strategies and typed result containers for RSFs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LeaveOneIndividualOut:
    """Outer leave-one-individual-out validation strategy.

    The strategy owns only estimator-independent CV settings. Bayesian- or
    frequentist-specific controls are passed to ``analysis.validate(..., **kwargs)``.
    """

    heldout: Any = "all"
    thin_train_dt: str | None = None
    thin_test_dt: str | None = None
    sampling_factor_train: int = 10
    n_background: int = 100_000
    n_bins: int = 20
    seed: int = 42

    def __post_init__(self):
        if self.sampling_factor_train <= 0:
            raise ValueError("sampling_factor_train must be positive.")
        if self.n_background <= 0:
            raise ValueError("n_background must be positive.")
        if self.n_bins < 3:
            raise ValueError("n_bins must be at least 3.")

    def run(self, analysis, **kwargs):
        """Dispatch the shared validation design to the estimator."""
        return analysis._run_loio(self, **kwargs)


@dataclass(frozen=True)
class BlockedBootstrap:
    """Finite validation-sample uncertainty using whole temporal blocks."""

    replicates: int = 2000
    block: str = "7D"

    def __post_init__(self):
        if self.replicates <= 0:
            raise ValueError("replicates must be positive.")
        if not self.block:
            raise ValueError("block must be a non-empty pandas duration string.")


@dataclass(frozen=True)
class ContiguousTemporalBlocks:
    """Temporal non-stationarity diagnostic using consecutive calendar units."""

    folds: int = 10
    unit: str = "W"

    def __post_init__(self):
        if self.folds < 2:
            raise ValueError("folds must be at least 2.")
        if not self.unit:
            raise ValueError("unit must be a non-empty pandas calendar unit.")


def _parse_uncertainty_strategies(strategies):
    """Validate post-fit strategies and translate them to kernel arguments."""
    if not strategies:
        strategies = (
            BlockedBootstrap(),
            ContiguousTemporalBlocks(),
        )

    bootstrap = None
    temporal = None

    for strategy in strategies:
        if isinstance(strategy, BlockedBootstrap):
            if bootstrap is not None:
                raise ValueError(
                    "Only one BlockedBootstrap strategy may be supplied."
                )
            bootstrap = strategy
        elif isinstance(strategy, ContiguousTemporalBlocks):
            if temporal is not None:
                raise ValueError(
                    "Only one ContiguousTemporalBlocks strategy may be supplied."
                )
            temporal = strategy
        else:
            raise TypeError(
                "Post-fit uncertainty strategies must be "
                "BlockedBootstrap or ContiguousTemporalBlocks."
            )

    methods: list[str] = []
    kwargs: dict[str, Any] = {}

    if bootstrap is not None:
        methods.append("bootstrap")
        kwargs.update(
            bootstrap_replicates=bootstrap.replicates,
            bootstrap_block=bootstrap.block,
        )

    if temporal is not None:
        methods.append("temporal_contiguous")
        kwargs.update(
            temporal_folds=temporal.folds,
            temporal_unit=temporal.unit,
        )

    return tuple(strategies), methods, kwargs


@dataclass
class CrossValidationResult:
    """Common state returned by frequentist and Bayesian validation workflows."""

    summary: pd.DataFrame
    params: pd.DataFrame
    boyce_bins: pd.DataFrame
    diagnostics: dict
    scheme: LeaveOneIndividualOut
    analysis: Any = None

    def __len__(self) -> int:
        return len(self.summary)

    def __repr__(self) -> str:
        estimator = (
            self.analysis.__class__.__name__
            if self.analysis is not None
            else "RSF"
        )
        return (
            f"{self.__class__.__name__}("
            f"estimator={estimator!r}, folds={len(self.summary)})"
        )


@dataclass
class ValidationUncertaintyResult:
    """Common wrapper for post-fit validation uncertainty outputs."""

    validation: dict[str, Any]
    source: CrossValidationResult
    strategies: tuple[Any, ...]

    @property
    def baseline_summary(self) -> pd.DataFrame:
        return self.validation["baseline_summary"]

    @property
    def baseline_curves(self) -> pd.DataFrame:
        return self.validation["baseline_curves"]

    @property
    def replicate_summary(self) -> pd.DataFrame:
        return self.validation["replicate_summary"]

    @property
    def method_summary(self) -> pd.DataFrame:
        return self.validation["method_summary"]

    @property
    def curves(self) -> pd.DataFrame:
        return self.validation["curves"]

    @property
    def curve_summary(self) -> pd.DataFrame:
        return self.validation["curve_summary"]

    @property
    def raw(self) -> dict:
        return self.validation["raw"]

    @property
    def config(self) -> dict:
        return self.validation["config"]

    def temporal_periods(self, heldout_id=None) -> pd.DataFrame:
        """Return the exact contiguous periods evaluated during validation."""
        d = self.replicate_summary
        d = d.loc[d["method"] == "temporal_contiguous"].copy()

        if heldout_id is not None:
            d = d.loc[d["heldout_ID"] == heldout_id].copy()

        columns = [
            col
            for col in (
                "heldout_ID",
                "replicate",
                "replicate_label",
                "heldout_units",
                "n_temporal_units",
                "n_used",
                "start",
                "end",
                "boyce",
                "boyce_lower",
                "boyce_upper",
            )
            if col in d.columns
        ]
        return d[columns].reset_index(drop=True)


@dataclass
class FrequentistLOIOResult(CrossValidationResult):
    """LOIO result for a frequentist RSF."""

    calibration_bins: pd.DataFrame | None = None

    def plot_boyce_curves(
        self,
        *,
        x_col: str = "q_mid",
        show_points: bool = True,
        show_reference: bool = True,
        ax=None,
        figsize: tuple[float, float] = (7, 5),
        title: str = "Boyce validation curves",
        alpha: float = 0.65,
        legend: bool | str = "auto",
    ):
        """Plot P/E curves across held-out individuals."""
        from hsa.rsf.validation import plot_boyce_curves

        return plot_boyce_curves(
            self.boyce_bins,
            x_col=x_col,
            show_points=show_points,
            show_reference=show_reference,
            ax=ax,
            figsize=figsize,
            title=title,
            alpha=alpha,
            legend=legend,
        )

    def plot_boyce_values(
        self,
        *,
        sort: bool = True,
        ax=None,
        figsize: tuple[float, float] = (9, 4),
        title: str = "Boyce validation by held-out individual",
    ):
        """Plot scalar Boyce indices across held-out individuals."""
        from hsa.rsf.validation import plot_boyce_values

        return plot_boyce_values(
            self.summary,
            sort=sort,
            ax=ax,
            figsize=figsize,
            title=title,
        )

    def evaluate_uncertainty(
        self,
        *strategies,
        n_bins: int | None = None,
        n_background: int | None = None,
        ci_prob: float = 0.95,
        seed: int | None = None,
    ) -> "FrequentistValidationUncertaintyResult":
        """Evaluate validation-sample uncertainty and temporal non-stationarity.

        The fitted fold-specific RSF is held fixed. Blocked bootstrap intervals
        therefore quantify finite validation-sample uncertainty only; contiguous
        temporal blocks are an empirical temporal-stability diagnostic.
        """
        from hsa.rsf.frequentist_validation import (
            evaluate_frequentist_loio_uncertainty,
        )

        strategies, methods, kwargs = _parse_uncertainty_strategies(strategies)

        validation = evaluate_frequentist_loio_uncertainty(
            self.diagnostics,
            methods=methods,
            n_bins=self.scheme.n_bins if n_bins is None else n_bins,
            n_background=(
                self.scheme.n_background
                if n_background is None
                else n_background
            ),
            ci_prob=ci_prob,
            seed=self.scheme.seed if seed is None else seed,
            **kwargs,
        )

        return FrequentistValidationUncertaintyResult(
            validation=validation,
            source=self,
            strategies=strategies,
        )


@dataclass
class BayesianLOIOResult(CrossValidationResult):
    """LOIO result for a hierarchical Bayesian RSF.

    Stored posterior score matrices make post-fit uncertainty experiments cheap:
    neither the environmental raster nor the PyMC model has to be recomputed.
    """

    def _fold_diagnostic(self, heldout_id) -> dict:
        if heldout_id not in self.diagnostics:
            raise KeyError(f"Unknown held-out individual: {heldout_id!r}")
        diagnostic = self.diagnostics[heldout_id]
        if "error" in diagnostic:
            raise ValueError(
                f"LOIO fold {heldout_id!r} failed: {diagnostic['error']}"
            )
        return diagnostic

    def posterior_correlation(self, heldout_id) -> pd.DataFrame:
        """Return posterior population-beta correlations for one LOIO fold."""
        from hsa.rsf.bayesian_shrinkage import posterior_beta_correlation

        diagnostic = self._fold_diagnostic(heldout_id)
        return posterior_beta_correlation(
            diagnostic["idata"],
            predictors=(
                self.analysis.predictors
                if self.analysis is not None
                else None
            ),
        )

    def collinearity_diagnostics(
        self,
        heldout_id,
        *,
        min_abs_predictor_corr: float = 0.5,
    ) -> pd.DataFrame:
        """Compare design and posterior coefficient correlation in one fold."""
        from hsa.rsf.bayesian_shrinkage import collinearity_diagnostics

        diagnostic = self._fold_diagnostic(heldout_id)
        if "bayes_data" not in diagnostic:
            raise KeyError(
                f"LOIO diagnostic {heldout_id!r} does not contain 'bayes_data'."
            )
        return collinearity_diagnostics(
            diagnostic["idata"],
            diagnostic["bayes_data"],
            predictors=(
                self.analysis.predictors
                if self.analysis is not None
                else None
            ),
            min_abs_predictor_corr=min_abs_predictor_corr,
        )

    def shrinkage_summary(
        self,
        heldout_id,
        *,
        ci_prob: float = 0.95,
    ) -> pd.DataFrame:
        """Summarize regularized-horseshoe shrinkage for one LOIO fold."""
        from hsa.rsf.bayesian_shrinkage import regularized_horseshoe_summary

        diagnostic = self._fold_diagnostic(heldout_id)
        return regularized_horseshoe_summary(
            diagnostic["idata"],
            ci_prob=ci_prob,
        )

    def evaluate_uncertainty(
        self,
        *strategies,
        n_bins: int | None = None,
        ci_prob: float = 0.95,
        seed: int | None = None,
    ) -> "BayesianValidationUncertaintyResult":
        """Evaluate sampling uncertainty and/or temporal non-stationarity."""
        from hsa.rsf.bayesian_validation import (
            evaluate_bayesian_loio_uncertainty,
        )

        strategies, methods, kwargs = _parse_uncertainty_strategies(strategies)

        validation = evaluate_bayesian_loio_uncertainty(
            self.diagnostics,
            methods=methods,
            n_bins=self.scheme.n_bins if n_bins is None else n_bins,
            ci_prob=ci_prob,
            seed=self.scheme.seed if seed is None else seed,
            **kwargs,
        )

        return BayesianValidationUncertaintyResult(
            validation=validation,
            source=self,
            strategies=strategies,
        )


@dataclass
class FrequentistValidationUncertaintyResult(ValidationUncertaintyResult):
    """Post-fit validation uncertainty for a frequentist LOIO result."""

    def plot(self, heldout_id, *, figsize=(12, 5)):
        """Plot bootstrap P/E uncertainty and contiguous temporal curves."""
        from hsa.rsf.frequentist_validation import (
            plot_frequentist_loio_uncertainty,
        )

        return plot_frequentist_loio_uncertainty(
            self.validation,
            heldout_id,
            figsize=figsize,
        )


@dataclass
class BayesianValidationUncertaintyResult(ValidationUncertaintyResult):
    """Post-fit validation uncertainty for a Bayesian LOIO result."""

    def plot(self, heldout_id, *, figsize=(12, 5)):
        """Plot bootstrap P/E uncertainty and contiguous temporal curves."""
        from hsa.rsf.bayesian_validation import plot_bayesian_loio_uncertainty

        return plot_bayesian_loio_uncertainty(
            self.validation,
            heldout_id,
            figsize=figsize,
        )


__all__ = [
    "LeaveOneIndividualOut",
    "BlockedBootstrap",
    "ContiguousTemporalBlocks",
    "CrossValidationResult",
    "ValidationUncertaintyResult",
    "FrequentistLOIOResult",
    "BayesianLOIOResult",
    "FrequentistValidationUncertaintyResult",
    "BayesianValidationUncertaintyResult",
]
