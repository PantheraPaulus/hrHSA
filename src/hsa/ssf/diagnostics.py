"""Information, collinearity and selection-opportunity diagnostics for SSFs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def within_stratum_correlation(
    df: pd.DataFrame,
    predictors: Sequence[str],
    *,
    id_col: str | None = None,
    stratum_col: str = "stratum_id",
) -> pd.DataFrame:
    """Correlate predictor contrasts after removing each stratum mean."""
    group_cols = [stratum_col] if id_col is None else [id_col, stratum_col]
    centered = pd.DataFrame(index=df.index)
    for predictor in predictors:
        centered[predictor] = (
            df[predictor]
            - df.groupby(group_cols, sort=False)[predictor].transform("mean")
        )
    return centered.corr()


def conditional_information(
    df: pd.DataFrame,
    beta,
    *,
    predictors: Sequence[str],
    id_col: str,
    stratum_col: str = "stratum_id",
) -> pd.DataFrame:
    """Compute per-stratum diagonal Fisher information for each predictor.

    The implementation is vectorized over candidate rows. For predictor ``k``
    and stratum ``s``, the returned quantity is
    ``Var_p(x_sjk)`` under the conditional probabilities implied by ``beta``.
    Supplying one common reference beta makes cross-individual opportunity
    comparisons reflect environmental contrasts rather than coefficient shifts.
    """
    predictors = tuple(predictors)
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (len(predictors),):
        raise ValueError("beta must contain one coefficient per predictor.")

    required = [id_col, stratum_col, *predictors]
    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"Missing information columns: {missing}")

    X = df[list(predictors)].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Conditional-information predictors must be finite.")

    eta = X @ beta
    keys = df[[id_col, stratum_col]].copy()
    work = keys.copy()
    work["_eta"] = eta
    eta_max = work.groupby(
        [id_col, stratum_col],
        sort=False,
    )["_eta"].transform("max").to_numpy(dtype=float)
    weights = np.exp(eta - eta_max)
    work["_weight"] = weights
    denominator = work.groupby(
        [id_col, stratum_col],
        sort=False,
    )["_weight"].transform("sum").to_numpy(dtype=float)
    probability = weights / denominator

    moment_columns = {}
    for j, predictor in enumerate(predictors):
        values = X[:, j]
        moment_columns[f"{predictor}__m1"] = probability * values
        moment_columns[f"{predictor}__m2"] = probability * values * values

    moments = pd.concat(
        [
            keys.reset_index(drop=True),
            pd.DataFrame(moment_columns),
        ],
        axis=1,
    )
    aggregated = moments.groupby(
        [id_col, stratum_col],
        sort=False,
        as_index=False,
    ).sum()

    frames = []
    for predictor in predictors:
        first = aggregated[f"{predictor}__m1"].to_numpy(dtype=float)
        second = aggregated[f"{predictor}__m2"].to_numpy(dtype=float)
        information = np.maximum(second - first * first, 0.0)
        frames.append(
            pd.DataFrame(
                {
                    id_col: aggregated[id_col].to_numpy(),
                    stratum_col: aggregated[stratum_col].to_numpy(),
                    "predictor": predictor,
                    "information": information,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarize_selection_opportunity(
    information: pd.DataFrame,
    *,
    id_col: str,
) -> pd.DataFrame:
    """Summarize conditional information by individual and predictor."""
    required = {id_col, "predictor", "information"}
    missing = required.difference(information.columns)
    if missing:
        raise KeyError(f"Missing information columns: {sorted(missing)}")
    return (
        information.groupby([id_col, "predictor"], sort=False)["information"]
        .agg(
            total_information="sum",
            median_information="median",
            mean_information="mean",
        )
        .reset_index()
    )


def canonical_ciif(
    model,
    params,
    *,
    predictor_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute the Conditional Information Inflation Factor from the Hessian.

    ``CIIF_k = I_kk * [I^{-1}]_kk`` quantifies coefficient-variance inflation
    relative to the diagonal-information limit. ``sqrt_ciif`` is the associated
    standard-error inflation factor. This uses the ordinary likelihood
    information matrix intentionally; temporal/bootstrap uncertainty belongs to
    a separate inferential diagnostic.
    """
    params_array = np.asarray(params, dtype=float)
    information = -np.asarray(model.hessian(params_array), dtype=float)
    if information.ndim != 2 or information.shape[0] != information.shape[1]:
        raise ValueError("Model Hessian must be square.")
    if information.shape[0] != params_array.size:
        raise ValueError("Model Hessian dimension does not match params.")

    covariance = np.linalg.pinv(information)
    ciif = np.diag(information) * np.diag(covariance)
    if predictor_names is None:
        predictor_names = [f"x{i}" for i in range(len(ciif))]
    if len(predictor_names) != len(ciif):
        raise ValueError(
            "predictor_names length does not match the information matrix."
        )

    return pd.DataFrame(
        {
            "predictor": list(predictor_names),
            "information_diagonal": np.diag(information),
            "variance_information_inverse": np.diag(covariance),
            "ciif": ciif,
            "sqrt_ciif": np.sqrt(np.maximum(ciif, 0.0)),
        }
    )


def merge_opportunity_diagnostics(
    estimates: pd.DataFrame,
    opportunity: pd.DataFrame,
    *,
    id_col: str,
    ciif_by_id: Mapping[Any, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Combine individual estimates, opportunity and optional canonical CIIF."""
    out = estimates.merge(
        opportunity,
        on=[id_col, "predictor"],
        how="left",
        validate="one_to_one",
    )
    if ciif_by_id:
        frames = []
        for animal_id, table in ciif_by_id.items():
            part = table[["predictor", "ciif", "sqrt_ciif"]].copy()
            part[id_col] = animal_id
            frames.append(part)
        ciif = pd.concat(frames, ignore_index=True)
        out = out.merge(
            ciif,
            on=[id_col, "predictor"],
            how="left",
            validate="one_to_one",
        )
    return out


def plot_selection_opportunity(
    diagnostic: pd.DataFrame,
    predictor: str,
    *,
    id_col: str = "individual-local-identifier",
    predictor_col: str = "predictor",
    beta_col: str = "beta",
    se_col: str = "se",
    lower_col: str = "lower",
    upper_col: str = "upper",
    n_col: str = "n_strata",
    information_col: str = "mean_information",
    label: str | None = None,
    label_map: Mapping[Any, str] | None = None,
    annotate: bool = True,
    show_ciif: bool = True,
    ciif_decimals: int = 2,
    figsize: tuple[float, float] = (12, 5),
    reference: bool = True,
):
    """Plot effect magnitude and precision against environmental opportunity.

    Panel A asks whether estimated effect magnitude changes with environmental
    contrast. Panel B removes the first-order effect of sample size by plotting
    ``SE(beta) * sqrt(n_strata)`` against mean conditional information. Canonical
    CIIF values, when present, describe additional precision loss from shared
    predictor information.
    """
    import matplotlib.pyplot as plt

    d = diagnostic.loc[diagnostic[predictor_col] == predictor].copy()
    if d.empty:
        raise ValueError(
            f"Predictor {predictor!r} not found in diagnostic table."
        )

    required = [
        id_col,
        beta_col,
        se_col,
        lower_col,
        upper_col,
        n_col,
        information_col,
    ]
    if show_ciif and "ciif" not in d:
        show_ciif = False

    d = (
        d.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
    )
    d = d.loc[
        (d[information_col] > 0)
        & (d[n_col] > 0)
    ].copy()
    if d.empty:
        raise ValueError("No positive-information rows remain for plotting.")

    d["precision_adjusted"] = (
        d[se_col] * np.sqrt(d[n_col])
    )
    d["information_reference"] = (
        1.0 / np.sqrt(d[information_col])
    )
    d = d.sort_values(information_col).reset_index(drop=True)

    if label is None:
        label = (
            predictor.removesuffix("_z")
            .replace("_", " ")
            .title()
        )

    def display_id(value):
        if label_map is not None:
            return label_map.get(value, str(value))
        text = str(value)
        return text.split("_", 1)[1] if "_" in text else text

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        constrained_layout=True,
    )
    ax_effect, ax_precision = axes

    x = d[information_col].to_numpy(dtype=float)
    beta = d[beta_col].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            beta - d[lower_col].to_numpy(dtype=float),
            d[upper_col].to_numpy(dtype=float) - beta,
        ]
    )
    ax_effect.errorbar(
        x,
        beta,
        yerr=yerr,
        fmt="o",
        capsize=4,
    )
    ax_effect.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )
    ax_effect.set_xlabel(
        "Mean conditional information per stratum"
    )
    ax_effect.set_ylabel(
        f"{label} selection coefficient"
    )
    ax_effect.set_title(
        "A. Selection strength versus opportunity"
    )

    precision = d["precision_adjusted"].to_numpy(dtype=float)
    ax_precision.scatter(x, precision)
    if reference:
        xmin, xmax = x.min(), x.max()
        margin = (
            0.05 * (xmax - xmin)
            if xmax > xmin
            else max(0.05 * xmin, 1e-6)
        )
        x_ref = np.linspace(
            max(np.finfo(float).eps, xmin - margin),
            xmax + margin,
            300,
        )
        ax_precision.plot(
            x_ref,
            1.0 / np.sqrt(x_ref),
            linestyle="--",
            label=r"$1/\sqrt{\bar{I}}$ reference",
        )
        ax_precision.legend()

    ax_precision.set_xlabel(
        "Mean conditional information per stratum"
    )
    ax_precision.set_ylabel(
        r"$SE(\hat{\beta})\sqrt{n_{\mathrm{strata}}}$"
    )
    ax_precision.set_title(
        "B. Environmental opportunity versus precision"
    )

    if annotate:
        offsets = [
            (5, 5),
            (5, -12),
            (-5, 7),
            (-5, -12),
            (7, 0),
        ]
        for i, (_, row) in enumerate(d.iterrows()):
            offset = offsets[i % len(offsets)]
            text = display_id(row[id_col])
            ax_effect.annotate(
                text,
                (row[information_col], row[beta_col]),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
            )

            precision_text = text
            if show_ciif and pd.notna(row.get("ciif")):
                precision_text += (
                    f"\nCIIF = {row['ciif']:.{ciif_decimals}f}"
                )
            ax_precision.annotate(
                precision_text,
                (row[information_col], row["precision_adjusted"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
            )

    if show_ciif:
        ax_precision.text(
            0.02,
            0.03,
            (
                r"$\mathrm{CIIF}=I_{kk}[I^{-1}]_{kk}$"
                "\n"
                r"$\sqrt{\mathrm{CIIF}}$: SE inflation from shared information"
            ),
            transform=ax_precision.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        f"{label}: selection opportunity diagnostic"
    )
    return fig, axes, d


__all__ = [
    "within_stratum_correlation",
    "conditional_information",
    "summarize_selection_opportunity",
    "canonical_ciif",
    "merge_opportunity_diagnostics",
    "plot_selection_opportunity",
]
