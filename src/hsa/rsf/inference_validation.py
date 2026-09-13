"""Backend-agnostic posterior agreement helpers for inference benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def posterior_parameter_summary(
    idata: Any,
    *,
    exclude: Iterable[str] = ("eta",),
) -> pd.DataFrame:
    """Summarize every posterior parameter element across chain/draw dimensions.

    The output deliberately flattens vector/matrix parameters into stable term
    names so summaries from different NUTS backends can be joined directly.
    Observation-level ``eta`` is excluded by default because it is a storage
    choice rather than a model parameter and can dominate output size.
    """

    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("InferenceData has no posterior group")

    excluded = set(exclude)
    rows: list[dict[str, Any]] = []
    for name, value in posterior.data_vars.items():
        if str(name) in excluded:
            continue
        sample_dims = [dim for dim in ("chain", "draw") if dim in value.dims]
        if not sample_dims:
            continue
        mean = value.mean(dim=sample_dims)
        sd = value.std(dim=sample_dims)
        mean_values = np.asarray(mean.values)
        sd_values = np.asarray(sd.values)

        if mean_values.ndim == 0:
            rows.append(
                {
                    "term": str(name),
                    "mean": float(mean_values),
                    "sd": float(sd_values),
                }
            )
            continue

        for index in np.ndindex(mean_values.shape):
            suffix = ",".join(str(i) for i in index)
            rows.append(
                {
                    "term": f"{name}[{suffix}]",
                    "mean": float(mean_values[index]),
                    "sd": float(sd_values[index]),
                }
            )

    return pd.DataFrame(rows, columns=["term", "mean", "sd"]).sort_values("term").reset_index(drop=True)


def compare_posterior_summaries(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, float | int]:
    """Compare backend posterior summaries on common parameter elements."""

    merged = reference.merge(candidate, on="term", suffixes=("_ref", "_candidate"))
    if merged.empty:
        return {
            "n_common_terms": 0,
            "max_abs_mean_diff": np.nan,
            "median_abs_mean_diff": np.nan,
            "rmse_mean_diff": np.nan,
            "max_posterior_scale_mean_diff": np.nan,
            "median_relative_sd_diff": np.nan,
        }

    mean_diff = merged["mean_candidate"] - merged["mean_ref"]
    abs_mean_diff = mean_diff.abs()
    pooled_scale = np.sqrt(
        np.square(merged["sd_ref"].to_numpy(dtype=float))
        + np.square(merged["sd_candidate"].to_numpy(dtype=float))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = abs_mean_diff.to_numpy(dtype=float) / pooled_scale
        relative_sd = np.abs(
            merged["sd_candidate"].to_numpy(dtype=float)
            - merged["sd_ref"].to_numpy(dtype=float)
        ) / np.maximum(np.abs(merged["sd_ref"].to_numpy(dtype=float)), np.finfo(float).eps)

    return {
        "n_common_terms": int(len(merged)),
        "max_abs_mean_diff": float(abs_mean_diff.max()),
        "median_abs_mean_diff": float(abs_mean_diff.median()),
        "rmse_mean_diff": float(np.sqrt(np.mean(np.square(mean_diff.to_numpy(dtype=float))))),
        "max_posterior_scale_mean_diff": float(np.nanmax(scaled)),
        "median_relative_sd_diff": float(np.nanmedian(relative_sd)),
    }


__all__ = ["posterior_parameter_summary", "compare_posterior_summaries"]
