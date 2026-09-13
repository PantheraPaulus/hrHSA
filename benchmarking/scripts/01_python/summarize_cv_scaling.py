"""Summarize hrHSA outer-fold CV scaling benchmark records."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hsa.compute import read_benchmark_records


def summarize(path: Path) -> pd.DataFrame:
    raw = read_benchmark_records(path)
    if raw.empty:
        return pd.DataFrame()

    required = [
        "benchmark",
        "workers",
        "wall_seconds",
        "metadata.mode",
        "metadata.folds_completed",
        "metadata.folds_per_second",
        "metadata.allocated_core_seconds_per_completed_fold",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"CV benchmark records are missing columns: {missing}")

    variant = pd.Series(index=raw.index, dtype="object")
    if "metadata.nuts_sampler" in raw:
        variant = variant.combine_first(raw["metadata.nuts_sampler"])
    if "metadata.optimizer" in raw:
        variant = variant.combine_first(raw["metadata.optimizer"])
    variant = variant.fillna("default")

    work = raw.assign(variant=variant)
    aggregations: dict[str, tuple[str, object]] = {
        "repeats": ("wall_seconds", "count"),
        "median_wall_seconds": ("wall_seconds", "median"),
        "q25_wall_seconds": ("wall_seconds", lambda x: x.quantile(0.25)),
        "q75_wall_seconds": ("wall_seconds", lambda x: x.quantile(0.75)),
        "median_folds_per_second": ("metadata.folds_per_second", "median"),
        "median_core_seconds_per_fold": (
            "metadata.allocated_core_seconds_per_completed_fold",
            "median",
        ),
        "median_completed_folds": ("metadata.folds_completed", "median"),
        "median_fold_seconds": ("metadata.median_fold_seconds", "median"),
        "median_load_seconds": ("metadata.median_load_seconds", "median"),
    }
    optional = {
        "median_operation_process_tree_rss_mb": "operation_peak_process_tree_rss_mb",
        "median_worker_operation_peak_total_rss_mb": "worker_operation_peak_rss_total_mb",
        "median_worker_operation_peak_max_rss_mb": "worker_operation_peak_rss_max_mb",
        "median_worker_current_rss_total_mb": "worker_current_rss_total_mb",
        "median_worker_current_rss_max_mb": "worker_current_rss_max_mb",
    }
    for output_name, source_name in optional.items():
        if source_name in work:
            aggregations[output_name] = (source_name, "median")

    summary = (
        work.groupby(
            ["benchmark", "metadata.mode", "variant", "workers"],
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
        .rename(columns={"metadata.mode": "mode"})
    )

    if "median_worker_operation_peak_total_rss_mb" in summary:
        gib = summary["median_worker_operation_peak_total_rss_mb"] / 1024.0
        summary["folds_per_second_per_worker_gib"] = np.where(
            gib > 0,
            summary["median_folds_per_second"] / gib,
            np.nan,
        )

    pieces = []
    for _, group in summary.groupby(["benchmark", "mode", "variant"], sort=False):
        group = group.sort_values("workers").copy()
        baseline = group.loc[group["workers"] == 1, "median_wall_seconds"]
        if len(baseline):
            t1 = float(baseline.iloc[0])
            group["speedup_vs_1"] = t1 / group["median_wall_seconds"]
            group["parallel_efficiency_vs_1"] = group["speedup_vs_1"] / group["workers"]
        else:
            group["speedup_vs_1"] = np.nan
            group["parallel_efficiency_vs_1"] = np.nan
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True) if pieces else summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = summarize(args.input.expanduser().resolve())
    if result.empty:
        print("No CV scaling records found.")
        return

    print(result.to_string(index=False))
    output = args.output
    if output is None:
        output = args.input.with_name(f"{args.input.stem}-summary.csv")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
