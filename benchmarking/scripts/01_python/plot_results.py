"""Aggregate hrHSA HPC JSONL benchmark records and create scaling plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsa.compute import read_benchmark_records


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = [read_benchmark_records(path) for path in paths]
    if not frames:
        raise ValueError("No benchmark files supplied.")
    return pd.concat(frames, ignore_index=True)


def _campaign_column(df: pd.DataFrame) -> pd.Series:
    explicit = df.get("metadata.campaign")
    if explicit is None:
        explicit = pd.Series([None] * len(df), index=df.index)
    partition = df.get("metadata.slurm_job_partition")
    if partition is None:
        partition = pd.Series([None] * len(df), index=df.index)
    backend = df.get("metadata.backend")
    if backend is None:
        backend = pd.Series([None] * len(df), index=df.index)
    return explicit.fillna(partition).fillna(backend).fillna("unknown").astype(str)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["campaign"] = _campaign_column(work)
    grouped = (
        work.groupby(["benchmark", "campaign", "workers"], dropna=False)
        .agg(
            repeats=("wall_seconds", "count"),
            median_seconds=("wall_seconds", "median"),
            q25_seconds=("wall_seconds", lambda x: x.quantile(0.25)),
            q75_seconds=("wall_seconds", lambda x: x.quantile(0.75)),
            median_throughput_rows_s=("throughput_rows_s", "median"),
            median_throughput_mb_s=("throughput_mb_s", "median"),
            median_tasks_per_worker=("tasks_per_worker", "median"),
            max_worker_peak_rss_mb=("worker_peak_rss_max_mb", "max"),
            total_worker_peak_rss_mb=("worker_peak_rss_total_mb", "max"),
        )
        .reset_index()
        .sort_values(["benchmark", "campaign", "workers"])
    )

    pieces = []
    for (_, _), group in grouped.groupby(["benchmark", "campaign"], sort=False):
        group = group.copy().sort_values("workers")
        baseline_workers = float(group.iloc[0]["workers"])
        baseline_time = float(group.iloc[0]["median_seconds"])
        group["speedup"] = baseline_time / group["median_seconds"]
        group["parallel_efficiency"] = group["speedup"] / (
            group["workers"] / baseline_workers
        )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _runtime_plot(summary: pd.DataFrame, benchmark: str, output: Path) -> None:
    subset = summary[summary["benchmark"] == benchmark]
    fig, ax = plt.subplots(figsize=(8, 5))
    for campaign, group in subset.groupby("campaign"):
        group = group.sort_values("workers")
        lower = group["median_seconds"] - group["q25_seconds"]
        upper = group["q75_seconds"] - group["median_seconds"]
        ax.errorbar(
            group["workers"],
            group["median_seconds"],
            yerr=np.vstack([lower, upper]),
            marker="o",
            capsize=4,
            label=campaign,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Dask workers")
    ax.set_ylabel("Median wall time (s)")
    ax.set_title(f"{benchmark}: strong-scaling wall time")
    ax.grid(True, alpha=0.25)
    ax.legend(title="campaign / partition")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _speedup_plot(summary: pd.DataFrame, benchmark: str, output: Path) -> None:
    subset = summary[summary["benchmark"] == benchmark]
    fig, ax = plt.subplots(figsize=(8, 5))
    for campaign, group in subset.groupby("campaign"):
        group = group.sort_values("workers")
        relative_workers = group["workers"] / float(group.iloc[0]["workers"])
        ax.plot(group["workers"], group["speedup"], marker="o", label=campaign)
        ax.plot(
            group["workers"],
            relative_workers,
            linestyle="--",
            alpha=0.5,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Dask workers")
    ax.set_ylabel("Speedup relative to campaign baseline")
    ax.set_title(f"{benchmark}: strong-scaling speedup")
    ax.grid(True, alpha=0.25)
    ax.legend(title="campaign / partition")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _efficiency_plot(summary: pd.DataFrame, benchmark: str, output: Path) -> None:
    subset = summary[summary["benchmark"] == benchmark]
    fig, ax = plt.subplots(figsize=(8, 5))
    for campaign, group in subset.groupby("campaign"):
        group = group.sort_values("workers")
        ax.plot(
            group["workers"],
            100.0 * group["parallel_efficiency"],
            marker="o",
            label=campaign,
        )
    ax.axhline(100.0, linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Dask workers")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title(f"{benchmark}: parallel efficiency")
    ax.grid(True, alpha=0.25)
    ax.legend(title="campaign / partition")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_many(args.inputs)
    summary = _summarize(raw)
    raw.to_csv(args.output_dir / "benchmark_raw.csv", index=False)
    summary.to_csv(args.output_dir / "benchmark_summary.csv", index=False)

    for benchmark in summary["benchmark"].dropna().unique():
        safe = str(benchmark).replace("/", "_").replace(" ", "_")
        _runtime_plot(summary, benchmark, args.output_dir / f"{safe}_runtime.png")
        _speedup_plot(summary, benchmark, args.output_dir / f"{safe}_speedup.png")
        _efficiency_plot(summary, benchmark, args.output_dir / f"{safe}_efficiency.png")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
