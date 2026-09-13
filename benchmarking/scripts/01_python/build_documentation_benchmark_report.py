"""Build documentation-ready benchmark tables and figures.

The raw benchmark campaign deliberately records more information than should be
shown on a documentation landing page. This module reduces a workstation reference
run to four user-facing questions:

1. How long does point sampling take for a stated number of points?
2. How long does raster surface prediction take for a stated number of cells?
3. How does either raster operation scale with physical CPU concurrency?
4. How long does a stated MCMC problem take, and how many *effective* posterior
   samples are obtained per second?

The report never treats MCMC runtime as linearly extrapolatable from raw rows or
nominal draws. Raster throughput is reported in physical problem units (points/s
and cells/s); MCMC throughput is quality-adjusted with bulk ESS/s and accompanied
by R-hat/divergence guardrails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsa.compute import read_benchmark_records


POINT_BENCHMARK = "workstation_zarr_point_sampling"
SURFACE_BENCHMARK = "workstation_zarr_surface_prediction"
BENCHMARK_LABELS = {
    POINT_BENCHMARK: "Point sampling",
    SURFACE_BENCHMARK: "Surface prediction",
}


def _q25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def _q75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _raster_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw raster records by operation and worker geometry."""
    if frame.empty:
        return pd.DataFrame()

    work = frame.copy()
    if "metadata.worker_geometry" in work:
        work["geometry"] = work["metadata.worker_geometry"].astype(str)
    else:
        work["geometry"] = (
            work["workers"].astype(int).astype(str)
            + "x"
            + work["threads_per_worker"].astype(int).astype(str)
        )
    work["total_threads"] = (
        work["workers"].astype(int) * work["threads_per_worker"].astype(int)
    )

    aggregations: dict[str, tuple[str, Any]] = {
        "repeats": ("wall_seconds", "count"),
        "median_seconds": ("wall_seconds", "median"),
        "q25_seconds": ("wall_seconds", _q25),
        "q75_seconds": ("wall_seconds", _q75),
        "median_units_s": ("throughput_rows_s", "median"),
        "rows": ("rows", "median"),
        "bytes_processed": ("bytes_processed", "median"),
    }

    memory_column = _first_existing(
        work,
        (
            "worker_operation_peak_rss_total_mb",
            "operation_peak_process_tree_rss_mb",
            "operation_peak_rss_mb",
        ),
    )
    if memory_column is not None:
        aggregations["median_peak_memory_mb"] = (memory_column, "median")

    if "metadata.bands" in work:
        aggregations["bands"] = ("metadata.bands", "median")
    if "metadata.raster_size_x" in work:
        aggregations["raster_size_x"] = ("metadata.raster_size_x", "median")
    if "metadata.raster_size_y" in work:
        aggregations["raster_size_y"] = ("metadata.raster_size_y", "median")

    grouped = (
        work.groupby(
            ["benchmark", "geometry", "workers", "threads_per_worker", "total_threads"],
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
    )

    grouped["throughput_million_units_s"] = grouped["median_units_s"] / 1e6
    grouped["core_seconds"] = grouped["median_seconds"] * grouped["total_threads"]

    baseline = (
        grouped.sort_values(["benchmark", "total_threads", "median_seconds"])
        .groupby("benchmark", as_index=False)
        .first()[["benchmark", "median_seconds"]]
        .rename(columns={"median_seconds": "baseline_seconds"})
    )
    grouped = grouped.merge(baseline, on="benchmark", how="left")
    grouped["speedup_vs_smallest"] = (
        grouped["baseline_seconds"] / grouped["median_seconds"]
    )
    smallest_threads = grouped.groupby("benchmark")["total_threads"].transform("min")
    grouped["parallel_efficiency"] = grouped["speedup_vs_smallest"] / (
        grouped["total_threads"] / smallest_threads
    )
    grouped["relative_to_fastest"] = grouped["median_seconds"] / grouped.groupby(
        "benchmark"
    )["median_seconds"].transform("min")
    return grouped.sort_values(["benchmark", "total_threads", "workers"])


def _mcmc_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.loc[frame["status"].eq("success")].copy()
    if work.empty:
        return pd.DataFrame()

    summary = (
        work.groupby("sampler", dropna=False)
        .agg(
            repeats=("completed_sampling_seconds", "count"),
            median_seconds=("completed_sampling_seconds", "median"),
            q25_seconds=("completed_sampling_seconds", _q25),
            q75_seconds=("completed_sampling_seconds", _q75),
            median_bulk_ess_s=(
                "median_ess_bulk_per_completed_second",
                "median",
            ),
            median_min_bulk_ess_s=(
                "min_ess_bulk_per_completed_second",
                "median",
            ),
            worst_rhat=("max_rhat", "max"),
            max_divergences=("n_divergences", "max"),
            median_peak_process_tree_mb=(
                "operation_peak_process_tree_rss_mb",
                "median",
            ),
            median_posterior_mib=("posterior_mib", "median"),
        )
        .reset_index()
    )
    summary["quality_guardrail_pass"] = (
        summary["worst_rhat"].le(1.01) & summary["max_divergences"].eq(0)
    )
    return summary.sort_values("median_bulk_ess_s", ascending=False)


def _recommended_row(summary: pd.DataFrame, benchmark: str) -> pd.Series:
    """Choose the lowest-concurrency point within 5% of the fastest median."""
    subset = summary.loc[summary["benchmark"].eq(benchmark)].copy()
    if subset.empty:
        raise ValueError(f"No benchmark rows found for {benchmark!r}")
    fastest = float(subset["median_seconds"].min())
    eligible = subset.loc[subset["median_seconds"].le(fastest * 1.05)].copy()
    return eligible.sort_values(
        ["total_threads", "median_seconds", "workers"],
        ascending=[True, True, True],
    ).iloc[0]


def _human_count(value: float) -> str:
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.2f} billion"
    if value >= 1e6:
        return f"{value / 1e6:.2f} million"
    if value >= 1e3:
        return f"{value / 1e3:.1f} thousand"
    return f"{value:.0f}"


def _markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _save_figure(fig, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_raster_scaling(summary: pd.DataFrame, output_dir: Path) -> None:
    for benchmark in (POINT_BENCHMARK, SURFACE_BENCHMARK):
        subset = summary.loc[summary["benchmark"].eq(benchmark)].sort_values(
            "total_threads"
        )
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        lower = subset["median_seconds"] - subset["q25_seconds"]
        upper = subset["q75_seconds"] - subset["median_seconds"]
        ax.errorbar(
            subset["total_threads"],
            subset["median_seconds"],
            yerr=np.vstack([lower, upper]),
            marker="o",
            capsize=4,
        )
        ax.set_xlabel("Physical execution threads")
        ax.set_ylabel("Wall time (s)")
        ax.set_title(f"{BENCHMARK_LABELS[benchmark]} strong scaling")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        stem = (
            "point_sampling_strong_scaling"
            if benchmark == POINT_BENCHMARK
            else "surface_prediction_strong_scaling"
        )
        _save_figure(fig, output_dir / stem)


def _plot_geometry(summary: pd.DataFrame, output_dir: Path) -> None:
    for benchmark in (POINT_BENCHMARK, SURFACE_BENCHMARK):
        subset = summary.loc[summary["benchmark"].eq(benchmark)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["median_seconds", "workers"])
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.barh(subset["geometry"], subset["relative_to_fastest"])
        ax.set_xlabel("Wall time relative to fastest configuration")
        ax.set_ylabel("Workers × threads/worker")
        ax.set_title(f"{BENCHMARK_LABELS[benchmark]} worker geometry")
        ax.axvline(1.05, linestyle="--", linewidth=1)
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        stem = (
            "point_sampling_geometry"
            if benchmark == POINT_BENCHMARK
            else "surface_prediction_geometry"
        )
        _save_figure(fig, output_dir / stem)


def _plot_mcmc(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    ordered = summary.sort_values("median_bulk_ess_s", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(ordered["sampler"], ordered["median_bulk_ess_s"])
    ax.set_xlabel("Median bulk ESS / completed second")
    ax.set_ylabel("NUTS backend")
    ax.set_title("MCMC quality-adjusted throughput")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_dir / "mcmc_ess_per_second")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(ordered["sampler"], ordered["median_seconds"])
    ax.set_xlabel("Completed sampling wall time (s)")
    ax.set_ylabel("NUTS backend")
    ax.set_title("MCMC completed runtime")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_dir / "mcmc_completed_runtime")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(suite_dir: Path, output_dir: Path) -> dict[str, Path]:
    suite_dir = suite_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    figures = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    machine = _load_json(suite_dir / "machine.json")
    raster_dir = suite_dir / "01-raster"
    strong_path = raster_dir / "workstation_strong_scaling.jsonl"

    physical = int(machine.get("physical_cores") or 0)
    geometry_path = raster_dir / f"workstation_geometry_{physical}t.jsonl"
    if not geometry_path.exists():
        candidates = sorted(raster_dir.glob("workstation_geometry_*t.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No workstation geometry benchmark found in {raster_dir}"
            )
        geometry_path = candidates[-1]

    if not strong_path.exists():
        raise FileNotFoundError(strong_path)

    strong_raw = read_benchmark_records(strong_path)
    geometry_raw = read_benchmark_records(geometry_path)
    strong_summary = _raster_summary(strong_raw)
    geometry_summary = _raster_summary(geometry_raw)

    mcmc_path = suite_dir / "02-mcmc" / "sampler_completion_runs.csv"
    if not mcmc_path.exists():
        raise FileNotFoundError(mcmc_path)
    mcmc_raw = pd.read_csv(mcmc_path)
    mcmc_summary = _mcmc_summary(mcmc_raw)

    strong_summary.to_csv(output_dir / "raster_strong_scaling.csv", index=False)
    geometry_summary.to_csv(output_dir / "raster_geometry.csv", index=False)
    mcmc_summary.to_csv(output_dir / "mcmc_summary.csv", index=False)

    _plot_raster_scaling(strong_summary, figures)
    _plot_geometry(geometry_summary, figures)
    _plot_mcmc(mcmc_summary, figures)

    reference_rows: list[dict[str, Any]] = []
    for benchmark in (POINT_BENCHMARK, SURFACE_BENCHMARK):
        selected = _recommended_row(geometry_summary, benchmark)
        units = float(selected["rows"])
        rate = float(selected["median_units_s"])
        if benchmark == POINT_BENCHMARK:
            normalized = 1e6 / rate
            normalized_label = "seconds per 1M points"
        else:
            normalized = 1e8 / rate
            normalized_label = "seconds per 100M cells"
        reference_rows.append(
            {
                "operation": BENCHMARK_LABELS[benchmark],
                "workload_units": units,
                "geometry": selected["geometry"],
                "threads": int(selected["total_threads"]),
                "median_seconds": float(selected["median_seconds"]),
                "q25_seconds": float(selected["q25_seconds"]),
                "q75_seconds": float(selected["q75_seconds"]),
                "throughput_million_units_s": float(
                    selected["throughput_million_units_s"]
                ),
                "normalized_runtime": normalized,
                "normalized_runtime_label": normalized_label,
                "peak_memory_gib": (
                    float(selected["median_peak_memory_mb"]) / 1024
                    if "median_peak_memory_mb" in selected.index
                    and pd.notna(selected["median_peak_memory_mb"])
                    else np.nan
                ),
            }
        )

    mcmc_config = machine.get("mcmc", {})
    draws = int(mcmc_config.get("draws", 0) or 0)
    chains = int(mcmc_config.get("chains", 0) or 0)
    for _, row in mcmc_summary.iterrows():
        reference_rows.append(
            {
                "operation": f"MCMC ({row['sampler']})",
                "workload_units": draws * chains,
                "geometry": f"{chains} chains / {mcmc_config.get('cores', '?')} cores",
                "threads": int(mcmc_config.get("cores", 0) or 0),
                "median_seconds": float(row["median_seconds"]),
                "q25_seconds": float(row["q25_seconds"]),
                "q75_seconds": float(row["q75_seconds"]),
                "throughput_million_units_s": np.nan,
                "normalized_runtime": float(row["median_bulk_ess_s"]),
                "normalized_runtime_label": "median bulk ESS/s",
                "peak_memory_gib": float(row["median_peak_process_tree_mb"]) / 1024,
            }
        )

    reference = pd.DataFrame(reference_rows)
    reference.to_csv(output_dir / "reference_operations.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = reference["operation"].tolist()
    values = reference["median_seconds"].to_numpy(dtype=float)
    ax.barh(labels, values)
    if np.nanmax(values) / max(np.nanmin(values), 1e-12) > 20:
        ax.set_xscale("log")
        ax.set_xlabel("Median wall time (s, log scale)")
    else:
        ax.set_xlabel("Median wall time (s)")
    ax.set_title("Canonical hrHSA workstation workloads")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, figures / "canonical_workload_runtime")

    topology = machine.get("topology", {})
    cpu_model = topology.get("cpu_model") or machine.get("cpu_model") or "unknown"
    physical_cores = machine.get("physical_cores", "unknown")
    logical_cpus = machine.get("logical_cpus", "unknown")
    memory_gib = machine.get("memory_gib", topology.get("memory_total_gib", "unknown"))
    git_sha = machine.get("git", {}).get("commit", "unknown")

    point_ref = reference.loc[reference["operation"].eq("Point sampling")].iloc[0]
    surface_ref = reference.loc[reference["operation"].eq("Surface prediction")].iloc[0]

    ref_table_rows = [
        [
            "Point sampling",
            _human_count(point_ref["workload_units"]) + " points",
            str(point_ref["geometry"]),
            f"{point_ref['median_seconds']:.2f}",
            f"{point_ref['throughput_million_units_s']:.2f} M points/s",
            f"{point_ref['normalized_runtime']:.3f} s / 1M points",
        ],
        [
            "Surface prediction",
            _human_count(surface_ref["workload_units"]) + " cells",
            str(surface_ref["geometry"]),
            f"{surface_ref['median_seconds']:.2f}",
            f"{surface_ref['throughput_million_units_s']:.2f} M cells/s",
            f"{surface_ref['normalized_runtime']:.3f} s / 100M cells",
        ],
    ]
    for _, row in mcmc_summary.iterrows():
        ref_table_rows.append(
            [
                f"MCMC ({row['sampler']})",
                (
                    f"{mcmc_config.get('rows', '?')} raw rows; "
                    f"{mcmc_config.get('chains', '?')} chains × "
                    f"{mcmc_config.get('draws', '?')} draws"
                ),
                f"{mcmc_config.get('cores', '?')} cores",
                f"{row['median_seconds']:.2f}",
                f"{row['median_bulk_ess_s']:.2f} median bulk ESS/s",
                (
                    f"R-hat ≤ {row['worst_rhat']:.4f}; "
                    f"max divergences {int(row['max_divergences'])}"
                ),
            ]
        )

    mcmc_table_rows = [
        [
            str(row["sampler"]),
            f"{row['median_seconds']:.2f}",
            f"{row['median_bulk_ess_s']:.2f}",
            f"{row['median_min_bulk_ess_s']:.2f}",
            f"{row['worst_rhat']:.4f}",
            str(int(row["max_divergences"])),
            "yes" if bool(row["quality_guardrail_pass"]) else "no",
        ]
        for _, row in mcmc_summary.iterrows()
    ]

    markdown = f"""# hrHSA workstation performance reference

This page was generated from the documentation benchmark suite. It reports measured
runtime for explicitly defined workloads rather than claiming a machine-independent
speed. Raster throughput can be used as a first-order planning aid for similar
band counts, chunking and storage. MCMC timings should **not** be extrapolated
linearly from raw rows or nominal draws; posterior geometry and effective sample
size matter.

## Reference machine

- CPU: `{cpu_model}`
- Physical cores: **{physical_cores}**
- Logical CPUs: **{logical_cpus}**
- RAM: **{memory_gib} GiB**
- Git commit: `{git_sha}`
- Benchmark profile: `{machine.get('profile', 'unknown')}`

The raster reference chooses the **lowest physical concurrency within 5% of the
fastest observed median** for each operation. This avoids presenting a tiny wall-time
win bought with substantially more CPU as the default production expectation.
The absolute fastest rows remain in `raster_geometry.csv`.

## Canonical workload runtime

{_markdown_table(
    ["Operation", "Canonical workload", "Configuration", "Median wall (s)", "Throughput", "Planning metric"],
    ref_table_rows,
)}

![Canonical workload runtime](figures/canonical_workload_runtime.png)

## Point sampling

Point throughput is reported in actual sampled points per second. For a similar
raster stack, `seconds per 1M points` is the most intuitive first-order estimate;
random-access storage behavior and the number/type of predictor bands can shift it.

![Point sampling strong scaling](figures/point_sampling_strong_scaling.png)

![Point sampling geometry](figures/point_sampling_geometry.png)

## Surface prediction

Prediction throughput is reported in raster cells per second. `seconds per 100M
cells` is a convenient planning unit, but it is only comparable when the model
structure, predictor count, numeric precision, chunking and storage layout are
similar. It is not a direct measurement of disk bandwidth.

![Surface prediction strong scaling](figures/surface_prediction_strong_scaling.png)

![Surface prediction geometry](figures/surface_prediction_geometry.png)

## MCMC

Completed sampling time includes dispatch plus explicit materialization so deferred
JAX work cannot disappear from the timer. Backend comparisons are quality-adjusted:
median bulk ESS/s is the primary throughput metric, with minimum bulk ESS/s, R-hat
and divergences retained as guardrails.

{_markdown_table(
    ["Backend", "Median completed (s)", "Median bulk ESS/s", "Median min ESS/s", "Worst R-hat", "Max divergences", "Guardrail"],
    mcmc_table_rows,
)}

![MCMC ESS per second](figures/mcmc_ess_per_second.png)

![MCMC completed runtime](figures/mcmc_completed_runtime.png)

## Files for reproducibility

- `reference_operations.csv`: compact documentation-facing reference table.
- `raster_strong_scaling.csv`: medians, IQR, speedup, efficiency and core-seconds.
- `raster_geometry.csv`: fixed-core process/thread geometry comparison.
- `mcmc_summary.csv`: completed runtime, quality-adjusted throughput and memory.
- `figures/*.svg`: vector versions of every documentation figure.
- The parent suite directory retains raw JSONL/CSV records, machine/software
  provenance and exact subprocess commands.
"""
    report_path = output_dir / "documentation_benchmark.md"
    report_path.write_text(markdown, encoding="utf-8")

    return {
        "report": report_path,
        "reference": output_dir / "reference_operations.csv",
        "strong_scaling": output_dir / "raster_strong_scaling.csv",
        "geometry": output_dir / "raster_geometry.csv",
        "mcmc": output_dir / "mcmc_summary.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build documentation tables/figures from a workstation reference suite."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    suite_dir = args.suite_dir.expanduser().resolve()
    output_dir = (
        suite_dir / "report"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    outputs = build_report(suite_dir, output_dir)
    print("Documentation benchmark report built:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
