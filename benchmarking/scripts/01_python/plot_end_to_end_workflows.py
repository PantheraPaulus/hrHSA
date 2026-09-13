"""Plot the six-panel RSF/SSF/iSSF end-to-end runtime decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGES = (
    "structure",
    "availability",
    "point_partitioning",
    "raster_sampling",
    "design",
    "model_build",
    "inference",
    "diagnostics",
)
ANALYSES = ("rsf", "ssf", "issf")
INFERENCE = ("frequentist", "bayesian")


def _read(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No E2E records in {path}")
    return records


def _median_record(records: list[dict], analysis: str, inference: str) -> dict:
    selected = [
        record
        for record in records
        if record.get("status") == "success"
        and record.get("analysis") == analysis
        and record.get("inference") == inference
    ]
    if not selected:
        raise RuntimeError(f"No successful {analysis}/{inference} records")

    wall = np.asarray(
        [float(record["wall_seconds"]) for record in selected],
        dtype=float,
    )
    stage_values = {
        stage: float(
            np.median(
                [
                    float(record["stage_seconds"].get(stage, 0.0))
                    for record in selected
                ]
            )
        )
        for stage in STAGES
    }
    return {
        "analysis": analysis,
        "inference": inference,
        "repeats": len(selected),
        "wall_seconds_median": float(np.median(wall)),
        "wall_seconds_min": float(np.min(wall)),
        "wall_seconds_max": float(np.max(wall)),
        "wall_seconds_cv": (
            float(np.std(wall) / np.mean(wall)) if np.mean(wall) else None
        ),
        "peak_process_tree_rss_mb_median": float(
            np.median(
                [float(record["peak_process_tree_rss_mb"]) for record in selected]
            )
        ),
        "n_candidate_rows_median": int(
            round(np.median([int(record["n_candidate_rows"]) for record in selected]))
        ),
        "model_predictors_median": int(
            round(np.median([int(record["model_predictors"]) for record in selected]))
        ),
        "stage_seconds_median": stage_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    records = _read(args.input)
    summary = [
        _median_record(records, analysis, inference)
        for inference in INFERENCE
        for analysis in ANALYSES
    ]

    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), squeeze=False)
    cmap = plt.get_cmap("tab10")
    stage_colors = {
        stage: cmap(index % 10)
        for index, stage in enumerate(STAGES)
    }

    lookup = {(row["analysis"], row["inference"]): row for row in summary}
    for row_index, inference in enumerate(INFERENCE):
        for col_index, analysis in enumerate(ANALYSES):
            ax = axes[row_index, col_index]
            item = lookup[(analysis, inference)]
            left = 0.0
            for stage in STAGES:
                value = float(item["stage_seconds_median"][stage])
                if value <= 0:
                    continue
                ax.barh(
                    [0],
                    [value],
                    left=[left],
                    color=stage_colors[stage],
                    label=stage.replace("_", " "),
                )
                left += value
            wall = float(item["wall_seconds_median"])
            ax.axvline(wall, linewidth=1.0, linestyle=":")
            ax.text(
                0.99,
                0.92,
                f"wall {wall:.1f} s",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
            ax.set_yticks([])
            ax.set_xlabel("Runtime [s]")
            if row_index == 0:
                ax.set_title(analysis.upper(), fontsize=13)
            if col_index == 0:
                ax.set_ylabel(inference.capitalize(), fontsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=stage_colors[stage])
        for stage in STAGES
    ]
    labels = [stage.replace("_", " ") for stage in STAGES]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "hrHSA end-to-end runtime: RSF, SSF and iSSF",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    table = pd.DataFrame(
        [
            {
                "analysis": row["analysis"],
                "inference": row["inference"],
                "repeats": row["repeats"],
                "wall_median_s": row["wall_seconds_median"],
                "wall_min_s": row["wall_seconds_min"],
                "wall_max_s": row["wall_seconds_max"],
                "peak_rss_mb": row["peak_process_tree_rss_mb_median"],
                "candidate_rows": row["n_candidate_rows_median"],
                "predictors": row["model_predictors_median"],
            }
            for row in summary
        ]
    )
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"summary: {summary_path}")
    print(f"figure:  {args.output}")


if __name__ == "__main__":
    main()
