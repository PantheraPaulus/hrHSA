"""Prepare deterministic point-density scenarios for workstation bisection tests.

The production point benchmark already owns a calibrated 24 GiB raster window and
an exact 100M-point spatially tiled workload.  This preparer builds *validation
roots* that reuse the same raster but expose three point-density patterns to the
unchanged production benchmark runners:

``uniform``
    The existing balanced point family.
``moderate``
    80% of rows in a compact 30% of the spatial tiles.
``strong``
    85% of rows in a compact 20% of the spatial tiles.

Preparation is intentionally outside benchmark timing.  Skewed scenarios are
materialized directly as x/y Parquet partitions in the cache format consumed by
``profile_point_partitioned_production.py``.  This avoids storing an additional
u/v source copy and guarantees that direct and bisection strategies read exactly
the same files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from profile_point_partitioned_production import (
    _cache_signature,
    _materialize_xy_cache,
    _xy_cache_directory,
)
from run_point_scaling import _load_inputs, _point_workload_entry
from run_surface_scaling import _window


SCENARIOS: dict[str, dict[str, float | None]] = {
    "uniform": {
        "hot_domain_fraction": None,
        "hot_point_fraction": None,
    },
    "moderate": {
        "hot_domain_fraction": 0.30,
        "hot_point_fraction": 0.80,
    },
    "strong": {
        "hot_domain_fraction": 0.20,
        "hot_point_fraction": 0.85,
    },
}


def _parse_scenarios(value: str) -> list[str]:
    scenarios = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(scenarios).difference(SCENARIOS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown scenario(s): {', '.join(unknown)}")
    if not scenarios:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    return scenarios


def _even_integer_allocation(total: int, indices: Iterable[int]) -> dict[int, int]:
    indices = list(sorted(int(index) for index in indices))
    if total < 0 or not indices:
        raise ValueError("allocation needs a non-negative total and non-empty indices")
    quotient, remainder = divmod(int(total), len(indices))
    return {
        index: quotient + (1 if order < remainder else 0)
        for order, index in enumerate(indices)
    }


def _centered_hot_indices(tile_rows: int, tile_cols: int, count: int) -> list[int]:
    """Choose a deterministic compact group of tiles nearest the domain centre."""
    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile grid dimensions must be positive")
    total = tile_rows * tile_cols
    if not 0 < count < total:
        raise ValueError("hot tile count must lie strictly between zero and total tiles")

    centre_row = (tile_rows - 1) / 2.0
    centre_col = (tile_cols - 1) / 2.0
    ranked = []
    for row in range(tile_rows):
        for col in range(tile_cols):
            index = row * tile_cols + col
            distance2 = (row - centre_row) ** 2 + (col - centre_col) ** 2
            ranked.append((distance2, abs(row - centre_row), abs(col - centre_col), index))
    return sorted(item[3] for item in sorted(ranked)[:count])


def _scenario_row_counts(
    *,
    total_rows: int,
    tile_rows: int,
    tile_cols: int,
    scenario: str,
) -> tuple[list[int], list[int]]:
    """Return exact per-tile row counts and the compact hot-tile indices."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    partitions = tile_rows * tile_cols
    if scenario == "uniform":
        allocation = _even_integer_allocation(total_rows, range(partitions))
        return [allocation[index] for index in range(partitions)], []

    spec = SCENARIOS[scenario]
    hot_domain_fraction = float(spec["hot_domain_fraction"])
    hot_point_fraction = float(spec["hot_point_fraction"])
    hot_count = max(1, min(partitions - 1, int(round(partitions * hot_domain_fraction))))
    hot_indices = _centered_hot_indices(tile_rows, tile_cols, hot_count)
    hot_set = set(hot_indices)
    cold_indices = [index for index in range(partitions) if index not in hot_set]

    hot_rows = int(round(total_rows * hot_point_fraction))
    cold_rows = int(total_rows) - hot_rows
    allocation = _even_integer_allocation(hot_rows, hot_indices)
    allocation.update(_even_integer_allocation(cold_rows, cold_indices))
    rows = [allocation[index] for index in range(partitions)]
    if sum(rows) != int(total_rows):
        raise RuntimeError("scenario row allocation does not preserve the requested total")
    return rows, hot_indices


def _write_xy_partition(
    *,
    path: str,
    start: int,
    count: int,
    seed: int,
    scenario_code: int,
    partition_index: int,
    tile_rows: int,
    tile_cols: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    compression: str,
) -> dict[str, Any]:
    """Generate one deterministic x/y tile and return cache-manifest metadata."""
    target = Path(path)
    tile_row, tile_col = divmod(partition_index, tile_cols)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, scenario_code, partition_index, count])
    )
    x = xmin + ((tile_col + rng.random(count, dtype=np.float64)) / tile_cols) * (
        xmax - xmin
    )
    y = ymin + ((tile_row + rng.random(count, dtype=np.float64)) / tile_rows) * (
        ymax - ymin
    )
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "point_id": np.arange(start, start + count, dtype=np.int64),
            "used": rng.random(count) < 0.1,
        }
    )
    tmp = target.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False, compression=compression)
    tmp.replace(target)
    bounds = [
        float(np.min(x)),
        float(np.min(y)),
        float(np.max(x)),
        float(np.max(y)),
    ]
    return {
        "partition_index": int(partition_index),
        "file": target.name,
        "rows": int(count),
        "bounds": bounds,
    }


def _ensure_environment_link(source_root: Path, scenario_root: Path) -> None:
    source = (source_root / "environment.zarr").resolve()
    target = scenario_root / "environment.zarr"
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(f"{target} points at the wrong raster")
        return
    if target.exists():
        raise RuntimeError(f"{target} exists and is not the expected symlink")
    target.symlink_to(source, target_is_directory=True)


def _write_scenario_manifests(
    *,
    source_root: Path,
    scenario_root: Path,
    entry: dict[str, Any],
    scenario: str,
    row_counts: list[int],
    hot_indices: list[int],
    cache_items: list[dict[str, Any]],
    cache_signature: dict[str, Any],
    raster_gib: float,
) -> None:
    shutil.copy2(
        source_root / "surface_family_manifest.json",
        scenario_root / "surface_family_manifest.json",
    )
    point_manifest = {
        "campaign_data": "workstation-point-bisection-validation-v1",
        "representation": "spatial-tiled-parquet-v2-bisection-validation",
        "interpretation": (
            "Synthetic point-density scenario used only to compare direct partitioned "
            "execution with recursive workload-balanced spatial bisection."
        ),
        "workloads": [entry],
    }
    (scenario_root / "point_family_manifest.json").write_text(
        json.dumps(point_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    cache = scenario_root / "points-xy-cache" / f"points-{int(entry['target_points'])}" / (
        f"raster-{cache_signature['raster_shape'][0]}x{cache_signature['raster_shape'][1]}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "signature": cache_signature,
                "partitions": cache_items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = np.asarray(row_counts, dtype=np.float64)
    validation = {
        "campaign": "workstation-point-bisection-validation-v1",
        "scenario": scenario,
        "source_root": str(source_root),
        "scenario_root": str(scenario_root),
        "target_points": int(entry["target_points"]),
        "partitions": int(entry["partitions"]),
        "tile_rows": int(entry["tile_rows"]),
        "tile_cols": int(entry["tile_cols"]),
        "raster_gib": float(raster_gib),
        "hot_domain_fraction": SCENARIOS[scenario]["hot_domain_fraction"],
        "hot_point_fraction": SCENARIOS[scenario]["hot_point_fraction"],
        "hot_partition_indices": hot_indices,
        "partition_rows": [int(value) for value in row_counts],
        "partition_rows_mean": float(np.mean(rows)),
        "partition_rows_std": float(np.std(rows)),
        "partition_rows_cv": float(np.std(rows) / np.mean(rows)),
        "partition_rows_min": int(np.min(rows)),
        "partition_rows_max": int(np.max(rows)),
        "partition_rows_max_over_mean": float(np.max(rows) / np.mean(rows)),
        "cache_manifest": str(cache / "manifest.json"),
    }
    (scenario_root / "bisection_validation_manifest.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_one(
    *,
    source_root: Path,
    validation_root: Path,
    env,
    entry: dict[str, Any],
    scenario: str,
    raster_gib: float,
    prepare_workers: int,
    seed: int,
    compression: str,
    force: bool,
) -> Path:
    scenario_root = validation_root / scenario
    scenario_root.mkdir(parents=True, exist_ok=True)
    _ensure_environment_link(source_root, scenario_root)

    logical_source = scenario_root / "points" / f"points-{int(entry['target_points'])}"
    logical_source.mkdir(parents=True, exist_ok=True)
    scenario_entry = {
        **entry,
        "directory": str(logical_source.resolve()),
    }
    signature = _cache_signature(scenario_entry, env)
    cache = _xy_cache_directory(scenario_root, scenario_entry, env)
    validation_manifest = scenario_root / "bisection_validation_manifest.json"

    if force and cache.exists():
        shutil.rmtree(cache)
    if force:
        validation_manifest.unlink(missing_ok=True)

    tile_rows = int(entry["tile_rows"])
    tile_cols = int(entry["tile_cols"])
    partition_count = int(entry["partitions"])
    if tile_rows * tile_cols != partition_count:
        raise RuntimeError(
            f"prepared tile grid {tile_rows}x{tile_cols} does not match {partition_count} partitions"
        )
    row_counts, hot_indices = _scenario_row_counts(
        total_rows=int(entry["target_points"]),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        scenario=scenario,
    )

    if validation_manifest.exists() and (cache / "manifest.json").exists():
        existing = json.loads(validation_manifest.read_text(encoding="utf-8"))
        if (
            existing.get("scenario") == scenario
            and int(existing.get("target_points", -1)) == int(entry["target_points"])
            and existing.get("partition_rows") == row_counts
            and all(
                (cache / item["file"]).exists()
                for item in json.loads((cache / "manifest.json").read_text(encoding="utf-8")).get(
                    "partitions", []
                )
            )
        ):
            print(f"reusing prepared {scenario} validation root: {scenario_root}")
            return scenario_root
        raise RuntimeError(
            f"Existing validation root {scenario_root} does not match this specification; "
            "rerun with --force."
        )

    if scenario == "uniform":
        original = _materialize_xy_cache(source_root, entry, env)
        cache_items = [
            {
                "file": str(part.path.resolve()),
                "rows": int(part.rows or 0),
                "bounds": [float(value) for value in part.bounds],
            }
            for part in original
        ]
        row_counts = [int(part.rows or 0) for part in original]
    else:
        cache.mkdir(parents=True, exist_ok=True)
        x = np.asarray(env["x"].values, dtype=np.float64)
        y = np.asarray(env["y"].values, dtype=np.float64)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        starts = np.cumsum([0, *row_counts[:-1]], dtype=np.int64)
        scenario_code = {"moderate": 1, "strong": 2}[scenario]
        specs = [
            {
                "path": str(cache / f"part-{index:06d}.parquet"),
                "start": int(starts[index]),
                "count": int(row_counts[index]),
                "seed": int(seed),
                "scenario_code": scenario_code,
                "partition_index": index,
                "tile_rows": tile_rows,
                "tile_cols": tile_cols,
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "compression": compression,
            }
            for index in range(partition_count)
        ]
        cache_items = []
        if prepare_workers == 1:
            for order, spec in enumerate(specs, start=1):
                cache_items.append(_write_xy_partition(**spec))
                if order % 10 == 0 or order == len(specs):
                    print(f"{scenario}: prepared {order}/{len(specs)} partitions")
        else:
            with ProcessPoolExecutor(max_workers=prepare_workers) as pool:
                futures = [pool.submit(_write_xy_partition, **spec) for spec in specs]
                for order, future in enumerate(as_completed(futures), start=1):
                    cache_items.append(future.result())
                    if order % 10 == 0 or order == len(specs):
                        print(f"{scenario}: prepared {order}/{len(specs)} partitions")
        cache_items.sort(key=lambda item: int(item["partition_index"]))
        for item in cache_items:
            item.pop("partition_index", None)

    _write_scenario_manifests(
        source_root=source_root,
        scenario_root=scenario_root,
        entry=scenario_entry,
        scenario=scenario,
        row_counts=row_counts,
        hot_indices=hot_indices,
        cache_items=cache_items,
        cache_signature=signature,
        raster_gib=raster_gib,
    )
    print(
        f"prepared {scenario}: rows={sum(row_counts):,} "
        f"min={min(row_counts):,} max={max(row_counts):,} root={scenario_root}"
    )
    return scenario_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare workstation point-bisection validation scenarios."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=100_000_000)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument(
        "--scenarios",
        type=_parse_scenarios,
        default=["uniform", "moderate", "strong"],
    )
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=926_2026)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.point_count <= 0 or args.raster_gib <= 0 or args.prepare_workers <= 0:
        parser.error("point-count, raster-gib and prepare-workers must be positive")

    source_root = args.root.expanduser().resolve()
    validation_root = args.validation_root.expanduser().resolve()
    validation_root.mkdir(parents=True, exist_ok=True)

    full_env, surface_manifest, point_manifest = _load_inputs(source_root)
    entry = _point_workload_entry(point_manifest, args.point_count)
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", args.spatial_chunk
        )
    )
    env, workload = _window(full_env, args.raster_gib, storage_chunk)

    prepared = []
    for scenario in args.scenarios:
        prepared.append(
            _prepare_one(
                source_root=source_root,
                validation_root=validation_root,
                env=env,
                entry=entry,
                scenario=scenario,
                raster_gib=float(workload.logical_gib),
                prepare_workers=args.prepare_workers,
                seed=args.seed,
                compression=args.compression,
                force=args.force,
            )
        )

    print("validation roots:")
    for path in prepared:
        print(path)


if __name__ == "__main__":
    main()
