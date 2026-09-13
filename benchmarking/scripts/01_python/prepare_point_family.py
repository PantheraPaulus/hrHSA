"""Prepare deterministic spatially tiled Parquet point workloads.

Each requested point count is stored as its own workload directory.  Rows are
balanced across roughly million-row spatial tiles, and every tile occupies a
non-overlapping rectangle of the normalized unit square.  At benchmark time
``u``/``v`` are mapped to the active raster extent.

This deliberately duplicates point coordinates across workload sizes.  The point
storage is small relative to the raster family, while spatial tiling prevents a
bounded-memory benchmark from repeatedly touching the complete raster once per
arbitrary point batch.  In a complete workload the tiles cover the full domain,
so chunk-aware sampling retains spatial locality and near-single-pass raster I/O.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hsa.compute.point_workloads import PointWorkload, point_workloads
from hsa.compute.workloads import parse_positive_ints


REPRESENTATION = "spatial-tiled-parquet-v2"


def _tile_grid(partitions: int) -> tuple[int, int]:
    """Return an exact near-square rows x columns factorization."""

    if partitions <= 0:
        raise ValueError("partitions must be positive")
    rows = math.isqrt(partitions)
    while rows > 1 and partitions % rows:
        rows -= 1
    cols = partitions // rows
    return rows, cols


def _partition_frame(
    *,
    start: int,
    count: int,
    seed: int,
    target_points: int,
    partition_index: int,
    tile_rows: int,
    tile_cols: int,
) -> pd.DataFrame:
    """Generate one deterministic spatial tile in normalized coordinates."""

    if tile_rows * tile_cols <= partition_index:
        raise ValueError("partition_index lies outside the tile grid")
    tile_row, tile_col = divmod(partition_index, tile_cols)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, target_points, partition_index])
    )
    u = (tile_col + rng.random(count, dtype=np.float64)) / tile_cols
    v = (tile_row + rng.random(count, dtype=np.float64)) / tile_rows
    return pd.DataFrame(
        {
            "point_id": np.arange(start, start + count, dtype=np.int64),
            "u": u,
            "v": v,
            "used": rng.random(count) < 0.1,
        }
    )


def _write_partition(
    *,
    path: str,
    start: int,
    count: int,
    seed: int,
    target_points: int,
    partition_index: int,
    tile_rows: int,
    tile_cols: int,
    compression: str,
) -> tuple[int, int, int]:
    target = Path(path)
    frame = _partition_frame(
        start=start,
        count=count,
        seed=seed,
        target_points=target_points,
        partition_index=partition_index,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    tmp = target.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False, compression=compression)
    tmp.replace(target)
    return target_points, partition_index, count


def _workload_entry(workload: PointWorkload, directory: Path) -> dict[str, object]:
    tile_rows, tile_cols = _tile_grid(workload.partitions)
    return {
        **workload.as_dict(),
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "directory": str(directory),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--points",
        type=parse_positive_ints,
        required=True,
        help="Increasing point-count targets, e.g. 1000000,10000000,30000000.",
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        default=1_000_000,
        help="Target maximum rows per spatial Parquet tile.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.partition_size <= 0 or args.workers <= 0:
        parser.error("partition-size and workers must be positive")

    root = args.root.expanduser().resolve()
    parts_root = root / "points"
    manifest_path = root / "point_family_manifest.json"
    root.mkdir(parents=True, exist_ok=True)

    if args.force:
        if parts_root.exists():
            shutil.rmtree(parts_root)
        manifest_path.unlink(missing_ok=True)

    if parts_root.exists() and not manifest_path.exists():
        raise RuntimeError(
            f"{parts_root} exists without {manifest_path.name}; use --force rather "
            "than guessing whether it matches."
        )

    existing_targets: list[int] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field, expected in (
            ("partition_size", args.partition_size),
            ("seed", args.seed),
            ("representation", REPRESENTATION),
        ):
            observed = existing.get(field)
            if observed != expected:
                raise RuntimeError(
                    f"Existing point family mismatch for {field}: expected "
                    f"{expected!r}, observed {observed!r}. Use --force."
                )
        existing_targets = [
            int(item["target_points"])
            for item in existing.get("workloads", [])
        ]

    targets = sorted(set(existing_targets).union(args.points))
    workloads = point_workloads(targets, partition_size=args.partition_size)
    parts_root.mkdir(parents=True, exist_ok=True)

    specs: list[dict[str, object]] = []
    reused = 0
    entries: list[dict[str, object]] = []

    for workload in workloads:
        workload_dir = parts_root / workload.label
        workload_dir.mkdir(parents=True, exist_ok=True)
        tile_rows, tile_cols = _tile_grid(workload.partitions)
        entries.append(_workload_entry(workload, workload_dir))

        for part_index in range(workload.partitions):
            path = workload_dir / f"part-{part_index:06d}.parquet"
            if path.exists():
                reused += 1
                continue
            specs.append(
                {
                    "path": str(path),
                    "start": workload.start_point_id(part_index),
                    "count": workload.points_in_partition(part_index),
                    "seed": args.seed,
                    "target_points": workload.target_points,
                    "partition_index": part_index,
                    "tile_rows": tile_rows,
                    "tile_cols": tile_cols,
                    "compression": args.compression,
                }
            )

    generated = 0
    if args.workers == 1:
        for spec in specs:
            target_points, part_index, count = _write_partition(**spec)
            generated += 1
            print(
                f"prepared points-{target_points}/part-{part_index:06d}: "
                f"{count:,} points ({generated}/{len(specs)} new)"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_write_partition, **spec) for spec in specs]
            for future in as_completed(futures):
                target_points, part_index, count = future.result()
                generated += 1
                print(
                    f"prepared points-{target_points}/part-{part_index:06d}: "
                    f"{count:,} points ({generated}/{len(specs)} new)"
                )

    manifest = {
        "campaign_data": "spatial-point-workloads-v2",
        "representation": REPRESENTATION,
        "interpretation": (
            "Each exact point-count workload is an independent deterministic "
            "uniform sample stored in balanced spatial tiles. Tiles cover the "
            "normalized domain without overlap, preserving bounded memory and "
            "raster-chunk locality during sampling."
        ),
        "seed": args.seed,
        "partition_size": args.partition_size,
        "compression": args.compression,
        "maximum_points": max(targets),
        "preparation_workers": args.workers,
        "columns": {
            "point_id": "int64",
            "u": "float64 in [0,1)",
            "v": "float64 in [0,1)",
            "used": "bool",
        },
        "workloads": entries,
        "parts_root": str(parts_root),
        "generated_this_run": generated,
        "reused_this_run": reused,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
