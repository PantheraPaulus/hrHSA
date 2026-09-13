"""Prepare one nested raster family for capacity/strong/weak scaling.

Only the largest requested raster is materialized. Smaller workloads are
storage-chunk-aligned ``isel`` windows of that same deterministic Zarr store.
This avoids duplicating hundreds of GiB/TiB of benchmark data and makes the
problem-size sequence directly comparable.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from hsa.compute.workloads import parse_positive_floats, raster_workloads
from prepare_data import _make_raster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--raster-gib",
        type=parse_positive_floats,
        required=True,
        help="Increasing logical raster targets in GiB, e.g. 24,48,96,192.",
    )
    parser.add_argument("--bands", type=int, default=6)
    parser.add_argument("--storage-chunk", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.bands <= 0 or args.storage_chunk <= 0 or args.workers <= 0:
        parser.error("bands, storage-chunk and workers must be positive")

    workloads = raster_workloads(
        args.raster_gib,
        bands=args.bands,
        dtype="float32",
        storage_chunk=args.storage_chunk,
    )
    largest = workloads[-1]

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    raster_path = root / "environment.zarr"
    manifest_path = root / "surface_family_manifest.json"

    if args.force:
        if raster_path.exists():
            shutil.rmtree(raster_path)
        manifest_path.unlink(missing_ok=True)

    if raster_path.exists():
        if not manifest_path.exists():
            raise RuntimeError(
                f"{raster_path} exists without {manifest_path.name}; "
                "use --force rather than guessing whether it matches."
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed = existing.get("materialized_raster", {})
        expected = largest.as_dict()
        for field in ("size", "bands", "dtype", "storage_chunk", "logical_bytes"):
            if observed.get(field) != expected.get(field):
                raise RuntimeError(
                    "Existing benchmark raster does not match requested family: "
                    f"{field} expected {expected.get(field)!r}, "
                    f"observed {observed.get(field)!r}. Use --force to rebuild."
                )
        print("Reusing existing nested surface benchmark raster:", raster_path)
    else:
        _make_raster(
            raster_path,
            size=largest.size,
            bands=largest.bands,
            storage_chunk=largest.storage_chunk,
            seed=args.seed,
            workers=args.workers,
        )

    manifest = {
        "campaign_data": "nested-surface-workloads-v1",
        "interpretation": (
            "Logical uncompressed raster sizes; only the largest deterministic "
            "Zarr raster is stored and smaller workloads are chunk-aligned nested windows."
        ),
        "seed": args.seed,
        "materialized_raster": largest.as_dict(),
        "workloads": [workload.as_dict() for workload in workloads],
        "raster": str(raster_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
