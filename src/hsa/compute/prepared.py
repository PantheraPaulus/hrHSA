"""Persistent prepared datasets for repeated RSF fitting and validation.

Cross-validation should not repeat immutable geospatial work. This module writes
one sampled Parquet table per individual together with a manifest and metadata,
so LOIO folds can reuse environmental extraction rather than rebuilding it for
every held-out animal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hsa.compute.chunking import sample_raster_stack_batched
from hsa.compute.config import ExecutionConfig, close_execution, resolve_execution
from hsa.sampling import sample_available_points, sample_raster_stack


def _safe_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return text[:80] or "individual"


@dataclass(frozen=True)
class PreparedDataset:
    """On-disk collection of reusable, per-individual sampled RSF tables."""

    root: Path
    manifest: pd.DataFrame
    metadata: dict[str, Any]

    @classmethod
    def open(cls, root: str | Path) -> "PreparedDataset":
        """Open an existing prepared dataset.

        New datasets use a Parquet manifest so numeric/string identifier types are
        preserved exactly. ``manifest.csv`` is retained as a human-readable sidecar
        and as a backwards-compatible fallback.
        """
        root = Path(root)
        manifest_parquet = root / "manifest.parquet"
        manifest_csv = root / "manifest.csv"
        metadata_path = root / "metadata.json"
        if not metadata_path.exists() or not (
            manifest_parquet.exists() or manifest_csv.exists()
        ):
            raise FileNotFoundError(
                f"{root} is not a prepared hrHSA dataset: manifest or metadata is missing."
            )
        manifest = (
            pd.read_parquet(manifest_parquet)
            if manifest_parquet.exists()
            else pd.read_csv(manifest_csv)
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(root=root, manifest=manifest, metadata=metadata)

    @property
    def id_col(self) -> str:
        return str(self.metadata["id_col"])

    @property
    def predictors(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("predictors", ()))

    @property
    def individuals(self) -> list[Any]:
        return self.manifest["individual_id"].tolist()

    def files(
        self,
        *,
        include: list[Any] | tuple[Any, ...] | None = None,
        exclude: list[Any] | tuple[Any, ...] | Any | None = None,
    ) -> list[Path]:
        """Return selected partition paths in manifest order."""
        selected = self.manifest.copy()
        if include is not None:
            selected = selected.loc[selected["individual_id"].isin(list(include))]
        if exclude is not None:
            excluded = (
                list(exclude)
                if isinstance(exclude, (list, tuple, set))
                else [exclude]
            )
            selected = selected.loc[~selected["individual_id"].isin(excluded)]
        return [self.root / relative for relative in selected["file"]]

    def load(
        self,
        *,
        include: list[Any] | tuple[Any, ...] | None = None,
        exclude: list[Any] | tuple[Any, ...] | Any | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Load selected individual partitions into one dataframe."""
        paths = self.files(include=include, exclude=exclude)
        if not paths:
            return pd.DataFrame(columns=list(columns or ()))
        frames = [pd.read_parquet(path, columns=columns) for path in paths]
        return pd.concat(frames, ignore_index=True)

    def describe(self) -> pd.Series:
        """Return a compact summary useful in notebooks and benchmark logs."""
        return pd.Series(
            {
                "root": str(self.root),
                "individuals": int(len(self.manifest)),
                "rows": int(self.manifest["n_rows"].sum()),
                "used_rows": int(self.manifest["n_used"].sum()),
                "available_rows": int(self.manifest["n_available"].sum()),
                "predictors": ", ".join(self.predictors),
                "sampling_factor": self.metadata.get("sampling_factor"),
                "engine": self.metadata.get("sampling_engine"),
            }
        )


def prepare_rsf_dataset(
    analysis,
    path: str | Path,
    *,
    sampling_factor: int = 10,
    thin_dt: str | None = None,
    seed: int = 42,
    execution: ExecutionConfig | None = None,
    client=None,
    engine: str = "auto",
    overwrite: bool = False,
    dropna: bool = True,
) -> PreparedDataset:
    """Materialise reusable RSF training samples once per individual.

    The expensive sequence ``availability -> raster extraction`` is performed
    exactly once for each animal. Subsequent pooled fits or LOIO folds read the
    resulting Parquet partitions instead of resampling the environmental raster.

    Parameters
    ----------
    analysis
        An :class:`hsa.rsf.RSFAnalysis` instance.
    path
        Destination directory, preferably persistent HPC scratch/project storage.
    engine
        ``"reference"`` uses the original xarray sampler, ``"chunked"`` uses
        bounded calls to the chunk-aware engine, and ``"auto"`` selects
        chunk-aware extraction whenever a distributed client is configured or
        supplied explicitly.
    overwrite
        Remove existing hrHSA prepared files in ``path`` before rebuilding.
    """
    if sampling_factor <= 0:
        raise ValueError("sampling_factor must be positive.")
    if engine not in {"auto", "reference", "chunked"}:
        raise ValueError("engine must be 'auto', 'reference' or 'chunked'.")

    # Imported lazily to keep hsa.compute independent of hsa.rsf during package
    # initialization. This prevents a compute -> rsf -> frequentist -> compute
    # cycle while preserving the established thinning implementation.
    from hsa.rsf.cv import thin_by_time

    root = Path(path)
    manifest_csv = root / "manifest.csv"
    manifest_parquet = root / "manifest.parquet"
    metadata_path = root / "metadata.json"
    partitions = root / "individuals"

    exists = (
        manifest_csv.exists()
        or manifest_parquet.exists()
        or metadata_path.exists()
    )
    if root.exists() and exists and not overwrite:
        raise FileExistsError(
            f"Prepared dataset already exists at {root}. Pass overwrite=True to rebuild it."
        )
    root.mkdir(parents=True, exist_ok=True)
    partitions.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for old_file in partitions.glob("*.parquet"):
            old_file.unlink()

    config, resolved_client, cluster, owns_client = resolve_execution(
        execution,
        client=client,
    )
    selected_engine = engine
    if selected_engine == "auto":
        selected_engine = (
            "chunked"
            if resolved_client is not None or config.distributed
            else "reference"
        )

    rows: list[dict[str, Any]] = []
    sample_bands = list(analysis.predictors)
    try:
        for j, (individual_id, used_i) in enumerate(
            analysis.reloc.groupby(analysis.id_col, sort=False)
        ):
            used_i = used_i.copy()
            if thin_dt is not None:
                used_i = thin_by_time(used_i, min_dt=thin_dt)
            if used_i.empty:
                continue

            points = sample_available_points(
                analysis.domains[individual_id],
                len(used_i) * sampling_factor,
                used=used_i,
                seed=seed + j,
                timestamp_col="Timestamp",
            )
            points[analysis.id_col] = individual_id

            if selected_engine == "chunked":
                sampled = sample_raster_stack_batched(
                    points,
                    analysis.env,
                    bands=sample_bands,
                    id_cols=analysis.id_col,
                    target_chunk_mb=config.chunk_mb,
                    batch_size=config.point_batch_rows,
                    target_batch_mb=config.point_batch_mb,
                    batches_in_flight=config.point_batches_in_flight,
                    client=resolved_client,
                )
            else:
                sampled = sample_raster_stack(
                    points,
                    analysis.env,
                    bands=sample_bands,
                    id_cols=analysis.id_col,
                )

            sampled = sampled.replace([np.inf, -np.inf], np.nan)
            if dropna:
                sampled = sampled.dropna(subset=sample_bands).reset_index(drop=True)

            filename = f"{j:05d}-{_safe_id(individual_id)}.parquet"
            relative = Path("individuals") / filename
            sampled.to_parquet(root / relative, index=False)
            used_mask = (
                sampled["used"].astype(bool)
                if "used" in sampled
                else pd.Series(False, index=sampled.index)
            )
            rows.append(
                {
                    "individual_id": individual_id,
                    "file": relative.as_posix(),
                    "n_rows": int(len(sampled)),
                    "n_used": int(used_mask.sum()),
                    "n_available": int((~used_mask).sum()),
                }
            )
    finally:
        close_execution(
            resolved_client,
            cluster,
            owns_client=owns_client,
        )

    if not rows:
        raise ValueError("No individual partitions were generated.")

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(manifest_parquet, index=False)
    manifest.to_csv(manifest_csv, index=False)
    metadata = {
        "format": "hrHSA-prepared-rsf-v2",
        "id_col": analysis.id_col,
        "predictors": sample_bands,
        "sampling_factor": int(sampling_factor),
        "thin_dt": thin_dt,
        "seed": int(seed),
        "sampling_engine": selected_engine,
        "chunk_mb": int(config.chunk_mb),
        "point_batch_mb": int(config.point_batch_mb),
        "point_batch_rows": (
            None if config.point_batch_rows is None else int(config.point_batch_rows)
        ),
        "point_batches_in_flight": int(config.point_batches_in_flight),
        "n_individuals": int(len(manifest)),
        "n_rows": int(manifest["n_rows"].sum()),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return PreparedDataset(root=root, manifest=manifest, metadata=metadata)