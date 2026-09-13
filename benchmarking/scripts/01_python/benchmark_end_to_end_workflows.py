"""End-to-end workstation benchmark for RSF, SSF and iSSF workflows.

The benchmark starts from one deterministic synthetic trajectory family and a
prepared storage-backed raster. Each measured run executes one scientific
workflow (RSF, SSF or iSSF) with either frequentist or Bayesian inference.

The immutable raster and raw synthetic relocations are benchmark fixtures and
are prepared outside the end-to-end timer. Everything from availability/model
structure construction through fitted-model diagnostics is timed in named
stages. Static raster extraction uses the production partition-native sampler
inside six workload-balanced spatial shards on the calibrated 12-core
workstation.

This is intentionally a fit-to-diagnostics benchmark. Cross-validation and
full-raster prediction are separate workloads because they multiply/refocus the
statistical work and do not have direct counterparts across all three analyses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rioxarray  # noqa: F401 - register xarray .rio accessor
import xarray as xr
from shapely.geometry import Point

from hsa.compute import (
    PointPartition,
    iter_sample_raster_stack_partitioned,
    plan_spatial_point_shards,
)
from hsa.movement import fit_movement_kernel_per_id
from hsa.rsf.bayesian import (
    build_bayesian_rsf_model,
    evaluate_bayesian_rsf,
    prepare_bayesian_rsf_data,
)
from hsa.rsf.bayesian_hpc import _materialize_inference_tree
from hsa.rsf.cv import _domains_by_id
from hsa.rsf.model import fit_prepared_rsf, prepare_rsf_design, predict_rsf_points
from hsa.sampling import sample_available_points
from hsa.ssf.bayesian import BayesianSSFFit, build_hierarchical_ssf_model
from hsa.ssf.choice_sets import build_movement_choice_sets
from hsa.ssf.data import (
    apply_ssf_scaling,
    build_ssf_choice_arrays,
    complete_ssf_strata,
    fit_ssf_scaling,
)
from hsa.ssf.frequentist import FrequentistSSFFit, fit_conditional_ssf
from hsa.ssf.issf import (
    BayesianISSFFit,
    FrequentistISSFFit,
    build_hierarchical_issf_model,
    prepare_issf_design,
)
from hsa.types import FeatureSpec

from profile_point_partitioned_spatial_shard import (
    _aligned_raster_window,
    _physical_core_groups,
    _pin_workers,
)
from run_surface_scaling import _window


STAGE_ORDER = (
    "structure",
    "availability",
    "point_partitioning",
    "raster_sampling",
    "design",
    "model_build",
    "inference",
    "diagnostics",
)


def _load_surface_family(root: Path):
    manifest_path = root / "surface_family_manifest.json"
    raster_path = root / "environment.zarr"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")
    if not raster_path.exists():
        raise FileNotFoundError(f"Missing {raster_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = xr.open_zarr(raster_path, chunks={}, decode_coords="all")
    if "environment" not in dataset:
        raise RuntimeError(f"{raster_path} has no 'environment' data variable")
    env = dataset["environment"]
    if env.rio.crs is None:
        env = env.rio.write_crs("EPSG:3857")
    return env, manifest


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@contextmanager
def _stage(stages: dict[str, float], name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        stages[name] = stages.get(name, 0.0) + (time.perf_counter() - started)


class _ProcessTreePeakRSS:
    """Sample RSS of this process and descendants during one E2E run."""

    def __init__(self, interval: float = 0.2):
        self.interval = float(interval)
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        root = psutil.Process()
        while not self._stop.is_set():
            total = 0
            processes = [root]
            try:
                processes.extend(root.children(recursive=True))
            except (psutil.Error, OSError):
                pass
            for process in processes:
                try:
                    total += int(process.memory_info().rss)
                except (psutil.Error, OSError):
                    pass
            self.peak_bytes = max(self.peak_bytes, total)
            self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5 * self.interval))
        self._sample_once()

    def _sample_once(self) -> None:
        root = psutil.Process()
        total = 0
        processes = [root]
        try:
            processes.extend(root.children(recursive=True))
        except (psutil.Error, OSError):
            pass
        for process in processes:
            try:
                total += int(process.memory_info().rss)
            except (psutil.Error, OSError):
                pass
        self.peak_bytes = max(self.peak_bytes, total)


def _reflect(value: float, low: float, high: float) -> float:
    """Reflect a scalar back into [low, high] without clipping to an edge."""
    if low >= high:
        raise ValueError("invalid reflection interval")
    width = high - low
    shifted = (value - low) % (2.0 * width)
    if shifted > width:
        shifted = 2.0 * width - shifted
    return low + shifted


def _synthetic_trajectories(
    env,
    *,
    individuals: int,
    steps_per_individual: int,
    seed: int,
    id_col: str,
) -> gpd.GeoDataFrame:
    """Create regular hourly trajectories safely inside the raster domain."""
    if individuals <= 0 or steps_per_individual <= 2:
        raise ValueError("individuals must be positive and steps_per_individual > 2")

    x = np.asarray(env["x"].values, dtype=np.float64)
    y = np.asarray(env["y"].values, dtype=np.float64)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("raster has invalid spatial extent")

    xlo, xhi = xmin + 0.20 * width, xmax - 0.20 * width
    ylo, yhi = ymin + 0.20 * height, ymax - 0.20 * height
    mean_step = 0.0012 * min(width, height)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    n_fixes = int(steps_per_individual) + 2
    start_time = pd.Timestamp("2025-01-01T00:00:00Z")

    for individual in range(individuals):
        px = rng.uniform(xlo + 0.15 * (xhi - xlo), xhi - 0.15 * (xhi - xlo))
        py = rng.uniform(ylo + 0.15 * (yhi - ylo), yhi - 0.15 * (yhi - ylo))
        heading = rng.uniform(-np.pi, np.pi)
        for fix in range(n_fixes):
            rows.append(
                {
                    id_col: f"ID{individual:03d}",
                    "Timestamp": start_time + pd.Timedelta(hours=fix),
                    "geometry": Point(px, py),
                }
            )
            if fix == n_fixes - 1:
                continue
            step = rng.gamma(shape=2.0, scale=mean_step / 2.0)
            heading += rng.vonmises(mu=0.0, kappa=2.0)
            nx = px + step * math.cos(heading)
            ny = py + step * math.sin(heading)
            reflected_x = _reflect(nx, xlo, xhi)
            reflected_y = _reflect(ny, ylo, yhi)
            if reflected_x != nx:
                heading = np.pi - heading
            if reflected_y != ny:
                heading = -heading
            px, py = reflected_x, reflected_y

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=env.rio.crs)


def _candidate_frame_from_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    preserve: Iterable[str],
) -> pd.DataFrame:
    preserve = list(dict.fromkeys(str(column) for column in preserve))
    missing = [column for column in preserve if column not in gdf.columns]
    if missing:
        raise KeyError(f"candidate table missing preserved columns: {missing}")
    frame = gdf[preserve].copy()
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    frame["x"] = gdf.geometry.x.to_numpy(dtype=np.float64)
    frame["y"] = gdf.geometry.y.to_numpy(dtype=np.float64)
    return frame


def _write_spatial_partitions(
    frame: pd.DataFrame,
    env,
    directory: Path,
    *,
    grid: int,
) -> list[PointPartition]:
    """Materialize a deterministic grid of spatial Parquet point partitions."""
    if grid <= 0:
        raise ValueError("grid must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("part-*.parquet"):
        stale.unlink()

    x = np.asarray(env["x"].values, dtype=np.float64)
    y = np.asarray(env["y"].values, dtype=np.float64)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    width = max(xmax - xmin, np.finfo(float).eps)
    height = max(ymax - ymin, np.finfo(float).eps)

    col = np.floor((frame["x"].to_numpy(dtype=float) - xmin) / width * grid).astype(int)
    row = np.floor((frame["y"].to_numpy(dtype=float) - ymin) / height * grid).astype(int)
    col = np.clip(col, 0, grid - 1)
    row = np.clip(row, 0, grid - 1)
    tile = row * grid + col

    partitions: list[PointPartition] = []
    part_index = 0
    for tile_id in np.unique(tile):
        subset = frame.loc[tile == tile_id].copy()
        if subset.empty:
            continue
        path = directory / f"part-{part_index:06d}.parquet"
        subset.to_parquet(path, index=False, compression="zstd")
        bounds = (
            float(subset["x"].min()),
            float(subset["y"].min()),
            float(subset["x"].max()),
            float(subset["y"].max()),
        )
        partitions.append(
            PointPartition(
                path=path,
                bounds=bounds,
                rows=len(subset),
                crs=env.rio.crs,
            )
        )
        part_index += 1
    if not partitions:
        raise RuntimeError("spatial partitioning produced no point partitions")
    return partitions


def _serialize_partition(partition: PointPartition) -> dict[str, Any]:
    return {
        "path": str(partition.path),
        "bounds": list(partition.bounds),
        "rows": partition.rows,
        "crs": str(partition.crs) if partition.crs is not None else None,
    }


def _deserialize_partition(payload: dict[str, Any]) -> PointPartition:
    return PointPartition(
        path=payload["path"],
        bounds=tuple(payload["bounds"]),
        rows=payload.get("rows"),
        crs=payload.get("crs"),
    )


def _sample_spatial_shard(spec: dict[str, Any]) -> dict[str, Any]:
    """Sample one bisection shard and materialize sampled partitions."""
    from distributed import Client, LocalCluster

    cpu_ids = [int(value) for value in spec["cpu_ids"]]
    os.sched_setaffinity(0, set(cpu_ids))
    core_groups = _physical_core_groups(cpu_ids)
    if len(core_groups) != int(spec["threads_per_shard"]):
        raise RuntimeError(
            f"shard {spec['shard_index']} has {len(core_groups)} physical cores; "
            f"expected {spec['threads_per_shard']}"
        )

    root = Path(spec["root"])
    full_env, surface_manifest = _load_surface_family(root)
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", spec["spatial_chunk"]
        )
    )
    env, _ = _window(full_env, float(spec["raster_gib"]), storage_chunk)
    partitions = [_deserialize_partition(item) for item in spec["partitions"]]
    shard_env, raster_window = _aligned_raster_window(
        env,
        partitions,
        spatial_chunk=int(spec["spatial_chunk"]),
    )
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    local_root = Path(spec["local_root"])
    local_root.mkdir(parents=True, exist_ok=True)

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=int(spec["threads_per_shard"]),
        processes=True,
        memory_limit=int(float(spec["managed_memory_gib"]) * 1024**3),
        local_directory=str(local_root),
        dashboard_address=":0",
    )
    client = Client(cluster)
    started = time.perf_counter()
    rows = 0
    try:
        client.wait_for_workers(1, timeout=float(spec["worker_startup_timeout"]))
        _pin_workers(
            client,
            task_core_groups=core_groups,
            workers=1,
            threads=int(spec["threads_per_shard"]),
        )
        chunks = {
            "band": -1,
            "y": int(spec["spatial_chunk"]),
            "x": int(spec["spatial_chunk"]),
        }
        for sampled in iter_sample_raster_stack_partitioned(
            partitions,
            shard_env,
            bands=spec["bands"],
            preserve_cols=spec["preserve_cols"],
            chunks=chunks,
            graph_partitions=min(int(spec["graph_partitions"]), len(partitions)),
            client=client,
            require_inside=True,
        ):
            target = output_dir / sampled.source.path.name
            sampled.frame.to_parquet(target, index=False, compression="zstd")
            rows += len(sampled.frame)
    finally:
        client.close()
        cluster.close()

    return {
        "shard_index": int(spec["shard_index"]),
        "rows": int(rows),
        "wall_seconds": time.perf_counter() - started,
        "raster_window": raster_window,
    }


def _workstation_cpu_sets(shard_count: int, threads_per_shard: int) -> list[list[int]]:
    allowed = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    groups = _physical_core_groups(allowed)
    required = int(shard_count) * int(threads_per_shard)
    if len(groups) != required:
        raise RuntimeError(
            "The calibrated workstation E2E profile expects exactly "
            f"{required} physical cores but this process sees {len(groups)}. "
            "Run on the 12-core workstation or override shard/thread settings."
        )
    cpu_sets = []
    for shard_index in range(shard_count):
        selected = groups[
            shard_index * threads_per_shard : (shard_index + 1) * threads_per_shard
        ]
        cpu_sets.append(sorted(cpu for group in selected for cpu in group))
    return cpu_sets


def _sample_bisection_materialized(
    *,
    partitions: list[PointPartition],
    root: Path,
    output_dir: Path,
    bands: list[str],
    preserve_cols: list[str],
    raster_gib: float,
    shard_count: int,
    threads_per_shard: int,
    graph_partitions: int,
    spatial_chunk: int,
    managed_memory_gib: float,
    worker_startup_timeout: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run weighted spatial shards concurrently and recover source row order."""
    if shard_count > len(partitions):
        raise ValueError(
            f"Need at least {shard_count} non-empty spatial partitions; got {len(partitions)}"
        )
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    planner_started = time.perf_counter()
    plan = plan_spatial_point_shards(partitions, shard_count)
    planner_seconds = time.perf_counter() - planner_started
    cpu_sets = _workstation_cpu_sets(shard_count, threads_per_shard)
    local_root = output_dir / "dask-local"

    specs = []
    for shard_index, shard in enumerate(plan):
        shard_parts = [partitions[index] for index in shard.partition_indices]
        specs.append(
            {
                "root": str(root),
                "output_dir": str(output_dir / "sampled"),
                "local_root": str(local_root / f"shard-{shard_index:02d}"),
                "shard_index": shard_index,
                "cpu_ids": cpu_sets[shard_index],
                "threads_per_shard": threads_per_shard,
                "partitions": [_serialize_partition(partition) for partition in shard_parts],
                "bands": list(bands),
                "preserve_cols": list(preserve_cols),
                "raster_gib": raster_gib,
                "graph_partitions": graph_partitions,
                "spatial_chunk": spatial_chunk,
                "managed_memory_gib": managed_memory_gib,
                "worker_startup_timeout": worker_startup_timeout,
            }
        )

    concurrent_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=shard_count) as pool:
        shard_results = list(pool.map(_sample_spatial_shard, specs))
    concurrent_seconds = time.perf_counter() - concurrent_started

    sampled_paths = sorted((output_dir / "sampled").glob("part-*.parquet"))
    if len(sampled_paths) != len(partitions):
        raise RuntimeError(
            f"Expected {len(partitions)} sampled partitions; found {len(sampled_paths)}"
        )
    sampled = pd.concat(
        [pd.read_parquet(path) for path in sampled_paths],
        ignore_index=True,
    ).sort_values("row_id").reset_index(drop=True)
    expected_rows = sum(int(partition.rows or 0) for partition in partitions)
    if len(sampled) != expected_rows:
        raise RuntimeError(
            f"Bisection sampler returned {len(sampled):,} rows; expected {expected_rows:,}"
        )

    rows_by_shard = [int(item.rows) for item in plan]
    return sampled, {
        "planner_seconds": planner_seconds,
        "concurrent_seconds": concurrent_seconds,
        "shard_count": shard_count,
        "threads_per_shard": threads_per_shard,
        "shard_rows_min": min(rows_by_shard),
        "shard_rows_max": max(rows_by_shard),
        "shard_rows_cv": float(np.std(rows_by_shard) / np.mean(rows_by_shard)),
        "shard_wall_seconds": [float(item["wall_seconds"]) for item in shard_results],
    }


def _choice_preserve_columns(choices: gpd.GeoDataFrame, id_col: str) -> list[str]:
    desired = [
        id_col,
        "stratum_id",
        "candidate_id",
        "used",
        "start_time",
        "end_time",
        "dt_h",
        "start_x",
        "start_y",
        "step_length",
        "turn_angle",
        "proposal_logpdf",
        "start_x_condition",
        "start_y_condition",
        "start_time_sin",
    ]
    return [column for column in desired if column in choices.columns]


def _add_start_conditions(choices: gpd.GeoDataFrame, env) -> gpd.GeoDataFrame:
    out = choices.copy()
    x = np.asarray(env["x"].values, dtype=float)
    y = np.asarray(env["y"].values, dtype=float)
    xmid = 0.5 * (float(np.min(x)) + float(np.max(x)))
    ymid = 0.5 * (float(np.min(y)) + float(np.max(y)))
    xscale = max(float(np.max(x)) - float(np.min(x)), np.finfo(float).eps)
    yscale = max(float(np.max(y)) - float(np.min(y)), np.finfo(float).eps)
    out["start_x_condition"] = (out["start_x"].astype(float) - xmid) / xscale
    out["start_y_condition"] = (out["start_y"].astype(float) - ymid) / yscale
    start_time = pd.to_datetime(out["start_time"], utc=True)
    hour = (
        start_time.dt.hour.to_numpy(dtype=float)
        + start_time.dt.minute.to_numpy(dtype=float) / 60.0
    )
    out["start_time_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    return out


def _sampling_kwargs(
    pm,
    *,
    sampler: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    seed: int,
) -> dict[str, Any]:
    signature = inspect.signature(pm.sample)
    kwargs: dict[str, Any] = {
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "cores": int(cores),
        "target_accept": float(target_accept),
        "progressbar": False,
        "return_inferencedata": True,
        "random_seed": int(seed),
    }
    if "compute_convergence_checks" in signature.parameters:
        kwargs["compute_convergence_checks"] = False
    if "blas_cores" in signature.parameters:
        kwargs["blas_cores"] = 1
    if "nuts_sampler" in signature.parameters:
        kwargs["nuts_sampler"] = sampler
    elif sampler != "pymc":
        raise RuntimeError("This PyMC version has no nuts_sampler argument")
    return kwargs


def _bayesian_choice_diagnostics(idata, *, completed_seconds: float) -> dict[str, Any]:
    import arviz as az

    summary = az.summary(idata, var_names=["mu_beta", "sigma_beta"], round_to=None)
    ess_bulk = summary["ess_bulk"].to_numpy(dtype=float)
    ess_tail = summary["ess_tail"].to_numpy(dtype=float)
    rhat = summary["r_hat"].to_numpy(dtype=float)
    stats = idata.sample_stats
    divergences = int(np.asarray(stats["diverging"]).sum()) if "diverging" in stats else 0
    mean_n_steps = (
        float(np.asarray(stats["n_steps"]).mean()) if "n_steps" in stats else np.nan
    )
    return {
        "min_ess_bulk": float(np.nanmin(ess_bulk)),
        "median_ess_bulk": float(np.nanmedian(ess_bulk)),
        "min_ess_tail": float(np.nanmin(ess_tail)),
        "median_ess_tail": float(np.nanmedian(ess_tail)),
        "median_ess_bulk_per_second": float(np.nanmedian(ess_bulk)) / completed_seconds,
        "max_rhat": float(np.nanmax(rhat)),
        "n_divergences": divergences,
        "mean_n_steps": mean_n_steps,
    }


def _load_benchmark_env(root: Path, raster_gib: float, spatial_chunk: int):
    full_env, surface_manifest = _load_surface_family(root)
    storage_chunk = int(
        surface_manifest.get("materialized_raster", {}).get(
            "storage_chunk", spatial_chunk
        )
    )
    env, workload = _window(full_env, raster_gib, storage_chunk)
    return env, workload


def _run_rsf(
    *,
    inference: str,
    reloc: gpd.GeoDataFrame,
    env,
    root: Path,
    workspace: Path,
    bands: list[str],
    args,
    stages: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    id_col = args.id_col
    with _stage(stages, "structure"):
        domains = _domains_by_id(reloc, id_col=id_col, domain=None, quantile=0.95)

    with _stage(stages, "availability"):
        parts = []
        for index, (individual_id, used_i) in enumerate(reloc.groupby(id_col, sort=False)):
            sampled_i = sample_available_points(
                domains[individual_id],
                len(used_i) * args.n_available,
                used=used_i,
                seed=args.seed + index,
                timestamp_col="Timestamp",
            )
            sampled_i[id_col] = individual_id
            parts.append(sampled_i)
        candidates = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True),
            geometry="geometry",
            crs=reloc.crs,
        )

    preserve = [id_col, "used"]
    frame = _candidate_frame_from_gdf(candidates, preserve=preserve)
    with _stage(stages, "point_partitioning"):
        partitions = _write_spatial_partitions(
            frame,
            env,
            workspace / "points",
            grid=args.partition_grid,
        )

    with _stage(stages, "raster_sampling"):
        sampled, point_meta = _sample_bisection_materialized(
            partitions=partitions,
            root=root,
            output_dir=workspace / "sampled",
            bands=bands,
            preserve_cols=["row_id", *preserve],
            raster_gib=args.raster_gib,
            shard_count=args.point_shards,
            threads_per_shard=args.threads_per_shard,
            graph_partitions=args.graph_partitions,
            spatial_chunk=args.spatial_chunk,
            managed_memory_gib=args.managed_memory_gib_per_shard,
            worker_startup_timeout=args.worker_startup_timeout,
        )

    sampled = (
        sampled.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=bands)
        .reset_index(drop=True)
    )

    if inference == "frequentist":
        spec = FeatureSpec(linear=list(bands))
        with _stage(stages, "design"):
            prepared = prepare_rsf_design(sampled, spec)
        stages.setdefault("model_build", 0.0)
        with _stage(stages, "inference"):
            result = fit_prepared_rsf(prepared, method="lbfgs")
        with _stage(stages, "diagnostics"):
            predicted = predict_rsf_points(
                sampled,
                result,
                prepared.scaler,
                prepared.spec,
                prepared.meta,
            )
            diagnostics = {
                "converged": bool(result.mle_retvals.get("converged", True)),
                "llf": float(result.llf),
                "prediction_mean": float(predicted["rsf_pred"].mean()),
            }
        model_predictors = (
            prepared.n_columns - 1 if prepared.spec.add_const else prepared.n_columns
        )
    else:
        import pymc as pm

        with _stage(stages, "design"):
            binning = {}
            for band in bands:
                sd = float(pd.to_numeric(sampled[band], errors="coerce").std(ddof=0))
                binning[band] = max(0.5 * sd, np.finfo(float).eps)
            bayes_data = prepare_bayesian_rsf_data(
                sampled,
                list(bands),
                id_col=id_col,
                binning=binning,
            )
        with _stage(stages, "model_build"):
            model = build_bayesian_rsf_model(
                bayes_data,
                predictors=bands,
                random_intercept=True,
                random_slopes=bands[:1],
                store_eta=False,
            )
        sampling = _sampling_kwargs(
            pm,
            sampler=args.rsf_sampler,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.bayesian_cores,
            target_accept=args.rsf_target_accept,
            seed=args.seed + 10_000,
        )
        with _stage(stages, "inference"):
            started = time.perf_counter()
            with model:
                raw_idata = pm.sample(**sampling)
            idata = _materialize_inference_tree(raw_idata)
            completed_sampling = time.perf_counter() - started
        with _stage(stages, "diagnostics"):
            evaluated = evaluate_bayesian_rsf(idata, include_random_effects=False)
            table = evaluated["diagnostics"]
            diagnostics = {
                "min_ess_bulk": float(evaluated["min_ess_bulk"]),
                "min_ess_tail": float(evaluated["min_ess_tail"]),
                "max_rhat": float(evaluated["max_rhat"]),
                "n_divergences": int(evaluated["n_divergences"]),
                "median_ess_bulk": (
                    float(table["ess_bulk"].median())
                    if "ess_bulk" in table
                    else np.nan
                ),
                "completed_sampling_seconds": completed_sampling,
                "rows_aggregated": int(
                    bayes_data["meta"]["_model"]["n_aggregated"]
                ),
                "compression_ratio": float(
                    bayes_data["meta"]["_model"]["compression_ratio"]
                ),
            }
        model_predictors = len(bands)

    meta = {
        "n_candidate_rows": int(len(sampled)),
        "n_used": int(sampled["used"].astype(bool).sum()),
        "n_available_rows": int((~sampled["used"].astype(bool)).sum()),
        "n_strata": None,
        "model_predictors": int(model_predictors),
        "point_sampling": point_meta,
    }
    return meta, diagnostics


def _prepare_sampled_choices(
    *,
    reloc: gpd.GeoDataFrame,
    env,
    root: Path,
    workspace: Path,
    bands: list[str],
    args,
    stages: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with _stage(stages, "structure"):
        movement = fit_movement_kernel_per_id(
            reloc,
            id_col=args.id_col,
            timestamp_col="Timestamp",
            round_freq=None,
            expected_interval_min=60,
            tolerance_min=1,
        )
    with _stage(stages, "availability"):
        choices = build_movement_choice_sets(
            movement,
            id_col=args.id_col,
            n_available=args.n_available,
            speed_margin=1.05,
            burst_gap="2h",
            seed=args.seed,
        )
        choices = _add_start_conditions(choices, env)

    preserve = _choice_preserve_columns(choices, args.id_col)
    frame = _candidate_frame_from_gdf(choices, preserve=preserve)
    with _stage(stages, "point_partitioning"):
        partitions = _write_spatial_partitions(
            frame,
            env,
            workspace / "points",
            grid=args.partition_grid,
        )
    with _stage(stages, "raster_sampling"):
        sampled, point_meta = _sample_bisection_materialized(
            partitions=partitions,
            root=root,
            output_dir=workspace / "sampled",
            bands=bands,
            preserve_cols=["row_id", *preserve],
            raster_gib=args.raster_gib,
            shard_count=args.point_shards,
            threads_per_shard=args.threads_per_shard,
            graph_partitions=args.graph_partitions,
            spatial_chunk=args.spatial_chunk,
            managed_memory_gib=args.managed_memory_gib_per_shard,
            worker_startup_timeout=args.worker_startup_timeout,
        )
    return sampled, {
        "point_sampling": point_meta,
        "n_movement_individuals": int(len(movement["summary"])),
    }


def _run_ssf(
    *,
    inference: str,
    reloc: gpd.GeoDataFrame,
    env,
    root: Path,
    workspace: Path,
    bands: list[str],
    args,
    stages: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sampled, prep_meta = _prepare_sampled_choices(
        reloc=reloc,
        env=env,
        root=root,
        workspace=workspace,
        bands=bands,
        args=args,
        stages=stages,
    )

    with _stage(stages, "design"):
        complete = complete_ssf_strata(
            sampled,
            predictors=bands,
            id_col=args.id_col,
            expected_n_choices=args.n_available + 1,
        )
        scaling = fit_ssf_scaling(complete, bands)
        scaled = apply_ssf_scaling(complete, bands, scaling)
        model_predictors = tuple(f"{band}_z" for band in bands)
        arrays = None
        if inference == "bayesian":
            arrays = build_ssf_choice_arrays(
                scaled,
                id_col=args.id_col,
                predictors=model_predictors,
                dtype="float32",
            )

    if inference == "frequentist":
        stages.setdefault("model_build", 0.0)
        with _stage(stages, "inference"):
            model, result = fit_conditional_ssf(
                scaled,
                predictors=model_predictors,
                id_col=args.id_col,
                engine="fast",
                method="lbfgs",
                maxiter=args.maxiter,
            )
        with _stage(stages, "diagnostics"):
            fit = FrequentistSSFFit(
                model=model,
                result=result,
                data=scaled,
                raw_predictors=tuple(bands),
                predictors=model_predictors,
                scaling=scaling,
                id_col=args.id_col,
                engine="fast",
            )
            scores = fit.choice_scores()["summary"]
            diagnostics = {
                "mean_log_score_gain": float(scores["mean_log_score_gain"]),
                "predictive_advantage": float(scores["predictive_advantage"]),
            }
    else:
        import pymc as pm

        assert arrays is not None
        with _stage(stages, "model_build"):
            model = build_hierarchical_ssf_model(arrays)
        sampling = _sampling_kwargs(
            pm,
            sampler=args.choice_sampler,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.bayesian_cores,
            target_accept=args.choice_target_accept,
            seed=args.seed + 20_000,
        )
        with _stage(stages, "inference"):
            started = time.perf_counter()
            with model:
                raw_idata = pm.sample(**sampling)
            idata = _materialize_inference_tree(raw_idata)
            completed_sampling = time.perf_counter() - started
        with _stage(stages, "diagnostics"):
            fit = BayesianSSFFit(
                model=model,
                idata=idata,
                arrays=arrays,
                data=scaled,
                raw_predictors=tuple(bands),
                predictors=model_predictors,
                scaling=scaling,
                id_col=args.id_col,
            )
            diagnostics = _bayesian_choice_diagnostics(
                idata,
                completed_seconds=completed_sampling,
            )
            score_summary = fit.choice_scores(
                batch_size=args.posterior_score_batch
            )["summary"]
            diagnostics.update(
                {
                    "mean_log_score_gain": float(
                        score_summary["mean_log_score_gain"]
                    ),
                    "predictive_advantage": float(
                        score_summary["predictive_advantage"]
                    ),
                    "completed_sampling_seconds": completed_sampling,
                }
            )

    n_strata = int(
        scaled[[args.id_col, "stratum_id"]].drop_duplicates().shape[0]
    )
    return {
        "n_candidate_rows": int(len(scaled)),
        "n_used": int(scaled["used"].astype(bool).sum()),
        "n_available_rows": int((~scaled["used"].astype(bool)).sum()),
        "n_strata": n_strata,
        "model_predictors": len(model_predictors),
        **prep_meta,
    }, diagnostics


def _run_issf(
    *,
    inference: str,
    reloc: gpd.GeoDataFrame,
    env,
    root: Path,
    workspace: Path,
    bands: list[str],
    args,
    stages: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sampled, prep_meta = _prepare_sampled_choices(
        reloc=reloc,
        env=env,
        root=root,
        workspace=workspace,
        bands=bands,
        args=args,
        stages=stages,
    )
    start_predictors = (
        "start_x_condition",
        "start_y_condition",
        "start_time_sin",
    )

    with _stage(stages, "design"):
        design = prepare_issf_design(
            sampled,
            endpoint_predictors=bands,
            start_predictors=start_predictors,
            directional_predictors=(),
            id_col=args.id_col,
            expected_n_choices=args.n_available + 1,
            movement_terms=(
                "step_length_km",
                "log_step_length",
                "cos_turn_angle",
            ),
            interaction_terms=("step_length_km", "log_step_length"),
            proposal_logpdf_col="proposal_logpdf",
            center_offset=True,
        )
        arrays = None
        if inference == "bayesian":
            arrays = build_ssf_choice_arrays(
                design.data,
                id_col=args.id_col,
                predictors=design.predictors,
                stratum_col=design.stratum_col,
                offset_col=design.offset_col,
                dtype="float32",
            )

    if inference == "frequentist":
        stages.setdefault("model_build", 0.0)
        with _stage(stages, "inference"):
            model, result = fit_conditional_ssf(
                design.data,
                predictors=design.predictors,
                id_col=args.id_col,
                stratum_col=design.stratum_col,
                engine="fast",
                method="lbfgs",
                maxiter=args.maxiter,
                offset_col=design.offset_col,
            )
        with _stage(stages, "diagnostics"):
            fit = FrequentistISSFFit(
                model=model,
                result=result,
                design=design,
                id_col=args.id_col,
            )
            scores = fit.choice_scores()["summary"]
            diagnostics = {
                "mean_log_score_gain": float(scores["mean_log_score_gain"]),
                "predictive_advantage": float(scores["predictive_advantage"]),
            }
    else:
        import pymc as pm

        assert arrays is not None
        with _stage(stages, "model_build"):
            model = build_hierarchical_issf_model(arrays)
        sampling = _sampling_kwargs(
            pm,
            sampler=args.choice_sampler,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.bayesian_cores,
            target_accept=args.choice_target_accept,
            seed=args.seed + 30_000,
        )
        with _stage(stages, "inference"):
            started = time.perf_counter()
            with model:
                raw_idata = pm.sample(**sampling)
            idata = _materialize_inference_tree(raw_idata)
            completed_sampling = time.perf_counter() - started
        with _stage(stages, "diagnostics"):
            fit = BayesianISSFFit(
                model=model,
                idata=idata,
                arrays=arrays,
                design=design,
                id_col=args.id_col,
            )
            diagnostics = _bayesian_choice_diagnostics(
                idata,
                completed_seconds=completed_sampling,
            )
            score_summary = fit.choice_scores(
                batch_size=args.posterior_score_batch
            )["summary"]
            diagnostics.update(
                {
                    "mean_log_score_gain": float(
                        score_summary["mean_log_score_gain"]
                    ),
                    "predictive_advantage": float(
                        score_summary["predictive_advantage"]
                    ),
                    "completed_sampling_seconds": completed_sampling,
                }
            )

    return {
        "n_candidate_rows": int(len(design.data)),
        "n_used": int(design.data["used"].astype(bool).sum()),
        "n_available_rows": int((~design.data["used"].astype(bool)).sum()),
        "n_strata": int(design.diagnostics["n_strata_retained"]),
        "model_predictors": len(design.predictors),
        "iSSF_design": design.diagnostics,
        **prep_meta,
    }, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end workstation benchmark for one hrHSA workflow."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--analysis", choices=("rsf", "ssf", "issf"), required=True)
    parser.add_argument(
        "--inference",
        choices=("frequentist", "bayesian"),
        required=True,
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--individuals", type=int, default=20)
    parser.add_argument("--steps-per-individual", type=int, default=1000)
    parser.add_argument("--n-available", type=int, default=10)
    parser.add_argument("--raster-gib", type=float, default=24.0)
    parser.add_argument("--rsf-predictors", type=int, default=3)
    parser.add_argument("--ssf-predictors", type=int, default=6)
    parser.add_argument("--issf-endpoint-predictors", type=int, default=5)
    parser.add_argument("--partition-grid", type=int, default=10)
    parser.add_argument("--point-shards", type=int, default=6)
    parser.add_argument("--threads-per-shard", type=int, default=2)
    parser.add_argument("--graph-partitions", type=int, default=100)
    parser.add_argument("--spatial-chunk", type=int, default=1024)
    parser.add_argument("--managed-memory-gib-per-shard", type=float, default=16.0)
    parser.add_argument("--worker-startup-timeout", type=float, default=600.0)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--bayesian-cores", type=int, default=4)
    parser.add_argument("--rsf-sampler", default="blackjax")
    parser.add_argument("--choice-sampler", default="nutpie")
    parser.add_argument("--rsf-target-accept", type=float, default=0.9)
    parser.add_argument("--choice-target-accept", type=float, default=0.95)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--posterior-score-batch", type=int, default=250)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--id-col", default="individual-local-identifier")
    parser.add_argument("--memory-sample-interval", type=float, default=0.2)
    args = parser.parse_args()

    positive = (
        "repeat",
        "individuals",
        "steps_per_individual",
        "n_available",
        "rsf_predictors",
        "ssf_predictors",
        "issf_endpoint_predictors",
        "partition_grid",
        "point_shards",
        "threads_per_shard",
        "graph_partitions",
        "spatial_chunk",
        "draws",
        "tune",
        "chains",
        "bayesian_cores",
        "maxiter",
        "posterior_score_batch",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.raster_gib <= 0 or args.managed_memory_gib_per_shard <= 0:
        parser.error("raster and managed-memory sizes must be positive")

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    env, raster_workload = _load_benchmark_env(
        root,
        args.raster_gib,
        args.spatial_chunk,
    )
    available_bands = [str(value) for value in env["band"].values]
    needed = max(
        args.rsf_predictors,
        args.ssf_predictors,
        args.issf_endpoint_predictors,
    )
    if len(available_bands) < needed:
        raise RuntimeError(
            f"E2E benchmark needs at least {needed} raster bands; "
            f"environment has {len(available_bands)}"
        )

    reloc = _synthetic_trajectories(
        env,
        individuals=args.individuals,
        steps_per_individual=args.steps_per_individual,
        seed=args.seed,
        id_col=args.id_col,
    )

    if args.analysis == "rsf":
        bands = available_bands[: args.rsf_predictors]
    elif args.analysis == "ssf":
        bands = available_bands[: args.ssf_predictors]
    else:
        bands = available_bands[: args.issf_endpoint_predictors]

    stages: dict[str, float] = {}
    overall_started = time.perf_counter()
    with _ProcessTreePeakRSS(args.memory_sample_interval) as memory:
        if args.analysis == "rsf":
            meta, diagnostics = _run_rsf(
                inference=args.inference,
                reloc=reloc,
                env=env,
                root=root,
                workspace=workspace,
                bands=bands,
                args=args,
                stages=stages,
            )
        elif args.analysis == "ssf":
            meta, diagnostics = _run_ssf(
                inference=args.inference,
                reloc=reloc,
                env=env,
                root=root,
                workspace=workspace,
                bands=bands,
                args=args,
                stages=stages,
            )
        else:
            meta, diagnostics = _run_issf(
                inference=args.inference,
                reloc=reloc,
                env=env,
                root=root,
                workspace=workspace,
                bands=bands,
                args=args,
                stages=stages,
            )
    wall_seconds = time.perf_counter() - overall_started

    for name in STAGE_ORDER:
        stages.setdefault(name, 0.0)
    stage_sum = float(sum(stages.values()))
    record = {
        "campaign": "end-to-end-workstation-v1",
        "analysis": args.analysis,
        "inference": args.inference,
        "repeat": args.repeat,
        "status": "success",
        "wall_seconds": wall_seconds,
        "stage_seconds": {name: float(stages[name]) for name in STAGE_ORDER},
        "stage_sum_seconds": stage_sum,
        "unattributed_seconds": wall_seconds - stage_sum,
        "peak_process_tree_rss_mb": memory.peak_bytes / 1024**2,
        "memory_samples": memory.samples,
        "git_commit": _git_commit(),
        "raw_relocations": int(len(reloc)),
        "individuals": args.individuals,
        "target_strata": args.individuals * args.steps_per_individual,
        "n_available_per_used_or_stratum": args.n_available,
        "raster_gib": float(raster_workload.logical_gib),
        "raster_shape": [int(env.sizes["y"]), int(env.sizes["x"])],
        "bands": bands,
        "point_policy": "recursive_weighted_spatial_bisection",
        "point_shards": args.point_shards,
        "threads_per_shard": args.threads_per_shard,
        "frequentist_optimizer": "lbfgs",
        "bayesian_sampler": (
            None
            if args.inference == "frequentist"
            else (
                args.rsf_sampler
                if args.analysis == "rsf"
                else args.choice_sampler
            )
        ),
        "draws": args.draws if args.inference == "bayesian" else None,
        "tune": args.tune if args.inference == "bayesian" else None,
        "chains": args.chains if args.inference == "bayesian" else None,
        "bayesian_cores": (
            args.bayesian_cores if args.inference == "bayesian" else None
        ),
        **meta,
        "diagnostics": diagnostics,
    }
    _append_jsonl(output, record)
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
