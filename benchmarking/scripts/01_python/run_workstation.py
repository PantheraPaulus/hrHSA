"""Run a workstation dry-run of the hrHSA HPC benchmark campaign.

The goal is not to emulate a cluster. It exercises the same storage-backed kernels,
strong-scaling logic, worker-memory accounting and process/thread geometry choices
on a substantial single machine before expensive HPC allocations are used.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr

from hsa import FeatureSpec
from hsa.compute import (
    ExecutionConfig,
    append_benchmark_record,
    benchmark_timer,
    make_benchmark_record,
    read_benchmark_records,
    sample_raster_stack_chunked,
    spatial_task_count,
    task_density,
)
from hsa.compute.persistence import cancel_distributed, persist_distributed
from hsa.rsf import predict_rsf_surface_chunked


PROFILES = {
    "quick": {"size": 12_288, "bands": 6, "points": 1_000_000},
    "standard": {"size": 24_576, "bands": 6, "points": 2_000_000},
    "stress": {"size": 32_768, "bands": 6, "points": 5_000_000},
}


def _deterministic_model(env: xr.DataArray):
    bands = [str(value) for value in env.band.values]
    spec = FeatureSpec(
        linear=bands,
        quadratic=[bands[0]],
        interactions=[(bands[0], bands[1])] if len(bands) > 1 else [],
        add_const=True,
    )
    params: dict[str, float] = {"const": -0.5}
    for index, band in enumerate(bands):
        params[band] = 0.08 * (1 if index % 2 == 0 else -1)
    params[f"{bands[0]}__sq"] = 0.02
    if len(bands) > 1:
        params[f"{bands[0]}__x__{bands[1]}"] = -0.015
    model = SimpleNamespace(params=pd.Series(params))
    scaler = SimpleNamespace(
        mean_=np.zeros(len(bands), dtype=float),
        scale_=np.ones(len(bands), dtype=float),
    )
    meta = {"categorical": {}, "columns": list(params)}
    return model, scaler, spec, meta


def _factor_geometries(total_threads: int) -> list[tuple[int, int]]:
    """Return worker/thread factor pairs from process-heavy to thread-heavy."""
    return [
        (workers, total_threads // workers)
        for workers in range(total_threads, 0, -1)
        if total_threads % workers == 0
    ]


def _default_strong_workers(physical_cores: int) -> list[int]:
    # Include 10 cores on 12-core workstations so the saturation region around
    # 8--12 physical cores is sampled more densely for publication figures.
    candidates = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32]
    result = [value for value in candidates if value <= physical_cores]
    if physical_cores not in result:
        result.append(physical_cores)
    return sorted(set(result))


def _memory_limit(total_memory_gib: float, workers: int, managed_fraction: float) -> str:
    managed_total = total_memory_gib * managed_fraction
    per_worker = managed_total / workers
    return f"{per_worker:.2f}GiB"


def _open_inputs(root: Path):
    env = xr.open_zarr(root / "environment.zarr", chunks={})["environment"]
    points = gpd.read_parquet(root / "points.parquet")
    return env, points


def _retire_local_workers(client) -> None:
    """Gracefully retire workers before closing a short-lived LocalCluster."""
    try:
        workers = list(client.scheduler_info().get("workers", {}))
        if workers:
            client.retire_workers(workers=workers, close_workers=True)
    except Exception:
        pass


def _run_configuration(
    *,
    env: xr.DataArray,
    points: gpd.GeoDataFrame,
    output: Path,
    campaign: str,
    workers: int,
    threads_per_worker: int,
    repeats: int,
    chunks: dict[str, int],
    chunk_mb: int,
    total_memory_gib: float,
    managed_memory_fraction: float,
    tmpdir: Path,
    surface_compute_dtype: str,
    include_sampling: bool = True,
    include_surface: bool = True,
) -> None:
    total_execution_threads = workers * threads_per_worker
    memory_limit = _memory_limit(total_memory_gib, workers, managed_memory_fraction)
    task_count = spatial_task_count(env, chunks)
    model, scaler, spec, meta = _deterministic_model(env)

    execution = ExecutionConfig(
        backend="local",
        n_workers=workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit=memory_limit,
        local_directory=str(tmpdir),
        dashboard_address=None,
        chunk_mb=chunk_mb,
        worker_startup_timeout=300,
    )
    client, cluster = execution.create_client()
    if client is None:
        raise RuntimeError("Workstation benchmark requires a Dask client")

    common_metadata = {
        "campaign": campaign,
        "worker_geometry": f"{workers}x{threads_per_worker}",
        "total_execution_threads": total_execution_threads,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "memory_limit_per_worker": memory_limit,
        "task_density_per_worker": task_density(task_count, workers),
        "data_root": str(output.parent.parent / "data"),
        "raster_size_x": int(env.sizes["x"]),
        "raster_size_y": int(env.sizes["y"]),
        "bands": int(env.sizes["band"]),
        "surface_compute_dtype": surface_compute_dtype,
        "memory_monitor_interval_seconds": 0.25,
    }

    try:
        client.wait_for_workers(workers, timeout=300)
        print(
            f"{campaign}: {workers} workers x {threads_per_worker} threads "
            f"({total_execution_threads} execution threads), memory={memory_limit}/worker"
        )

        for repeat in range(1, repeats + 1):
            if include_sampling:
                with benchmark_timer(client=client) as timer:
                    sampled = sample_raster_stack_chunked(
                        points,
                        env,
                        chunks=chunks,
                        client=client,
                    )
                if len(sampled) != len(points):
                    raise RuntimeError("Sampling benchmark returned the wrong row count")
                record = make_benchmark_record(
                    "workstation_zarr_point_sampling",
                    timer["wall_seconds"],
                    rows=len(points),
                    workers=workers,
                    threads_per_worker=threads_per_worker,
                    chunk_mb=chunk_mb,
                    tasks=task_count,
                    bytes_processed=int(env.nbytes),
                    metadata={**common_metadata, "repeat": repeat},
                    operation_memory=timer,
                    client=client,
                )
                append_benchmark_record(record, output)
                del sampled
                gc.collect()

            if include_surface:
                persisted = None
                predicted = None
                try:
                    with benchmark_timer(client=client) as timer:
                        predicted = predict_rsf_surface_chunked(
                            env,
                            model,
                            scaler,
                            spec,
                            meta,
                            chunks=chunks,
                            compute_dtype=surface_compute_dtype,
                        )
                        persisted = persist_distributed(client, predicted.data)

                    cells = int(env.sizes["x"] * env.sizes["y"])
                    output_partitions = int(getattr(persisted, "npartitions", 1))
                    record = make_benchmark_record(
                        "workstation_zarr_surface_prediction",
                        timer["wall_seconds"],
                        rows=cells,
                        workers=workers,
                        threads_per_worker=threads_per_worker,
                        chunk_mb=chunk_mb,
                        tasks=task_count,
                        bytes_processed=int(env.nbytes),
                        metadata={
                            **common_metadata,
                            "repeat": repeat,
                            "materialization": "distributed_persist_no_final_assembly",
                            "output_partitions": output_partitions,
                        },
                        operation_memory=timer,
                        client=client,
                    )
                    append_benchmark_record(record, output)
                finally:
                    cancel_distributed(client, persisted)
                    del persisted, predicted
                    gc.collect()
    finally:
        _retire_local_workers(client)
        client.close()
        if cluster is not None:
            cluster.close()


def _plot_geometry(path: Path, output_dir: Path) -> None:
    if not path.exists():
        return
    df = read_benchmark_records(path)
    if df.empty or "metadata.worker_geometry" not in df:
        return

    aggregations: dict[str, tuple[str, object]] = {
        "median_seconds": ("wall_seconds", "median"),
        "q25_seconds": ("wall_seconds", lambda x: x.quantile(0.25)),
        "q75_seconds": ("wall_seconds", lambda x: x.quantile(0.75)),
        "median_throughput": ("throughput_rows_s", "median"),
        "max_worker_peak_rss_mb": ("worker_peak_rss_max_mb", "max"),
    }
    if "worker_operation_peak_rss_max_mb" in df:
        aggregations["max_worker_operation_peak_rss_mb"] = (
            "worker_operation_peak_rss_max_mb",
            "max",
        )
    if "worker_operation_peak_rss_total_mb" in df:
        aggregations["max_worker_operation_peak_total_rss_mb"] = (
            "worker_operation_peak_rss_total_mb",
            "max",
        )
    if "operation_peak_process_tree_rss_mb" in df:
        aggregations["max_operation_process_tree_rss_mb"] = (
            "operation_peak_process_tree_rss_mb",
            "max",
        )

    grouped = (
        df.groupby(
            [
                "benchmark",
                "metadata.worker_geometry",
                "metadata.total_execution_threads",
            ]
        )
        .agg(**aggregations)
        .reset_index()
    )
    grouped.to_csv(output_dir / "workstation_geometry_summary.csv", index=False)

    for benchmark, subset in grouped.groupby("benchmark"):
        subset = subset.copy()
        subset["workers_order"] = (
            subset["metadata.worker_geometry"].str.split("x").str[0].astype(int)
        )
        subset = subset.sort_values("workers_order", ascending=False)
        labels = subset["metadata.worker_geometry"].tolist()
        x = np.arange(len(labels))
        lower = subset["median_seconds"] - subset["q25_seconds"]
        upper = subset["q75_seconds"] - subset["median_seconds"]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(
            x,
            subset["median_seconds"],
            yerr=np.vstack([lower, upper]),
            marker="o",
            capsize=4,
        )
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_xlabel("Dask geometry: workers x threads/worker")
        ax.set_ylabel("Median wall time (s)")
        ax.set_title(f"{benchmark}: process/thread geometry")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        safe = benchmark.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"{safe}_geometry.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Fast local-disk benchmark root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--storage-chunk", type=int, default=1024)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--managed-memory-fraction", type=float, default=0.75)
    parser.add_argument("--geometry-threads", type=int, default=None)
    parser.add_argument(
        "--surface-compute-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-strong-scaling", action="store_true")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--include-smt", action="store_true")
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if not 0 < args.managed_memory_fraction < 1:
        raise ValueError("--managed-memory-fraction must lie in (0, 1)")

    physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    logical = psutil.cpu_count(logical=True) or physical
    total_memory_gib = psutil.virtual_memory().total / 1024**3
    geometry_threads = physical if args.geometry_threads is None else args.geometry_threads
    if geometry_threads <= 0 or geometry_threads > physical:
        raise ValueError(f"--geometry-threads must lie between 1 and {physical}")

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_root = root / "data"
    tmpdir = root / "dask-tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["TMPDIR"] = str(tmpdir)

    profile = PROFILES[args.profile]
    if not args.skip_prepare:
        prepare_script = Path(__file__).with_name("prepare_data.py")
        command = [
            sys.executable,
            str(prepare_script),
            "--root",
            str(data_root),
            "--size",
            str(profile["size"]),
            "--bands",
            str(profile["bands"]),
            "--storage-chunk",
            str(args.storage_chunk),
            "--points",
            str(profile["points"]),
            "--workers",
            str(min(physical, 8)),
        ]
        if args.force_prepare:
            command.append("--force")
        subprocess.run(command, check=True)

    env, points = _open_inputs(data_root)
    chunks = {"band": -1, "y": args.storage_chunk, "x": args.storage_chunk}
    task_count = spatial_task_count(env, chunks)

    print(
        f"Machine: physical={physical}, logical={logical}, RAM={total_memory_gib:.1f} GiB\n"
        f"Profile={args.profile}, raster={env.sizes['x']}x{env.sizes['y']}x{env.sizes['band']}, "
        f"uncompressed={env.nbytes / 1024**3:.2f} GiB, spatial_tasks={task_count}\n"
        f"Geometry budget={geometry_threads} physical threads, "
        f"surface compute dtype={args.surface_compute_dtype}"
    )

    strong_path = output_dir / "workstation_strong_scaling.jsonl"
    geometry_path = output_dir / f"workstation_geometry_{geometry_threads}t.jsonl"
    smt_path = output_dir / "workstation_smt_geometry.jsonl"
    if not args.skip_strong_scaling and strong_path.exists():
        strong_path.unlink()
    if not args.skip_geometry and geometry_path.exists():
        geometry_path.unlink()
    if args.include_smt and smt_path.exists():
        smt_path.unlink()

    if not args.skip_strong_scaling:
        for workers in _default_strong_workers(physical):
            _run_configuration(
                env=env,
                points=points,
                output=strong_path,
                campaign="workstation_strong_scaling",
                workers=workers,
                threads_per_worker=1,
                repeats=args.repeats,
                chunks=chunks,
                chunk_mb=args.chunk_mb,
                total_memory_gib=total_memory_gib,
                managed_memory_fraction=args.managed_memory_fraction,
                tmpdir=tmpdir,
                surface_compute_dtype=args.surface_compute_dtype,
            )

    if not args.skip_geometry:
        for workers, threads in _factor_geometries(geometry_threads):
            _run_configuration(
                env=env,
                points=points,
                output=geometry_path,
                campaign=f"workstation_physical_core_geometry_{geometry_threads}t",
                workers=workers,
                threads_per_worker=threads,
                repeats=args.repeats,
                chunks=chunks,
                chunk_mb=args.chunk_mb,
                total_memory_gib=total_memory_gib,
                managed_memory_fraction=args.managed_memory_fraction,
                tmpdir=tmpdir,
                surface_compute_dtype=args.surface_compute_dtype,
            )

    if args.include_smt and logical > physical:
        for workers, threads in _factor_geometries(logical):
            if workers > physical:
                continue
            _run_configuration(
                env=env,
                points=points,
                output=smt_path,
                campaign="workstation_smt_geometry",
                workers=workers,
                threads_per_worker=threads,
                repeats=args.repeats,
                chunks=chunks,
                chunk_mb=args.chunk_mb,
                total_memory_gib=total_memory_gib,
                managed_memory_fraction=args.managed_memory_fraction,
                tmpdir=tmpdir,
                surface_compute_dtype=args.surface_compute_dtype,
            )

    plotter = Path(__file__).with_name("plot_results.py")
    if strong_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(plotter),
                str(strong_path),
                "--output-dir",
                str(output_dir / "strong_scaling"),
            ],
            check=True,
        )
    _plot_geometry(geometry_path, output_dir)
    _plot_geometry(smt_path, output_dir)

    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
