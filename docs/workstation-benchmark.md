# Workstation dry-run benchmark

Before using CoolMUC-4 allocations, hrHSA can exercise the same storage-backed
kernels and much of the same benchmark instrumentation on a substantial Linux
workstation. This is not a substitute for the HPC campaign: it cannot reproduce
CoolMUC's dual-socket topology, DSS filesystem or HDR InfiniBand. It is, however,
a useful algorithmic and execution calibration.

## What the workstation run tells us

The local campaign measures:

- Zarr-backed chunk-routed point extraction;
- fused blockwise RSF surface prediction without gathering the raster to the
  driver;
- strong scaling across physical CPU cores;
- Dask process/thread geometry at a configurable fixed concurrency;
- optional SMT/logical-CPU geometry;
- driver and worker lifetime peak RSS plus current RSS snapshots;
- task density, wall time, throughput and parallel efficiency.

It therefore answers several questions before CoolMUC is available: whether the
optimized kernels scale at all on real hardware, where scheduler overhead begins
to dominate, whether fewer threaded worker processes outperform one process per
core, and whether memory or local-storage I/O becomes limiting.

## Probe the machine

From the repository root:

```bash
python benchmarks/hpc/probe_workstation.py > workstation-probe.json
```

The probe records physical/logical CPU counts, CPU affinity, RAM, `lscpu`, NUMA
information when available, local filesystem/block-device information and NumPy's
runtime SIMD/BLAS dispatch. Archive it with the benchmark results.

## Recommended run for a 12-core / 128 GiB workstation

Use a fast local SSD/NVMe path, not a network mount. For example:

```bash
python benchmarks/hpc/run_workstation.py \
    --root /fast/local/hrhsa-benchmark \
    --output-dir /fast/local/hrhsa-benchmark-results \
    --profile standard \
    --repeats 3
```

The `standard` profile contains:

- `24576 x 24576 x 6` float32 raster;
- about 13.5 GiB uncompressed raster input;
- `1024 x 1024` storage/computational chunks;
- 576 spatial chunks;
- 2,000,000 random point queries.

On a 12-physical-core machine the strong-scaling sequence is automatically

```text
1, 2, 4, 6, 8, 12 workers x 1 thread
```

and the default fixed-12-core worker-geometry sequence is

```text
12 x 1
 6 x 2
 4 x 3
 3 x 4
 2 x 6
 1 x 12
```

where the first number is worker processes and the second is threads per worker.
All default geometry runs therefore expose exactly 12 Dask execution threads.

The total geometry budget can now be changed independently of the machine size.
For example, after strong scaling identifies eight execution threads as a likely
surface-prediction optimum, compare process/thread layouts at exactly that budget:

```bash
python benchmarks/hpc/run_workstation.py \
    --root /fast/local/hrhsa-benchmark \
    --output-dir /fast/local/hrhsa-benchmark-results-8t \
    --profile standard \
    --skip-prepare \
    --skip-strong-scaling \
    --geometry-threads 8 \
    --repeats 3
```

This runs:

```text
8 x 1
4 x 2
2 x 4
1 x 8
```

and avoids confusing the best *total concurrency* with the best *worker geometry*.

By default hrHSA assigns only 75% of installed RAM to Dask workers. On a 128 GiB
host this leaves roughly 32 GiB outside Dask for the operating system, page cache,
the benchmark coordinator and other services. Override with
`--managed-memory-fraction` only when appropriate for the machine.

## Surface arithmetic precision

The accelerated fused surface kernel now performs intermediate arithmetic in the
requested output dtype by default. Normal `float32` surface output therefore stays
float32 inside each block instead of silently promoting every raster layer and
intermediate to float64.

Benchmark the high-precision path explicitly when useful:

```bash
python benchmarks/hpc/run_workstation.py \
    --root /fast/local/hrhsa-benchmark \
    --output-dir /fast/local/hrhsa-benchmark-results-f64 \
    --profile standard \
    --skip-prepare \
    --surface-compute-dtype float64 \
    --repeats 3
```

The default is `--surface-compute-dtype float32`. Correctness tests compare the
float32 accelerated output with the reference implementation and retain an explicit
float64 mode for analyses that require double-precision intermediate arithmetic.

## Dataset profiles

Three reusable profiles are available:

| Profile | Raster | Approx. uncompressed raster | Spatial chunks | Points |
| --- | --- | ---: | ---: | ---: |
| `quick` | `12288 x 12288 x 6` | 3.4 GiB | 144 | 1,000,000 |
| `standard` | `24576 x 24576 x 6` | 13.5 GiB | 576 | 2,000,000 |
| `stress` | `32768 x 32768 x 6` | 24 GiB | 1024 | 5,000,000 |

Start with `standard`. Use `quick` for development/debugging. Use `stress` after a
successful standard run if the workstation can be dedicated to the benchmark; it
matches the CoolMUC medium raster dimensions and point count used by the HPC
campaign.

Prepared input is reused automatically on subsequent runs. Use `--skip-prepare`
to guarantee that an existing dataset is reused or `--force-prepare` to rebuild it.

## Optional SMT experiment

The physical-core baseline is intentionally kept separate from simultaneous
multithreading. On a 12-core / 24-thread CPU, test SMT explicitly with:

```bash
python benchmarks/hpc/run_workstation.py \
    --root /fast/local/hrhsa-benchmark \
    --output-dir /fast/local/hrhsa-benchmark-results-smt \
    --profile standard \
    --skip-prepare \
    --include-smt
```

This adds logical-CPU geometries such as `12 x 2` and `6 x 4`. Do not mix these
points into the physical-core strong-scaling curve; SMT is a separate architectural
experiment.

## Outputs

The output directory contains, depending on the requested campaigns:

```text
workstation_strong_scaling.jsonl
workstation_geometry_12t.jsonl      # default 12-core geometry budget
workstation_geometry_8t.jsonl       # example custom geometry budget
workstation_smt_geometry.jsonl      # when requested
strong_scaling/
    benchmark_raw.csv
    benchmark_summary.csv
    *_runtime.png
    *_speedup.png
    *_efficiency.png
workstation_geometry_summary.csv
*_geometry.png
```

JSONL files retain per-repeat timings and memory statistics. `peak_rss` fields are
process-lifetime peaks; the newer `current_rss` fields are snapshots taken when the
benchmark record is created and should be used when distinguishing current memory
occupancy from a peak reached earlier in the worker lifetime.

The strong-scaling plots use the same machinery as the CoolMUC campaign, making the
result structure directly comparable later.

## Inference benchmark

Raster scaling and model inference are intentionally separated. After the spatial
campaign, use `benchmarks/hpc/benchmark_inference.py` to measure frequentist design
matrix construction and logistic optimization, or Bayesian aggregation/model build
and NUTS efficiency. See {doc}`inference-performance` for the full protocol.

## Interpretation

The most transferable workstation findings are algorithmic:

- whether chunk routing materially accelerates extraction;
- whether fused prediction scales across cores;
- whether one-process-per-core creates excessive Dask overhead;
- the chunk/task density at which parallel execution stops paying;
- memory scaling and spill behavior;
- whether float32 intermediate arithmetic materially relieves a memory-bandwidth
  ceiling;
- which total concurrency and process/thread geometry are independently optimal.

Do **not** transfer workstation local-SSD throughput, Zarr-sharding performance or
inter-process scaling ratios directly to CoolMUC DSS/InfiniBand. Those must still
be measured on the target system.

If the workstation geometry clearly favors a configuration such as `6 x 2` or
`4 x 3`, that is useful evidence that the CoolMUC geometry sweep should emphasize
mixed process/thread layouts rather than assuming `112 x 1`.
