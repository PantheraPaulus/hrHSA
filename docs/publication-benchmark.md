# Publication benchmark protocol

This protocol freezes the single-machine reference campaign before hrHSA is scaled
on CoolMUC-4. The goal is to report not only the fastest configuration, but also
strong-scaling efficiency, memory cost, statistical quality, and the point at which
additional CPU resources stop paying off.

## 1. Run the smoke gate first

The smoke gate is deliberately small and should be run after pulling a new
performance branch, changing environments, or moving to a new machine:

```bash
python benchmarks/hpc/run_production_smoke.py \
    --root "$BASE"
```

It creates `$BASE/production-smoke`, then exercises:

- a `2048 x 2048 x 3` Zarr raster with 20,000 point queries;
- one- and two-worker distributed point extraction and persisted surface prediction;
- Newton and L-BFGS frequentist fitting;
- completion-aware PyMC and BlackJAX sampling;
- two-worker frequentist LOIO execution;
- two-worker PyMC and BlackJAX LOIO execution;
- operation-local driver/process-tree and Dask-worker RAM monitoring.

The smoke command exits non-zero on the first failed stage and prints the tail of
that stage's log. A run is considered clean only when
`production-smoke/SUCCESS` exists. `smoke_report.json` records dependency versions,
stage timings, and semantic checks. The checks verify that surface prediction
remains partitioned under distributed persistence, both Bayesian backends complete,
all CV folds succeed, and the new operation-local memory fields are populated.

## 2. Publication-quality workstation run

After the smoke gate passes, run the full suite on a dedicated workstation:

```bash
python benchmarks/hpc/run_production_workstation_suite.py \
    --root "$BASE" \
    --output-dir "$BASE/production-suite" \
    --profile standard \
    --geometry-threads 8 \
    --repeats 3 \
    --frequentist-repeats 5 \
    --bayesian-repeats 3 \
    --cv-bayesian-repeats 3 \
    --reuse-raster-data
```

Add `--include-smt` for the supplementary logical-CPU experiment. Use `--resume`
to continue after an interrupted stage; completed stages contain `SUCCESS` markers.

For a 12-physical-core workstation, physical-core strong scaling is sampled at

```text
1, 2, 4, 6, 8, 10, 12 workers
```

rather than stopping at eight workers. The denser 8--12-core region is intentional:
that is where the raster kernels previously approached their bandwidth/scheduler
ceiling. The suite also records the fixed 8-thread process/thread geometry and a
second full-physical-core geometry. SMT, when requested, is kept as a separate
campaign and must not be mixed into the physical-core strong-scaling curve.

Frequentist inference uses five repeats because individual fits take only seconds
and are more sensitive to operating-system noise. Bayesian single-model and CV
campaigns use three repeats for the publication run. Inner-core Bayesian sensitivity
at `cores=1` and `cores=2` remains a diagnostic one-repeat experiment because those
points can be substantially more expensive and are not the primary backend
comparison.

## 3. Primary performance metrics

For each scaling point report, where applicable:

| Quantity | Interpretation |
| --- | --- |
| wall time | elapsed time to complete the defined workload |
| throughput | points/s, raster cells/s, or completed folds/s |
| speedup | `T1 / Tn` for a fixed workload |
| parallel efficiency | `(T1 / Tn) / n` |
| core-seconds/fold | allocation cost of one completed CV fold |
| operation-local peak RAM | maximum sampled resident memory during that operation |
| max worker RAM | largest operation-local Dask-worker RSS |
| total worker RAM | sum of operation-local Dask-worker RSS across workers |
| load time | time spent reading prepared fold partitions |
| ESS/s | effective posterior sample size per completed sampling second |
| R-hat/divergences | statistical-quality guardrails for sampler comparisons |

Runtime alone is insufficient. A configuration that is two percent faster while
using fifty percent more cores or RAM may not be the preferred production setting.
Resource-efficiency curves should therefore accompany the fastest-wall-time result.

## 4. RAM semantics

hrHSA retains the historical memory fields for backward compatibility:

```text
peak_rss_mb                    process-lifetime high-water mark
current_rss_mb                 post-operation current RSS
worker_peak_rss_*              worker process-lifetime high-water mark
worker_current_rss_*           post-operation worker RSS
```

Publication runs additionally record operation-local sampled maxima:

```text
operation_peak_rss_mb
operation_peak_process_tree_rss_mb
worker_operation_peak_rss_total_mb
worker_operation_peak_rss_max_mb
memory_samples
```

`operation_peak_process_tree_rss_mb` includes child processes and is therefore
important for PyMC multi-process chain execution. Dask worker RAM is recorded
separately because workers on an HPC cluster are not descendants of the benchmark
coordinator. The monitor samples every 0.25 s by default; this interval is stored in
benchmark metadata. These sampled maxima, not lifetime high-water marks inherited
from previous repeats, should be used for scaling figures.

On CoolMUC-4 also archive Slurm `MaxRSS`/job accounting as an independent scheduler
measurement.

## 5. Bayesian timing

JAX CPU execution is asynchronous. A BlackJAX fold is not considered complete when
`pm.sample()` merely returns a JAX-backed object. Publication timing uses

```text
dispatch_seconds
materialize_seconds
completed_sampling_seconds = dispatch + materialize
diagnostics_seconds
```

and sampler throughput/ESS per second is calculated from
`completed_sampling_seconds`. Host NumPy materialization is also inside the
operation-memory interval, so deferred JAX work cannot disappear from either the
time or RAM measurement.

## 6. CV scaling

The fixed workload is a fixed set of held-out folds. Only the outer Dask worker
count changes. On a 12-core workstation the publication sweep is

```text
1, 2, 4, 6, 8, 10, 12 concurrent folds
```

for frequentist, PyMC, and BlackJAX campaigns, bounded by the number of available
folds. Frequentist folds use L-BFGS with one BLAS thread per fit. Bayesian folds use
one inner PyMC core by default so outer and inner parallelism do not multiply
unintentionally.

Interpret the CV scaling table using both wall-time speedup and
`median_core_seconds_per_fold`. `median_load_seconds` diagnoses storage fan-out:
if fit/sampling time remains stable while load time grows with worker count, the
shared filesystem rather than inference is becoming the scaling bottleneck.

The summary additionally reports operation-local worker RAM and, when available,
fold throughput per GiB of worker memory.

## 7. Reproducibility archive

Archive at minimum:

- the JSONL/raw CSV benchmark records;
- all summary CSVs and generated plots;
- `machine.json`, the workstation probe, and `suite_manifest.json`;
- git commit SHA and working-tree status;
- Python and dependency versions;
- CPU topology, physical/logical core count, and RAM;
- the exact benchmark command line;
- CoolMUC Slurm job scripts, allocation metadata, and scheduler accounting.

For figures, prefer medians with IQR across repeats. Rerun a decisive or unexpectedly
noisy scaling point five times rather than smoothing or discarding it.
