# CoolMUC-4 benchmark campaign

This page defines the concrete KONWIHR performance campaign for hrHSA on LRZ
CoolMUC-4. It is intentionally staged so that expensive resources are requested
only after smaller runs show that additional parallelism is useful.

CoolMUC-4 nodes provide 112 CPU cores and 512 GB RAM. LRZ documents `cm4_tiny` for
single-node parallel/shared-memory work and `cm4_std` for full-node parallel work;
`serial_*` is intended for serial/single-core jobs and is therefore not used for
this Dask scaling campaign. Large temporary benchmark data belong on
`$SCRATCH_DSS`; legacy `$SCRATCH` is not available on CoolMUC-4.

## What changed after the CI smoke benchmark

The GitHub Actions smoke benchmark identified four practical safeguards that are
now built into the benchmark path:

1. **Storage-backed input.** Distributed runs open one chunked Zarr store instead
   of embedding a large NumPy raster in every Dask graph.
2. **No distributed surface gather.** Surface timing waits for the distributed
   future to finish but does not call `.result()` and transfer the complete
   prediction back to the driver.
3. **Task-density guard.** The benchmark reports spatial tasks per worker and, by
   default, refuses configurations with fewer than eight independent chunk tasks
   per worker.
4. **Scheduler reuse and memory accounting.** A client is reused across repeated
   measurements at one worker count, and JSONL records include both driver RSS and
   aggregate/max Dask-worker peak RSS.

The eight-tasks-per-worker rule is a conservative benchmark heuristic, not a
scientific requirement. It exists to prevent a tiny graph from being presented as
an HPC scaling result.

## Environment

From the repository checkout:

```bash
pip install -e ".[hpc,io]"
```

The supplied CoolMUC scripts look for an environment named `hsa` under Miniforge,
Mambaforge or micromamba. If your environment has another name, set
`HRHSA_CONDA_ENV` or adapt `activate_hrhsa()` in
`benchmarks/hpc/coolmuc4_common.sh`.

All batch scripts set native numerical libraries to one thread per Dask worker:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Dask spill directories use node-local `$TMPDIR`; environmental Zarr data and JSONL
results remain on shared `$SCRATCH_DSS`.

## Stage 0: correctness gate

Do not spend cluster time until the focused HPC CI workflow is green:

```bash
pytest -q tests/test_hpc_*.py
```

CI also compiles the benchmark Python scripts, checks the shell/SBATCH files with
`bash -n`, and imports each benchmark CLI via `--help`.

## Stage 1: prepare the medium input once

Submit from the repository root:

```bash
sbatch benchmarks/hpc/coolmuc4_prepare_medium.sbatch
```

This is an 8-core `cm4_tiny` preparation job. The medium dataset is:

- raster: `32768 x 32768 x 6`, float32;
- uncompressed raster size: about 24 GiB;
- Zarr storage chunk: `1024 x 1024` with all six bands together;
- spatial chunks: 1024;
- random point queries: 5,000,000.

It is written to

```text
$SCRATCH_DSS/hrhsa-benchmark-medium/
```

unless `HRHSA_DATA_ROOT` is changed in `coolmuc4_common.sh`.

Preparation uses a Dask threaded scheduler constrained to the eight CPUs allocated
by the batch job, so data generation itself does not oversubscribe the node.

## Stage 2: controlled 1-8 worker baseline on one node

Run one 8-core `cm4_tiny` allocation and execute the identical workload with 1, 2,
4 and 8 local Dask workers:

```bash
sbatch benchmarks/hpc/coolmuc4_low_scaling.sbatch
```

The allocation remains fixed at eight CPUs throughout, but only the requested
number of Dask workers is active for each timing. Keeping these measurements in the
same allocation reduces node-to-node and filesystem noise and gives the cleanest
low-worker strong-scaling baseline.

Both kernels are measured three times:

- chunk-routed Zarr point sampling;
- fused Zarr surface prediction.

This stage answers the first important question: **is this workload large enough
for distributed scheduling to become worthwhile before eight workers?**

## Stage 3: 16-112 workers on one CoolMUC-4 node

For larger worker pools, request only the cores and memory actually used:

```bash
bash benchmarks/hpc/submit_coolmuc4_tiny_scaling.sh
```

It submits separate `cm4_tiny` jobs for

```text
16, 32, 64, 112 workers
```

with four GB of requested node memory per worker and a three-GB Dask worker memory
limit. Each job repeats both kernels three times against the same medium Zarr store.
The 8-worker point comes from Stage 2, so no duplicate job is necessary.

At 112 workers the task density is still approximately

```text
1024 chunks / 112 workers = 9.14 chunks per worker
```

which stays just above the default task-density guard.

Because Stage 2 and Stage 3 both run on `cm4_tiny`, the 1-112 worker curve remains
within the same CoolMUC-4 partition. The 1/2/4/8 measurements are more tightly
controlled because they share one allocation; the higher points necessarily come
from separate allocations so their three repeats should be retained in the plot.

## Stage 4: prepare the large multi-node input only if needed

If surface prediction is still improving near 112 workers, prepare the larger
input:

```bash
sbatch benchmarks/hpc/coolmuc4_prepare_large.sbatch
```

This is a 32-core `cm4_tiny` preparation job. The large dataset is:

- raster: `65536 x 65536 x 6`, float32;
- uncompressed raster size: about 96 GiB;
- spatial chunks: 4096 at `1024 x 1024`;
- point table: only a tiny placeholder because the multi-node campaign is
  surface-only.

The default location is

```text
$SCRATCH_DSS/hrhsa-benchmark-large/
```

This gives approximately nine spatial tasks per worker even at 447 workers.

## Stage 5: 1/2/4 full-node surface scaling

Submit the full-node campaign with

```bash
bash benchmarks/hpc/submit_coolmuc4_multinode_surface.sh
```

The jobs request 1, 2 and 4 complete `cm4_std` nodes. `cm4_std` requires its
matching QoS, which is included in the supplied batch script. Within each allocation
the script starts Dask directly with `srun` rather than submitting nested SLURM
jobs, so scheduler/worker queue startup is outside the timed kernel.

One CPU on the first node is reserved for the Dask scheduler. Consequently the
actual measured worker counts are:

| Nodes | Worker CPUs | Scheduler CPUs | Total allocated cores |
| ---: | ---: | ---: | ---: |
| 1 | 111 | 1 | 112 |
| 2 | 223 | 1 | 224 |
| 4 | 447 | 1 | 448 |

This is preferable to reporting 112/224/448 workers while silently oversubscribing
the scheduler node.

Only fused surface prediction is benchmarked across nodes. The point-sampling API
returns one pandas table to the driver, so beyond one node its final gather is an
intentional centralized part of the API and would dominate a raw extraction
strong-scaling experiment. Multi-node extraction should instead be evaluated in an
end-to-end prepared-dataset workflow where outputs are partitioned and reused.

## Result files

Every job writes machine-readable JSONL records under

```text
$SCRATCH_DSS/hrhsa-benchmark-results/
```

Records include:

- git SHA and Python/platform information;
- SLURM job/partition metadata when available;
- requested and observed Dask workers;
- wall time and row/cell throughput;
- approximate uncompressed input throughput;
- spatial task count and tasks per worker;
- driver peak RSS;
- total and maximum worker peak RSS;
- benchmark-specific metadata and repeat number.

Because `$SCRATCH_DSS` is temporary storage, copy the final JSONL records and plots
to persistent project storage after the campaign.

## Plot and summarize

After the jobs finish:

```bash
python benchmarks/hpc/plot_results.py \
    "$SCRATCH_DSS"/hrhsa-benchmark-results/*.jsonl \
    --output-dir "$SCRATCH_DSS"/hrhsa-benchmark-plots
```

The script writes a raw CSV, an aggregated CSV and separate wall-time, speedup and
parallel-efficiency plots for each kernel. Repeats are summarized with the median
and interquartile range.

Scaling baselines are calculated separately by campaign/partition. For the main
single-node curve, Stage 2 provides the 1-worker baseline. The full-node multi-node
series is a separate scaling experiment because its smallest feasible point is 111
workers on one complete node.

## Cache interpretation

Do not label repeat 1 as a strict "cold-cache" run. Linux page cache and the shared
parallel filesystem cannot be flushed by an ordinary user, and surface prediction
may encounter data already touched by another benchmark. Instead report:

- the three individual repeats;
- their median and IQR;
- whether repeat 1 differs systematically from repeats 2-3.

A true cold-I/O experiment should use a dataset substantially larger than available
cache or an LRZ-approved cache-control method. The normal KONWIHR scaling figure
should not claim cache state that was not controlled.

## Additional SLURM accounting

After completion, retain scheduler-level accounting alongside the JSONL records:

```bash
sacct -M cm4 \
    -o jobid,nnodes,ntasks,start,elapsed,maxrss,state,exitcode,nodelist
```

This provides an independent job-level memory/runtime check. Avoid frequent polling
of `squeue`; LRZ recommends low-frequency scheduler queries.

## Decision rule before the multi-node run

The multi-node stage is worth running when the 64 -> 112 worker surface result still
shows useful improvement and worker memory/spill behavior is healthy. If the curve
has already flattened, stop there and report the single-node optimum rather than
consuming additional nodes merely to demonstrate scheduler overhead.
