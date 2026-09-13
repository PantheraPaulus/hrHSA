# Cross-validation performance

For expensive validation, hrHSA treats the **outer fold** as the primary unit of
parallelism. Independent leave-one-individual-out (LOIO) fits are generally better
suited to horizontal scaling than assigning a large native thread pool to one
low-dimensional statistical fit.

The high-performance path has two stages:

1. prepare each individual's used/available environmental samples once with
   `analysis.prepare(...)`; and
2. run held-out folds from those cached Parquet partitions, optionally with a Dask
   client.

Compare serial and distributed execution using the **same prepared dataset** when
validating execution equivalence.

## Frequentist LOIO

Workstation benchmarks with two million rows showed that L-BFGS reduced optimizer
latency relative to Newton while several BLAS threads produced little end-to-end
benefit. The throughput path therefore keeps one BLAS thread per concurrent fold:

```python
from hsa.rsf import LeaveOneIndividualOut

prepared = analysis.prepare(
    "prepared-rsf",
    sampling_factor=10,
    thin_dt=None,
    client=client,
)

scheme = LeaveOneIndividualOut(
    sampling_factor_train=10,
    thin_train_dt=None,
    thin_test_dt=None,
)

cv = analysis.validate(
    scheme,
    prepared=prepared,
    client=client,
    fit_method="lbfgs",
    fit_kwargs={"maxiter": 200},
    blas_threads=1,
)
```

Keep Newton as a scientific/reference optimizer and verify L-BFGS convergence,
coefficients and likelihood agreement before adopting it for a new model family.

## Bayesian LOIO

Prepared Bayesian LOIO also accepts `client=`:

```python
cv = bayes_analysis.validate(
    scheme,
    prepared=prepared,
    client=client,
    nuts_sampler="blackjax",
    sample_kwargs={
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
    },
)
```

When a Dask client is supplied, hrHSA defaults to `cores=1` **inside each fold**
unless explicitly overridden. This prevents each Dask worker from spawning another
large PyMC process pool. Native BLAS/OpenMP threads should also remain capped.

A useful conceptual resource constraint is

\[
N_{\mathrm{concurrent\ folds}}\,
N_{\mathrm{cores\ per\ fold}}
\leq N_{\mathrm{allocated\ physical\ cores}}.
\]

For BlackJAX this is only an approximation: `cores` controls PyMC process behavior
but does not guarantee a hard JAX/XLA CPU-thread limit. BlackJAX concurrency must
therefore be benchmarked empirically under Slurm CPU binding.

## Completion-aware Bayesian timing

`pm.sample()` can return JAX-backed arrays while queued computation is still
outstanding. hrHSA therefore materializes JAX posterior/sample-stat arrays before a
fold is considered complete. Backend comparisons must use:

- completed inference time (dispatch + materialization);
- ESS per completed second;
- R-hat/divergence diagnostics; and
- posterior agreement.

Do not use dispatch-only BlackJAX timing as a speedup metric.

## Keep fold results compact

Distributed Bayesian CV defaults to compact fold state:

```text
store_scores=False
keep_idata=False
keep_model=False
keep_bayes_data=False
```

Opt in only when the next analysis stage needs those objects. Posterior score
matrices can become much larger than the scalar validation summary.

## Benchmark outer-fold scaling

`benchmarks/hpc/benchmark_cv_scaling.py` constructs a deterministic
PreparedDataset-like Parquet cache and times a fixed set of held-out folds.
Cluster startup and cache construction are outside the timed campaign.

A workstation frequentist point can be run directly:

```bash
python benchmarks/hpc/benchmark_cv_scaling.py frequentist \
    --output cv-frequentist.jsonl \
    --work-dir /tmp/hrhsa-cv \
    --workers 8 \
    --folds 24 \
    --rows-per-individual 5000 \
    --predictors 6 \
    --method lbfgs \
    --blas-threads 1 \
    --repeats 3
```

The Bayesian form additionally records dispatch and materialization:

```bash
python benchmarks/hpc/benchmark_cv_scaling.py bayesian \
    --output cv-blackjax.jsonl \
    --work-dir /tmp/hrhsa-cv \
    --workers 4 \
    --folds 24 \
    --rows-per-individual 500 \
    --predictors 3 \
    --sampler blackjax \
    --draws 250 --tune 250 --chains 4 --cores 1
```

These direct Python commands are appropriate on a workstation or inside an
allocated compute job. **Do not run them directly on an LRZ login node.**

Primary metrics are:

\[
\mathrm{fold\ throughput}=\frac{N_{\mathrm{completed\ folds}}}{T_{\mathrm{wall}}}
\]

and

\[
\mathrm{allocated\ core\ seconds/fold}
=\frac{T_{\mathrm{wall}}N_{\mathrm{workers}}}{N_{\mathrm{completed\ folds}}}.
\]

The benchmark also records median fold/load/fit or completed-sampling times and
operation-local process/Dask-worker memory.

## CoolMUC-4: LRZ-compliant campaign

Current LRZ production rules direct 1-16 core jobs to `serial_std`; `cm4_tiny` is
for 17-112 physical cores on one node and `cm4_std` for 2-4 complete nodes.
Accordingly, the CoolMUC CV experiment does **not** attempt a 1-worker production
baseline on the Sapphire-Rapids cm4 partition and does not reserve 112 cores while
running only a handful of workers.

Instead, each CV concurrency point gets its own allocation containing exactly the
requested number of worker CPUs. Generate and submit the concrete scripts with:

```bash
bash benchmarks/hpc/submit_coolmuc4_cv_scaling.sh frequentist
bash benchmarks/hpc/submit_coolmuc4_cv_scaling.sh bayesian pymc
bash benchmarks/hpc/submit_coolmuc4_cv_scaling.sh bayesian blackjax
```

The default publication points are:

```text
28, 56, 112 workers
```

against 112 folds. These produce exactly 4, 2 and 1 scheduling waves. The generated
Slurm files contain their own resource requests rather than relying on `sbatch`
command-line overrides, and are retained under the benchmark submission directory.

The template job itself is a valid 28-core point and can be used as a first smoke:

```bash
sbatch benchmarks/hpc/coolmuc4_cv_scaling.sbatch frequentist
```

For Bayesian runs:

```bash
sbatch benchmarks/hpc/coolmuc4_cv_scaling.sbatch bayesian pymc
sbatch benchmarks/hpc/coolmuc4_cv_scaling.sbatch bayesian blackjax
```

Each Slurm job measures **one** worker count. This is deliberate: a former launcher
that allocated a full 112-core node and swept 1/2/4/... workers inside that
allocation was retired because it intentionally left allocated CPU resources idle
and conflicts with LRZ's resource-use policy.

## Interpreting the cluster results

Do not combine workstation and CoolMUC speedups into one normalized curve. They are
separate hardware experiments:

- workstation: low-core 1-12 strong scaling;
- `cm4_tiny`: high-concurrency single-node scaling, normalized to its smallest
  measured legal production point;
- `cm4_std`: multi-node scaling when justified by the single-node result.

For PyMC, outer-fold scaling is usually close to process-level scaling when each
fold uses `cores=1`. For BlackJAX, inspect both campaign throughput and individual
fold time: increasing fold time with concurrency is evidence of JAX/XLA or shared
memory-bandwidth contention even when batch latency improves.

If load time rises with workers, investigate shared-storage contention. If loading
is stable while fit/sampling time rises, the statistical kernel or memory subsystem
is contending.

## LRZ operational rules

Production CV should be submitted with `sbatch`, not started on login nodes. Avoid
high-frequency `squeue`/`sacct` polling; LRZ recommends monitoring on much longer
intervals (roughly ten minutes for automated/manual `watch`). The hrHSA scripts do
poll their own private Dask scheduler during worker startup/retirement, but those
calls never query Slurm and are unrelated to LRZ scheduler polling limits.
