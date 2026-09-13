# Inference performance benchmarking

The raster benchmark campaign measures environmental extraction and RSF surface
projection. Statistical inference is benchmarked separately so filesystem and Dask
behavior do not obscure the cost of building and solving the statistical model.
Cross-validation scaling is then benchmarked as a third layer; see
{doc}`cv-performance`.

## Frequentist logistic RSF

The frequentist benchmark separates two stages:

1. design-matrix construction (`StandardScaler`, quadratic/interactions and
   categorical encoding), and
2. `statsmodels.Logit` optimization.

Run, for example:

```bash
python benchmarks/hpc/benchmark_inference.py frequentist \
    --output-dir inference-results/frequentist \
    --rows 2000000 \
    --predictors 6 \
    --methods newton,lbfgs,bfgs \
    --blas-threads 1,2,4,8,12 \
    --repeats 3
```

The same design matrix is used for every optimizer within a replicate. The first
optimizer in `--methods` is treated as the numerical reference and later methods
record their maximum absolute coefficient difference from it.

`fit_rsf()` and the stateful `FrequentistRSF.fit()` expose Statsmodels' optimizer
choice directly:

```python
fit = analysis.fit(
    prepared=prepared,
    method="lbfgs",
    fit_kwargs={"maxiter": 200},
)
```

The scientific default remains `method="newton"`. A production benchmark on a
large, low-dimensional continuous RSF found L-BFGS faster while agreeing closely
with Newton in the fitted coefficients. Treat L-BFGS as a large-data option only
after convergence and coefficient/log-likelihood agreement have been verified for
the analysis at hand.

The continuous-only design-matrix path is preallocated in NumPy and constructs one
DataFrame at the end. Categorical specifications retain the reference pandas
encoding path. This targets the observation that, once L-BFGS is used,
design-matrix construction can cost as much as or more than the optimizer.

The `--blas-threads` sweep is intentionally independent of Dask. Low-dimensional
logistic MLE gained little from large BLAS thread pools in the workstation campaign.
The high-throughput CV path therefore uses one BLAS thread per fold by default and
parallelizes independent folds instead.

## Frequentist SSF/iSSF individual fitting

The specialized conditional-logit engine is the production path; Statsmodels is a
reference implementation for numerical agreement.  Workstation production
benchmarks showed that independent individual fits benefit from a small thread
pool, while process-based execution pays substantial dataframe/model serialization
and memory overhead.

Keep `workers=1` as the composable package default.  On the 12-core workstation,
the measured local production policy is:

```python
individual = analysis.fit_individuals(
    engine="fast",
    method="lbfgs",
    workers=4,
    executor="thread",
    native_threads=1,
)
```

The same geometry applies to `fit_issf_per_id(...)`.  Do not nest this local thread
pool inside a Dask/Slurm layer that already distributes independent models or
folds; use `workers=1` inside the outer task instead.

## Hierarchical Bayesian RSF

Bayesian performance has different useful metrics. A NUTS chain is sequential, but
multiple chains and multiple model/fold fits can run independently. Wall time alone
is insufficient: sampler quality per unit time matters.

A broad workstation comparison can be run with:

```bash
python benchmarks/hpc/benchmark_inference.py bayesian \
    --output-dir inference-results/bayesian \
    --rows 100000 \
    --predictors 3 \
    --individuals 20 \
    --bin-width 0.5 \
    --samplers pymc,nutpie,numpyro,blackjax \
    --draws 1000 \
    --tune 1000 \
    --chains 4 \
    --cores 4
```

Optional NUTS backends that fail to initialize or sample are written explicitly as
failure records containing the backend, elapsed time, exception type and exception
message. A missing row must never be interpreted as a successful run.

The benchmark records, where available:

- raw and aggregated observation counts and compression ratio;
- data-preparation and model-build wall time;
- sampling/dispatch wall time;
- posterior size and whether observation-level `eta` was stored;
- bulk/tail ESS, R-hat, divergences, tree depth and NUTS steps;
- ESS/s and raw draws/s.

ESS/s remains the primary sampler-efficiency quantity:

\[
\mathrm{ESS/s} = \frac{\mathrm{effective\ sample\ size}}{\mathrm{completed\ wall\ time}}.
\]

A fast backend is not accepted on speed alone. Convergence, divergences and
posterior agreement must also be checked.

## Completion-aware timing for JAX backends

JAX execution is asynchronous in places: returning a JAX-backed array does not by
itself prove that all queued CPU work is complete. The BlackJAX investigation
showed that timing only the return from `pm.sample()` can therefore measure dispatch
latency rather than completed inference.

The completion-aware validation tooling separates:

- `dispatch_seconds`: time until the sampling call returns;
- `materialize_seconds`: synchronization plus conversion of posterior/sample
  statistics to host NumPy arrays;
- `completed_sampling_seconds`: dispatch + materialization;
- `diagnostics_seconds`: ArviZ/hrHSA diagnostics after materialization.

Only `completed_sampling_seconds` should be used for backend speedup and ESS/s
claims. The production Bayesian CV path follows the same rule and materializes JAX
output inside each worker before declaring a fold complete.

JAX/XLA flags, when used, must be configured before JAX initializes and their exact
behavior can depend on the installed JAX/jaxlib version. For CPU HPC benchmarking,
start from explicit Slurm CPU binding and measure completed throughput rather than
assuming that PyMC's `cores` setting controls XLA threads.

## JAX chain execution is not PyMC `cores`

For native PyMC sampling, `cores` controls how many chains are run concurrently.
For PyMC's JAX-backed BlackJAX/NumPyro path, chain execution is controlled by the
JAX sampler's `chain_method` (`"parallel"` or `"vectorized"`).  Treat these as
separate execution controls.

hrHSA exposes a version-aware helper so RSF and SSF/iSSF workflows can keep sampler
geometry explicit while continuing to pass ordinary `sample_kwargs` through to
PyMC:

```python
from hsa import configure_pymc_sampling

sample_kwargs = configure_pymc_sampling(
    {
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.9,
    },
    nuts_sampler="blackjax",
    chain_method="vectorized",
)

fit = analysis.fit(sample_kwargs=sample_kwargs)
```

The same `sample_kwargs` object can be passed to Bayesian validation/CV.  The
helper deliberately does **not** choose BlackJAX or a chain method automatically;
backend promotion remains benchmark-driven.

A workstation affinity diagnostic tested the hypothesis that BlackJAX's loss of
outer-fold scaling came from unrestricted Dask-worker CPU affinity.  Pinning each
of 12 workers to a distinct physical core changed total wall time by only about one
percent and did not recover per-fold performance.  CPU affinity is therefore not a
production optimization for this workload.

The next mechanism diagnostic compares `chain_method="parallel"` with
`"vectorized"` and records each worker's fold sequence.  This separates first-fold
JIT/initialization cost from steady-state contention instead of averaging both into
a single median.

## Backend validation campaign

Use the dedicated validator when comparing samplers consequentially:

```bash
python benchmarks/hpc/validate_bayesian_samplers.py \
    --output-dir inference-results/backend-validation \
    --rows 100000 \
    --predictors 3 \
    --individuals 20 \
    --bin-width 0.5 \
    --random-slopes 1 \
    --samplers pymc,nutpie,numpyro,blackjax \
    --reference-sampler pymc \
    --draws 1000 \
    --tune 1000 \
    --chains 4 \
    --cores 4 \
    --repeats 3
```

For each successful backend the runner records completed wall time, ESS/s, R-hat,
divergences, tree depth/steps and posterior size. It also compares flattened
posterior summaries with the reference sampler using differences in posterior mean,
posterior scale and pooled-posterior-SD units.

Large backend differences should be reproduced in fresh Python processes because
JIT compilation and in-process caches can materially affect timing. Keep the
synthetic data seed fixed while changing sampling seeds between independent
repetitions. After the baseline model, repeat the comparison with at least one
larger and more hierarchical model before changing any recommended backend.

BlackJAX is therefore a viable backend candidate, but not a package-wide default
based on dispatch timing or one synthetic workload. Promotion requires reproducible
completed-time performance, acceptable NUTS diagnostics and posterior agreement on
representative hrHSA models.

## Observation-level `eta` storage

The hierarchical model always computes the linear predictor `eta` for the Binomial
likelihood, but hrHSA does not store it as a posterior deterministic by default.
For a model with `N` aggregated observations, `C` chains and `D` retained draws,
storing `eta` adds approximately

\[
N \times C \times D
\]

posterior values. This can increase posterior size by orders of magnitude without
changing the likelihood. Enable it only when observation-level posterior linear
predictors are explicitly required:

```python
build_bayesian_rsf_model(data, store_eta=True)
```

or through `BayesianRSF(..., model_kwargs={"store_eta": True})`.

## Relation to cross-validation and CoolMUC-4

Single-fit benchmarks answer which numerical method is efficient **inside** one
model. They do not determine the best node-level execution geometry. For production
LOIO, the natural outer unit is an independent held-out fold.

The recommended order is:

1. validate the preferred frequentist optimizer against Newton;
2. validate Bayesian backends using completed inference time, posterior agreement
   and NUTS diagnostics;
3. keep native BLAS/OpenMP threads bounded inside each fold;
4. benchmark fixed-workload outer-fold scaling on the workstation;
5. repeat the outer-fold sweep on CoolMUC-4 while recording fold throughput,
   core-seconds/fold and per-fold loading time;
6. only then spend additional cores inside one Bayesian fit if fold-level
   concurrency is insufficient to occupy the allocation.

The dedicated commands and CoolMUC Slurm launcher are documented in
{doc}`cv-performance`.
