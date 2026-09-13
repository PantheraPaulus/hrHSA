# Inference performance benchmarking

The raster benchmark campaign measures environmental extraction and RSF surface
projection. Statistical inference is benchmarked separately so filesystem and Dask
behavior do not obscure the cost of building and solving the statistical model.

## Frequentist logistic RSF

The frequentist benchmark uses deterministic synthetic use/availability data and
separates two stages:

1. design-matrix construction (`StandardScaler`, quadratic/interactions,
   categorical encoding where applicable), and
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
record their maximum absolute coefficient difference from it. Records also retain
convergence state, iteration/function-call counts when Statsmodels exposes them,
log likelihood, throughput and memory information.

`fit_rsf()` now exposes Statsmodels' optimizer choice directly:

```python
fit_rsf(df, spec, method="lbfgs", fit_kwargs={"maxiter": 200})
```

The scientific default remains `method="newton"`. Do not change a project default
solely because another solver is faster: require convergence and coefficient/log
likelihood agreement first.

The `--blas-threads` sweep is intentionally independent of Dask. Logistic MLE is a
dense numerical problem and should first be benchmarked through the underlying
BLAS/thread pool rather than by wrapping one fit in distributed Dask tasks.

## Hierarchical Bayesian RSF

Bayesian performance has different useful metrics. A NUTS chain is sequential, but
multiple chains and multiple model/fold fits can run independently. Wall time alone
is therefore insufficient: sampler quality per unit time matters.

A workstation comparison can be run with:

```bash
python benchmarks/hpc/benchmark_inference.py bayesian \
    --output-dir inference-results/bayesian \
    --rows 100000 \
    --predictors 3 \
    --individuals 20 \
    --bin-width 0.5 \
    --samplers pymc,nutpie,numpyro,blackjax \
    --draws 500 \
    --tune 500 \
    --chains 4 \
    --cores 4
```

Optional NUTS backends are skipped when they are not installed. Install the desired
backend in the benchmark environment before making a formal comparison.

The JSONL record includes, where available:

- raw and aggregated observation counts;
- aggregation compression ratio;
- model-build wall time;
- total `pm.sample()` wall time;
- sampler-reported sampling time;
- raw draws/s;
- minimum and median bulk ESS/s;
- minimum tail ESS;
- maximum R-hat;
- divergences;
- maximum tree depth and mean NUTS steps;
- posterior size;
- whether observation-level `eta` was stored.

ESS/s is the primary performance quantity for comparing NUTS implementations:

\[
\mathrm{ESS/s} = \frac{\mathrm{effective\ sample\ size}}{\mathrm{wall\ time}}.
\]

A sampler that produces twice as many raw draws per second is not faster in a
scientifically useful sense if those draws contain substantially less effective
information.

## Observation-level `eta` storage

The hierarchical model always computes the linear predictor `eta` for the Binomial
likelihood, but hrHSA no longer stores it as a posterior deterministic by default.
For a model with `N` aggregated observations, `C` chains and `D` retained draws,
storing `eta` adds approximately

\[
N \times C \times D
\]

posterior values. At large `N` this can dominate posterior memory and serialization.
Enable it only when observation-level posterior linear predictors are explicitly
needed:

```python
build_bayesian_rsf_model(data, store_eta=True)
```

or through `BayesianRSF(..., model_kwargs={"store_eta": True})`.

To measure the actual cost on a machine, use:

```bash
python benchmarks/hpc/benchmark_inference.py bayesian \
    --output-dir inference-results/eta-storage \
    --samplers pymc \
    --eta-storage both
```

This builds and samples otherwise identical models with and without posterior `eta`
storage.

## Relation to CoolMUC-4

For one Bayesian model, the first scaling axis is usually chains rather than Dask
workers. On CoolMUC-4, the larger opportunity is expected to be model-level
parallelism: independent LOIO folds, candidate model specifications or simulation
replicates can occupy separate CPU groups while each model runs its own chains.

Recommended order:

1. benchmark one-chain and four-chain behavior on the workstation;
2. compare installed NUTS engines using ESS/s;
3. determine useful BLAS/thread allocation per chain;
4. on CoolMUC-4, benchmark several independent model/fold fits concurrently;
5. only then benchmark larger within-model CPU allocations if the chosen sampler
   can exploit them.

This keeps inference scaling conceptually separate from raster strong scaling and
makes end-to-end performance results attributable to the correct subsystem.
