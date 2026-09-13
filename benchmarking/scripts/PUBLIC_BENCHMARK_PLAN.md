# hrHSA public benchmark plan

This runbook keeps four benchmark families separate while using the same three
questions wherever they are scientifically meaningful:

1. **Capacity** — fixed calibrated hardware geometry, increasing scientific problem.
2. **Strong scaling** — fixed scientific problem, increasing physical resources.
3. **Weak scaling** — problem grows with physical resources at a fixed workload density.

Geometry discovery/profiling is a prerequisite, not part of the primary scaling
curves.

## 1. Surface prediction

The surface campaign uses nested chunk-aligned windows from one deterministic Zarr
family. The calibrated workstation geometry is currently `12x1`; the provisional
CoolMUC-4 per-node geometry is `8x14`.

### Workstation capacity validation

The exploratory capacity experiment identified a stable out-of-core transition. A
small reverse-order validation is sufficient:

```bash
python benchmarking/scripts/01_python/run_surface_scaling_instrumented.py \
  --root "$BASE" \
  --output "$BASE/capacity-knee-reverse-12x1.jsonl" \
  --mode capacity \
  --backend local \
  --geometry 12x1 \
  --raster-gib 160,144,128,120,112,96 \
  --warmup-repeats 1 \
  --repeats 3
```

Do not manually drop Linux page cache; the purpose is to test robustness to the
opposite natural access history.

### Production strong/weak scaling on CoolMUC-4

LRZ resource policy requires the one-node reference and multinode jobs to be
submitted separately:

- one node: `cm4_tiny`;
- two to four nodes: `cm4_std`;
- `cm4_inter` remains a one-node diagnostic/capacity environment.

Use:

```text
coolmuc4_tiny_surface_scaling.sbatch
coolmuc4_std_surface_scaling.sbatch
```

The surface weak-scaling density is currently 2 logical raster GiB per physical
core.

## 2. Partitioned point sampling

The legacy benchmark loads one complete GeoDataFrame and returns one complete
pandas table to the driver. The public point campaign instead stores a deterministic
family of primitive Parquet partitions:

```text
point_id : int64
u        : float64 in [0, 1)
v        : float64 in [0, 1)
used     : bool
```

`u` and `v` are mapped to the active raster window at run time. This makes the same
point family reusable when the raster itself grows during weak scaling.

No benchmark requires the complete point family or sampled result to exist in
driver memory at once.

### Prepare a workstation pilot family

Start with 100 million points, using one million rows per Parquet partition:

```bash
python benchmarking/scripts/01_python/prepare_point_family.py \
  --root "$BASE" \
  --points 1000000,10000000,30000000,100000000 \
  --partition-size 1000000
```

The script is resumable and only generates missing matching partitions.

### Workstation point capacity pilot

Keep one representative raster fixed while increasing only point count:

```bash
python benchmarking/scripts/01_python/run_point_scaling.py \
  --root "$BASE" \
  --output "$BASE/point-capacity.jsonl" \
  --mode capacity \
  --backend local \
  --geometry 12x1 \
  --raster-gib 24 \
  --point-counts 1000000,10000000,30000000,100000000 \
  --warmup-points 1000000 \
  --warmup-repeats 1 \
  --repeats 3
```

Every repeat emits both a pure sampling-kernel record and an end-to-end partition
pipeline record. `--write-sampled --sampled-output <dir>` can additionally time
partitioned result persistence without ever concatenating the result.

### Point strong scaling

Fix both the raster and point count:

```bash
python benchmarking/scripts/01_python/run_point_scaling.py \
  --root "$BASE" \
  --output "$BASE/point-strong.jsonl" \
  --mode strong \
  --backend local \
  --geometry 12x1 \
  --raster-gib 24 \
  --fixed-points 30000000 \
  --core-counts 1,2,4,6,8,10,12 \
  --repeats 3
```

### Point weak scaling

Grow both raster size and point count per physical core. The initial density is:

```text
2 logical raster GiB / physical core
2.5 million points / physical core
```

```bash
python benchmarking/scripts/01_python/run_point_scaling.py \
  --root "$BASE" \
  --output "$BASE/point-weak.jsonl" \
  --mode weak \
  --backend local \
  --geometry 12x1 \
  --core-counts 1,2,4,6,8,10,12 \
  --gib-per-core 2 \
  --points-per-core 2500000 \
  --repeats 3
```

Before creating a billion-row CoolMUC family, use the workstation pilot to quantify
partition overhead and choose the final HPC maximum.

## 3. RSF inference

`benchmark_inference.py` already separates:

- frequentist design-matrix construction;
- frequentist optimization;
- Bayesian RSF bin aggregation;
- Bayesian model construction;
- completed MCMC sampling/materialization.

For Bayesian results, report completed sampling time, memory, ESS/s, R-hat and
divergences. Keep raw-data preparation separate from the compressed likelihood
stage so binning benefits are visible rather than hidden.

## 4. SSF/iSSF inference

`benchmark_ssf_inference.py` provides deterministic fixed-choice-set workloads.

Frequentist mode benchmarks the optimized vectorized conditional likelihood and,
for bounded sizes, the Statsmodels reference implementation:

```bash
python benchmarking/scripts/01_python/benchmark_ssf_inference.py frequentist \
  --output "$BASE/ssf-frequentist.jsonl" \
  --analysis ssf \
  --strata 1000,10000,50000,100000,500000 \
  --choices 20 \
  --predictors 6 \
  --individuals 20 \
  --engines fast,statsmodels \
  --statsmodels-max-strata 50000 \
  --blas-threads 1 \
  --repeats 3
```

For iSSF, use `--analysis issf`; the synthetic design then includes `log_sl` and
`cos_ta` movement predictors.

Bayesian mode uses the same one-outcome-per-choice-set categorical likelihood:

```bash
python benchmarking/scripts/01_python/benchmark_ssf_inference.py bayesian \
  --output "$BASE/ssf-bayesian.jsonl" \
  --analysis ssf \
  --strata 1000,5000,10000 \
  --choices 20 \
  --predictors 6 \
  --individuals 20 \
  --samplers pymc,blackjax \
  --draws 1000 \
  --tune 1000 \
  --chains 4 \
  --cores 4 \
  --repeats 3
```

Bayesian capacity curves should increase strata/choice sets. Classical inner-model
weak scaling is not claimed unless the likelihood itself is distributed; parallel
outer folds/independent fits are the appropriate HPC scaling target.

## Publication interpretation

Keep these distinctions explicit:

- surface capacity changes raster size;
- point capacity changes point count at fixed raster;
- point weak scaling grows raster and points together at fixed per-core density;
- inference capacity changes observations/choice sets while reporting preparation
  separately from optimization/sampling;
- CV/independent-model scaling is an outer-parallel inference experiment, not
  evidence that one MCMC chain scales across nodes.

All records must retain exact source commit, workload definition, resource geometry,
warm-up policy and machine/allocation metadata.
