# Bayesian CPU execution policy

hrHSA treats Bayesian sampler selection and CPU placement as two separate decisions:

1. **Which sampler is most efficient for one posterior?**
2. **Where should available CPU cores be placed when many independent posteriors exist?**

The recommendations below are explicit rather than hidden defaults so software-version and site-specific benchmark evidence can override them.

## Current measured policy

| Analysis | One/few fits (latency) | Many independent fits (generic CPU) | CoolMUC-4 full-node throughput |
| --- | --- | --- | --- |
| Bayesian RSF | BlackJAX | nutpie, outer-first | nutpie, **56 outer x 2 cores/fit** |
| Bayesian SSF | nutpie | nutpie, outer-first | nutpie, **28 outer x 4 cores/fit** |
| Bayesian iSSF | nutpie | nutpie, outer-first | nutpie, **28 outer x 4 cores/fit** |

The RSF single-fit exception is deliberate: BlackJAX produced higher ESS per completed second for the full hierarchical RSF even though nutpie scaled much better across many independent folds. For SSF/iSSF, nutpie was substantially faster than the native PyMC sampler in the tested CPU workloads while retaining clean posterior agreement and diagnostics.

## Portable outer-first scheduling

For generic throughput workloads, independent models/folds are filled first. Spare cores are then assigned to concurrent chains within each fit:

```python
from hsa.compute import plan_bayesian_cpu_execution

plan = plan_bayesian_cpu_execution(
    analysis="rsf",
    n_models=24,
    available_cores=24,
    chains=4,
)
```

On the 24-core workstation the measured RSF CV geometries were naturally represented as:

```text
6 models   -> 6 x 4 cores/fit
12 models  -> 12 x 2 cores/fit
24 models  -> 24 x 1 core/fit
```

Four statistical chains remain in every posterior. `cores_per_fit=1` means the chains execute sequentially within a fit while more independent fits execute concurrently.

## Measured CoolMUC-4 profile

A complete CoolMUC-4 node exposes 112 physical cores. Fixed-size one-node diagnostics held the number and statistical size of independent posteriors constant while moving the same 112 cores between outer-model and within-fit chain concurrency.

For 112 independent Bayesian RSF posteriors (12,000 raw rows/model, 24 individuals/model, 3 predictors, 250+250 draws/tune), the measured topology points were:

| Geometry | Wall [s] | Models/s | Median completed sampling [s] |
| --- | ---: | ---: | ---: |
| 28 x 1 | 138.58 | 0.808 | 28.17 |
| 56 x 1 | 97.56 | 1.148 | 41.98 |
| **56 x 2** | **85.66** | **1.308** | 36.01 |
| 28 x 4 | 92.19 | 1.215 | **15.84** |
| 112 x 1 | 120.57 | 0.929 | 99.15 |

Thus **56 x 2** is the measured large-ensemble RSF throughput geometry. `28 x 4` remains attractive when the ensemble is small enough that per-fit latency dominates.

For 112 independent 5,000-stratum hierarchical choice models (11 choices/stratum, 20 individuals, 250+250), the full-node comparison was:

| Analysis | Geometry | Wall [s] | Models/s | Median completed sampling [s] | Median min bulk ESS/s |
| --- | --- | ---: | ---: | ---: | ---: |
| SSF | 56 x 2 | 286.13 | 0.391 | 106.16 | 1.94 |
| SSF | **28 x 4** | **244.14** | **0.459** | **47.79** | **4.15** |
| iSSF | 56 x 2 | 397.57 | 0.282 | 153.76 | 0.862 |
| iSSF | **28 x 4** | **395.40** | **0.283** | **73.05** | **1.739** |

For SSF, `28 x 4` is clearly faster in total wall time. For iSSF the ensemble wall times are effectively tied (about 0.5% apart), but `28 x 4` halves the number of active worker processes, roughly halves the active worker-memory footprint, cuts median per-fit latency by about 2.1x, and doubles minimum bulk ESS/s. It is therefore the frozen CoolMUC-4 iSSF production geometry as the safer tie-breaker.

The site-specific planner is opt-in:

```python
plan = plan_bayesian_cpu_execution(
    analysis="issf",
    n_models=112,
    available_cores=112,
    chains=4,
    objective="throughput",
    profile="coolmuc4",
)

assert (plan.outer_workers, plan.cores_per_fit) == (28, 4)
```

The `coolmuc4` profile applies the measured policy only to the tested 112-core/four-chain full-node case; other allocations retain the portable rule.

## CLI planning

```bash
python benchmarking/scripts/01_python/plan_bayesian_execution.py \
  --analysis ssf \
  --models 112 \
  --available-cores 112 \
  --chains 4 \
  --objective throughput \
  --profile coolmuc4
```

For batch scripts, shell output can be evaluated directly with `--format shell`.

## Installation

The validated CPU samplers remain optional dependencies:

```bash
pip install -e ".[bayesian,nutpie]"
pip install -e ".[bayesian,blackjax]"
```

For the complete SLURM benchmark environment use:

```bash
pip install -e ".[bayesian-hpc]"
```

The BlackJAX version remains constrained by the PyMC integration compatibility note in `pyproject.toml`.

## CoolMUC-4 capacity campaign

Topology and capacity are deliberately separated. After freezing the full-node geometry, the choice-model capacity campaign uses **one complete 28-model wave at 28 x 4** and varies only statistical size. This avoids mixing model-size scaling with the number of scheduling waves.

The capacity ladder is 5k, 10k, 20k, and 50k strata/model, with 11 choices, 20 individuals, and **500 tune + 500 draws** so it is directly comparable to the workstation capacity curve. Each point is submitted separately because LRZ association limits can prevent bulk submission.

```bash
bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh ssf-capacity 5000
bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh ssf-capacity 10000
bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh ssf-capacity 20000
bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh ssf-capacity 50000
```

and equivalently for iSSF:

```bash
bash benchmarking/scripts/03_sh/submit_coolmuc4_inter_bayesian.sh issf-capacity 5000
```

Submit the next point only after queue capacity is available. The choice benchmark streams completed per-model records to a `.tasks.jsonl` sidecar so a walltime-limited capacity point still preserves completed-fit evidence.

## Reproducibility rule

The package planner does not silently switch an analysis to whichever backend happens to be installed. Production notebooks and batch jobs should record sampler, chains, outer workers, cores per fit, PyMC/backend versions, site profile, and software revision. A faster backend or geometry should only be promoted after posterior agreement and diagnostics have been checked for the relevant model family.
