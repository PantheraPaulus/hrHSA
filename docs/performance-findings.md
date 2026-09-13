# Workstation performance findings

This page records empirical conclusions from the production-workstation dry run.
They are calibration results for the tested hardware/workloads, not universal
hrHSA defaults.

## Distributed surface persistence

The corrected surface benchmark keeps completed Dask chunks distributed rather than
assembling the final raster on one worker. For the standard 24,576 x 24,576 x 6
workload (576 output partitions), an eight-process/one-thread geometry reduced the
largest post-operation worker RSS from roughly 2.6 GiB under final-array assembly
to roughly 0.63 GiB while retaining all 576 output partitions. Surface wall time
also decreased because the final inter-worker assembly was removed.

Use records with

```text
metadata.materialization = distributed_persist_no_final_assembly
```

as the corrected distributed-surface baseline. Older
`distributed_wait_no_driver_gather` records remain useful for within-protocol
comparisons but include benchmark-only finalization work.

## Frequentist inference

On a two-million-row, six-predictor synthetic RSF, L-BFGS was modestly faster than
Newton and converged to essentially the same solution. Increasing BLAS threads from
one to the full workstation did not produce useful scaling for this low-dimensional
logistic fit. The preferred scaling axis is therefore independent models/folds,
not many BLAS threads inside one model.

Once the optimizer is this fast, continuous design-matrix construction becomes a
comparable or larger fraction of end-to-end fit time. hrHSA therefore uses a
preallocated continuous-only matrix path while retaining the categorical reference
path.

## Bayesian inference

Aggregating repeated covariate combinations into Binomial counts is effectively
free relative to NUTS sampling and materially reduces observation count. Posterior
storage of observation-level `eta` should remain opt-in because it can dominate
posterior size and memory without changing posterior geometry.

An initial benchmark showed an exceptionally large BlackJAX speed advantage over
the native PyMC backend on one synthetic hierarchical model. That result is not a
package-default decision. Validate such differences with
`validate_bayesian_samplers.py`, which uses fresh Python processes, paired sampling
seeds, standard diagnostics and posterior location/scale agreement checks.
