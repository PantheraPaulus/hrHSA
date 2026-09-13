# Object-oriented RSF workflows

`hrHSA` exposes a stateful object layer for complete resource-selection analyses
while keeping the lower-level functional kernels public. The classes own analysis
state; the existing functions remain the numerical/statistical implementation.

This deliberately uses **shallow inheritance** and **composition**:

```text
RSFAnalysis
├── FrequentistRSF
└── BayesianRSF

RSFFit
├── FrequentistRSFFit
└── BayesianRSFFit

validation strategies
├── LeaveOneIndividualOut
├── BlockedBootstrap
└── ContiguousTemporalBlocks
```

The estimator classes inherit common RSF state and workflow. Validation strategies
are separate objects rather than subclasses of the estimator, avoiding a
combinatorial hierarchy such as `BayesianTemporalLOIORSF`.

## Why the object layer exists

A fitted frequentist RSF is not only a statsmodels model: its scaler, feature
specification and design-matrix metadata must travel with it. A Bayesian fit
likewise consists of the PyMC model, `InferenceData`, aggregation metadata and
predictor definitions. Keeping those pieces in result objects prevents accidental
mixing of state from different models.

The object layer also centralizes information common to both estimators:

- relocations;
- environmental raster stack;
- predictors;
- individual ID and timestamp columns;
- availability domains;
- used/available sampling;
- validation strategy.

Raster extraction follows the same rule for both estimators: only bands required
by the fitted predictor specification are sampled by default. This applies to
pooled fits and LOIO workflows, including Bayesian held-out Boyce scoring. Keeping
unused bands out of point extraction reduces raster I/O and avoids constructing
unnecessarily large Dask indexing graphs.

## Frequentist example

```python
from hsa.rsf import FrequentistRSF
from hsa.types import FeatureSpec

rsf = FrequentistRSF(
    reloc,
    env,
    spec=FeatureSpec(
        linear=[
            "ruggedness_2070m",
            "elevation",
            "slope",
        ],
    ),
    id_col="individual-local-identifier",
)

fit = rsf.fit(
    sampling_factor=10,
    thin_dt="12h",
)

fit.coefficients()
surface = fit.predict_surface()
```

The low-level `fit_rsf`, `predict_rsf_points` and `predict_rsf_surface`
functions remain available.

## Hierarchical Bayesian example

```python
from hsa.rsf import BayesianRSF

rsf = BayesianRSF(
    reloc,
    env,
    predictors=[
        "ruggedness_2070m",
        "elevation",
        "slope",
    ],
    id_col="individual-local-identifier",
    binning={
        "ruggedness_2070m": 50,
        "elevation": 100,
        "slope": 5,
    },
    random_slopes=[
        "ruggedness_2070m",
        "elevation",
    ],
)

fit = rsf.fit(
    sampling_factor=50,
    sample_kwargs={
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.9,
    },
)

fit.coefficients()
fit.summary()
fit.plot_diagnostics()
surface = fit.predict_surface(n_draws=500)
```

The underlying functions in `hsa.rsf.bayesian` remain public for advanced use
and testing.

## Composable leave-one-individual-out validation

The same validation strategy can be passed to either estimator:

```python
from hsa.rsf import LeaveOneIndividualOut

loio = LeaveOneIndividualOut(
    heldout="all",
    sampling_factor_train=50,
    n_background=100_000,
    n_bins=20,
    seed=42,
)

cv = rsf.validate(loio)
```

For a `BayesianRSF`, `cv` is a `BayesianLOIOResult`; for a `FrequentistRSF`, it
is a `FrequentistLOIOResult`. Common result state is available through:

```python
cv.summary
cv.params
cv.boyce_bins
cv.diagnostics
```

The frequentist result additionally retains `calibration_bins` and provides
convenience plotting methods:

```python
cv.plot_boyce_values()
cv.plot_boyce_curves()
```

For LOIO curves the default x-axis is the available-landscape RSF quantile
(`q_mid`), which is directly comparable across independently fitted folds.

## Post-fit validation uncertainty

Both frequentist and Bayesian LOIO results accept the same post-fit validation
strategies:

```python
from hsa.rsf import (
    BlockedBootstrap,
    ContiguousTemporalBlocks,
)

uncertainty = cv.evaluate_uncertainty(
    BlockedBootstrap(
        replicates=2000,
        block="7D",
    ),
    ContiguousTemporalBlocks(
        folds=10,
        unit="W",
    ),
)
```

The result exposes a common interface:

```python
uncertainty.baseline_summary
uncertainty.baseline_curves
uncertainty.replicate_summary
uncertainty.method_summary
uncertainty.curves
uncertainty.curve_summary
uncertainty.temporal_periods()
```

Estimator-independent rank-based Boyce and temporal-block operations live in
`hsa.rsf.validation_utils`. The frequentist and Bayesian validation modules both
call these shared helpers instead of depending on one another's private internals.

### Temporally blocked bootstrap

The held-out trajectory is divided into fixed-duration temporal blocks. Every
bootstrap replicate samples the same number of **whole blocks with replacement**.
Blocks are never truncated merely to force an identical number of GPS fixes in
every replicate.

For a **frequentist RSF**, the fold-specific fitted model is held fixed. The
bootstrap therefore represents finite validation-sample uncertainty only. A
single availability sample is drawn for each held-out individual and reused for
all replicates.

For a **Bayesian RSF**, one posterior coefficient draw is additionally paired
with each bootstrap replicate. Its distribution therefore combines posterior
parameter uncertainty with finite validation-sample uncertainty.

These intervals must not be interpreted as equivalent: the frequentist interval
is conditional on the fitted model, whereas the Bayesian interval also propagates
posterior coefficient uncertainty.

### Contiguous temporal blocks

The real held-out chronology is split into consecutive calendar periods. This is
an empirical diagnostic of temporal non-stationarity rather than a bootstrap or
credible interval.

For a frequentist RSF each period is evaluated using the fixed fitted model. For
a Bayesian RSF each period is evaluated across posterior draws.

The result records exact provenance for every period:

```python
uncertainty.temporal_periods("BG1057_Curro")
```

including:

- numeric `replicate`;
- `replicate_label`, e.g. `F03 | 2022-W14 → 2022-W26`;
- `heldout_units`, the exact ISO weeks that actually contained observations;
- `start`, `end`, `n_temporal_units`, and `n_used`;
- Boyce performance for that period.

Tracking exact weeks matters when telemetry contains gaps: a range label does not
imply that every intermediate week contributed observations.

Randomized temporal folds are intentionally absent. Once the bootstrap already
resamples temporally coherent blocks, randomized temporal folds answer a largely
redundant robustness question. Contiguous periods are retained because they ask
the distinct ecological question of whether predictive performance changes
systematically through time.

## Plotting post-fit validation

```python
fig, axes, periods = uncertainty.plot(
    "BG1057_Curro"
)
```

The first panel shows the blocked-bootstrap P/E uncertainty envelope. The second
shows the actual contiguous temporal P/E curves and their temporal labels.

## Architecture rule

Classes **orchestrate and own state**; low-level functions **perform numerical
work**. Statistical kernels should normally remain functional and be called by
the workflow classes rather than being reimplemented as methods.

This keeps the package testable, preserves backwards compatibility, and makes the
high-level notebook API concise.
