# Hierarchical Bayesian RSF workflow

The Bayesian RSF machinery lives in the package rather than in the vulture
notebook. The recommended public interface is now the stateful `BayesianRSF`
workflow; the functional kernels in `hsa.rsf.bayesian`,
`hsa.rsf.bayesian_cv`, and `hsa.rsf.bayesian_validation` remain public for
advanced use and backwards compatibility.

The bearded-vulture notebook
`notebooks/vultures/src/bayesian-rsf-package-verification.ipynb`
is the runnable integration example.

Install Bayesian dependencies with:

```bash
pip install -e ".[bayesian]"
```

## Analysis object

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
```

The object owns the relocations, environmental stack, predictor definition,
availability-domain configuration, and estimator configuration.

## Hierarchical model

Continuous predictors can be binned in their original ecological units before
aggregation. Standardization uses the mean and standard deviation of the
original unbinned observations. The PyMC model supports population intercept
and slopes plus non-centred individual random intercepts and random slopes.
The aggregated use--availability likelihood is Binomial.

A full-data fit is:

```python
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
```

The fitted object keeps the PyMC model, `InferenceData`, aggregation metadata,
predictor definition and environmental stack together.

## Optional shrinkage for correlated candidate predictors

The default population coefficient prior remains an independent
`Normal(0, 1)` on standardized predictors. This is the recommended default for
small, biologically pre-specified predictor sets.

For a broader candidate set containing correlated predictors, a regularized
horseshoe prior can be enabled through `model_kwargs`:

```python
rsf = BayesianRSF(
    reloc,
    env,
    predictors=[
        "slope",
        "ruggedness",
        "tpi",
        "cliff_fraction",
        "elevation",
        "ndvi",
    ],
    id_col="individual-local-identifier",
    model_kwargs={
        "beta_prior": "regularized_horseshoe",
        "shrinkage": {
            "expected_nonzero": 3,
            "slab_scale": 2.0,
            "slab_df": 4.0,
        },
    },
)
```

`expected_nonzero` expresses a prior expectation for how many of the submitted
population-level coefficients will escape strong global shrinkage. It must be
positive and smaller than the number of predictors. Alternatively, an explicit
`global_scale` can be supplied instead of `expected_nonzero`.

When `expected_nonzero` is used, hrHSA derives the global horseshoe scale as

```text
tau0 = expected_nonzero / (p - expected_nonzero) / sqrt(effective_n)
```

where `p` is the number of candidate predictors. For an RSF, the default
`effective_n` is the number of **used locations** represented in the fitted
data, not the total number of used plus sampled-available trials. Availability
points are under analyst control, so using all trials would make the prior
artificially tighter when `sampling_factor` is increased. Advanced users may
override `effective_n` explicitly:

```python
model_kwargs={
    "beta_prior": "regularized_horseshoe",
    "shrinkage": {
        "expected_nonzero": 3,
        "effective_n": 500,
    },
}
```

The regularized horseshoe requires standardized predictors so that a common
shrinkage scale has a coherent interpretation. The standard Bayesian RSF data
preparation already standardizes predictors by default.

Shrinkage should be interpreted as regularization of independently supported
predictive information, not as proof that a retained predictor is causal. When
two covariates are nearly interchangeable, the likelihood may identify their
combined predictive contribution much more strongly than either coefficient
separately.

### Correlation and shrinkage diagnostics

A fitted model exposes the population-beta posterior correlation matrix:

```python
fit.posterior_correlation()
```

and a combined predictor/posterior diagnostic:

```python
fit.collinearity_diagnostics(
    min_abs_predictor_corr=0.6,
)
```

The latter compares trial-weighted predictor correlations from the fitted
aggregated design with posterior correlations among population coefficients.
For strongly correlated predictors, an oppositely signed posterior coefficient
correlation indicates that the model is trading effect size between the two
predictors. Such rows are marked with `competing_posterior=True`.

For a regularized-horseshoe fit:

```python
fit.shrinkage_summary()
```

reports each population coefficient together with the posterior global
shrinkage parameter (`tau_median`), local scale, regularized local scale, and
effective coefficient-prior scale. These quantities describe how strongly a
coefficient is regularized; they are **not posterior inclusion probabilities**.

The same diagnostics are available for individual LOIO folds:

```python
cv.posterior_correlation("BG1018_Kika")
cv.collinearity_diagnostics(
    "BG1018_Kika",
    min_abs_predictor_corr=0.6,
)
cv.shrinkage_summary("BG1018_Kika")
```

## Leave-one-individual-out validation

Validation is composed with the estimator rather than encoded in a model
subclass:

```python
from hsa.rsf import LeaveOneIndividualOut

loio = LeaveOneIndividualOut(
    heldout=3,
    sampling_factor_train=50,
    n_background=100_000,
    n_bins=20,
    seed=42,
)

cv = rsf.validate(
    loio,
    n_boyce_draws=500,
    sample_kwargs={
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.9,
    },
)
```

Each outer fold is fitted using all individuals except the held-out animal.
Training availability is sampled independently inside each training animal's
availability domain. The held-out animal is scored against a fixed set of
available/background locations that is reused for every posterior draw.

The `BayesianLOIOResult` stores:

```python
cv.summary
cv.params
cv.boyce_bins
cv.diagnostics
```

The diagnostics retain posterior score matrices so post-fit uncertainty can be
rerun without repeating MCMC. If a regularized horseshoe is used, sampler
quality summaries also include the horseshoe global/local latent variables so
poorly mixed shrinkage parameters are not hidden by otherwise well-behaved
population coefficients.

## Validation uncertainty

Two complementary post-fit strategies are exposed.

### Temporally blocked bootstrap

```python
from hsa.rsf import BlockedBootstrap

uncertainty = cv.evaluate_uncertainty(
    BlockedBootstrap(
        replicates=2000,
        block="7D",
    )
)
```

The observed held-out trajectory is split into fixed-duration blocks. The same
number of **whole blocks** is sampled with replacement in every bootstrap
replicate; no final block is truncated to force an identical fix count. One
posterior coefficient draw is paired with each replicate.

The resulting distribution therefore combines posterior parameter uncertainty
with finite validation-sample uncertainty while respecting temporal dependence
better than a point bootstrap.

### Contiguous temporal validation

```python
from hsa.rsf import ContiguousTemporalBlocks

uncertainty = cv.evaluate_uncertainty(
    ContiguousTemporalBlocks(
        folds=10,
        unit="W",
    )
)
```

Whole calendar units are ordered chronologically and divided into consecutive
periods. Each real period is evaluated across all posterior draws. This is a
diagnostic of temporal non-stationarity—such as seasonality, behavioural
changes, dispersal phases, or changing accessibility—not a bootstrap interval.

### Both together

```python
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

The contiguous output keeps exact temporal provenance:

- `replicate`, a compact numeric fold key;
- `replicate_label`, e.g. `F03 | 2022-W14 → 2022-W26`;
- `heldout_units`, the exact ISO weeks actually represented;
- `start`, `end`, and `n_temporal_units`.

The exact week list matters when telemetry contains gaps: the first-to-last
range must not imply that every intermediate week contributed observations.

```python
uncertainty.temporal_periods(
    "BG1057_Curro"
)
```

The randomized temporal-fold mode is deliberately absent. Once the bootstrap
itself samples temporally coherent blocks, randomized temporal folds address a
largely redundant robustness question with far fewer realizations. Contiguous
periods are retained because they test the distinct ecological question of
systematic temporal change.

## Plotting

```python
fig, axes, temporal_periods = uncertainty.plot(
    "BG1057_Curro"
)
```

The first panel shows the blocked-bootstrap P/E uncertainty envelope; the second
shows the actual contiguous temporal P/E curves and labels them by period.

## Functional layer

The object-oriented classes orchestrate existing kernels rather than replacing
them. The following functions remain public:

- `prepare_bayesian_rsf_data`
- `build_bayesian_rsf_model`
- `evaluate_bayesian_rsf`
- `plot_bayesian_rsf_diagnostics`
- `predict_bayesian_rsf_surface`
- `posterior_beta_correlation`
- `predictor_correlation`
- `collinearity_diagnostics`
- `regularized_horseshoe_summary`
- `leave_one_individual_out_bayesian_rsf`
- `prepare_bayesian_boyce_scores`
- `bayesian_boyce_quantile_scores`
- `evaluate_bayesian_loio_uncertainty`

This separation keeps numerical code independently testable while giving
notebooks and applications a concise, state-safe workflow.
