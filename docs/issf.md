# Integrated step-selection workflows

`hsa.ssf` exposes integrated step-selection functions (iSSFs) through a stateful workflow that separates four concerns:

1. create an analysis from telemetry;
2. generate movement-informed choice sets;
3. annotate static and dynamic environmental conditions;
4. declare the ecological model and fit it.

The public API deliberately does **not** require an environmental raster or a predictor list when the object is created. Environmental fields are passed only when they are sampled, and the model can be changed repeatedly without regenerating availability or resampling covariates.

For stratum `s` and candidate `j`, the proposal-corrected working utility is

\[
\eta_{sj}=\mathbf{x}_{sj}^{T}\boldsymbol\beta-\log q_{sj},
\]

where `q` is the candidate-generation density stored as `proposal_logpdf`. The fixed offset

\[
o_{sj}=-\log q_{sj}
\]

is an importance-sampling correction rather than an estimated ecological effect. It is centered within each stratum by default because adding a common constant to all alternatives leaves the conditional softmax likelihood unchanged.

The default movement basis is

\[
L,\qquad \log L,\qquad \cos\theta,
\]

with `L` represented as `step_length_km`. Movement terms remain on their natural scale. When the fitted coefficients imply a proper Gamma-like kernel,

\[
k=1+\gamma_{\log L},
\qquad
\lambda=-\gamma_L,
\qquad
E[L]=k/\lambda.
\]

## 1. Create the analysis

```python
from hsa.ssf import FrequentistISSF

issf = FrequentistISSF(
    reloc,
    id_col="individual-local-identifier",
    timestamp_col="Timestamp",
    expected_interval_min=60,
    tolerance_min=10,
)
```

No raster and no model specification are required at this stage.

## 2. Generate movement-informed availability

```python
issf.sample(
    n_available=20,
    seed=42,
)
```

If candidate endpoints must remain inside a static environmental domain, pass that raster only for choice-set generation:

```python
issf.sample(
    n_available=20,
    seed=42,
    domain=terrain,
    observed_outside="exclude",
)
```

The domain is used for candidate-wise rejection sampling and is not retained as analysis state.

The older `prepare_choice_sets()` method remains available as a low-level compatibility API.

## 3. Annotate static covariates

Endpoint and start conditions can be sampled in one call:

```python
issf.annotate_static(
    terrain,
    endpoint=[
        "elevation",
        "slope",
    ],
    start={
        "vrm_2070m": "vrm_start",
    },
)
```

`endpoint=` samples every candidate endpoint. `start=` samples each unique stratum origin once and merges the result back to all alternatives in that stratum.

When `start=` is supplied as a sequence rather than a mapping, `_start` is appended automatically:

```python
issf.annotate_static(
    terrain,
    start=["elevation", "vrm_2070m"],
)
```

produces `elevation_start` and `vrm_2070m_start`.

## 4. Annotate dynamic covariates

The same pattern applies to time-varying fields:

```python
issf.annotate_dynamic(
    met,
    endpoint={
        "sensible_heat_flux_upward": "heat",
        "surface_air_temperature_excess": "temp_excess",
    },
    start={
        "sensible_heat_flux_upward": "heat_start",
        "surface_air_temperature_excess": "temp_excess_start",
    },
    method="linear",
)
```

The defaults encode the canonical iSSF timing semantics:

- endpoint covariates are sampled at `end_time`;
- start covariates are sampled at `start_time`.

These can be overridden with `endpoint_time_col=` and `start_time_col=` when necessary.

## 5. Vector fields and wind support

Vector annotation is exposed as one convenience operation:

```python
issf.annotate_vector(
    met,
    u="wind_u_10m",
    v="wind_v_10m",
    prefix="wind",
    at=("endpoint", "start"),
    support=True,
    method="linear",
)
```

This samples the vector components and derives vector magnitude plus geodesic support/alignment. With `prefix="wind"`, the important model-ready outputs include:

```text
wind_support
wind_start_support
```

`wind_support` uses the wind vector at each candidate endpoint. `wind_start_support` uses the single wind vector at the shared step start but projects it onto each candidate bearing, so it varies among alternatives and is therefore a candidate-level directional predictor.

Positive support is tailwind support; negative support is headwind support.

## 6. Declare the ecological model

The preferred model API is `set_model()`.

A plain endpoint-selection model is:

```python
issf.set_model(
    selection=[
        "elevation",
        "slope",
    ],
    movement=False,
)
```

With the proposal correction retained, this is an endpoint-only proposal-corrected choice model. To deliberately fit an uncorrected conventional conditional-choice model:

```python
issf.set_model(
    selection=["elevation", "slope"],
    movement=False,
    proposal_correction=False,
)
```

A baseline iSSF is:

```python
issf.set_model(
    selection=["elevation", "slope"],
    movement="default",
)
```

where `movement="default"` expands to:

```text
step_length_km
log_step_length
cos_turn_angle
```

### Environmental movement modifiers

Conditions at departure can modify the movement kernel:

```python
issf.set_model(
    selection=[
        "elevation",
        "slope",
    ],
    modifiers={
        "heat_start": "step_length",
        "temp_excess_start": "step_length",
        "vrm_start": "step_length",
        "wind_start_support": "step_length",
    },
)
```

The shorthand `"step_length"` means both `step_length_km` and `log_step_length`.

The package automatically determines whether each modifier is:

- **stratum-constant**, such as `heat_start` or `vrm_start`; or
- **candidate-varying**, such as `wind_start_support`.

This distinction remains explicit in the prepared `ISSFDesign`, but users no longer need to manually maintain `start_predictors` and `directional_predictors` in the common workflow.

Stratum-constant conditions enter only through movement interactions because their main effects cancel from a conditional likelihood. Candidate-varying directional modifiers receive a main effect as well as their requested movement interactions.

Different modifiers can target different movement components:

```python
issf.set_model(
    selection=["elevation", "slope"],
    modifiers={
        "heat_start": "step_length",
        "vrm_start": "step_length",
        "wind_start_support": [
            "step_length",
            "turning",
        ],
    },
)
```

Here only wind support modifies `cos_turn_angle`.

Inspect the inferred model before fitting:

```python
issf.model_spec_.summary()

design = issf.prepare_design()
design.predictors
design.scaling
design.summary()
```

`set_predictors()` remains available as a backwards-compatible wrapper but `set_model()` is preferred for new code.

## 7. Frequentist fit

```python
fit = issf.fit(
    engine="fast",
    method="lbfgs",
    maxiter=1000,
)

fit.summary()
fit.coefficients()
fit.choice_scores()["summary"]
```

No-pooling individual fits reuse the same model design and scaling:

```python
individual = issf.fit_individuals()
individual.summary
```

Directional predictor metadata is retained in these individual designs.

## 8. Movement diagnostics

Expected displacement can be plotted against any movement modifier, including directional modifiers:

```python
fit.plot_movement_response(
    "heat_start",
)
```

or

```python
fit.plot_movement_response(
    "wind_start_support",
)
```

Step-length distributions can be compared across standardized environmental levels:

```python
fit.plot_step_length_distribution(
    "heat_start",
    levels=(-1, 0, 1),
)
```

and turning-angle distributions can be inspected with:

```python
fit.plot_turning_angle_distribution(
    "wind_start_support",
    levels=(-1, 0, 1),
)
```

These are distributions of net displacement between fixes, not total distance flown.

## 9. Bayesian hierarchy

A prepared frequentist analysis can be converted without rebuilding choice sets or resampling the environment:

```python
bayes = issf.to_bayesian(
    model_kwargs={
        "mu_sigma": 1.0,
        "heterogeneity_sigma": 0.35,
    }
)
```

Then sample normally:

```python
bayes_fit = bayes.fit(
    sample_kwargs={
        "draws": 1500,
        "tune": 1500,
        "chains": 4,
        "target_accept": 0.95,
    }
)
```

For individual `i` and predictor `p`,

\[
\beta_{ip}=\mu_p+\sigma_p z_{ip},
\qquad
z_{ip}\sim N(0,1).
\]

Population means, partially pooled individual coefficients, and heterogeneity are available separately:

```python
bayes_fit.coefficients()
bayes_fit.individual_coefficients()
bayes_fit.heterogeneity()
```

Bayesian movement-response plots honor both stratum-constant and directional movement modifiers.

### External MCMC

External sampling belongs to the analysis object:

```python
bayes.export_fit("results/issf_run")
```

After the external process saves `idata.nc`:

```python
bayes_fit = bayes.load_fit(
    "results/issf_run/idata.nc"
)
```

The export contains only the model-ready numeric design and metadata; spatial objects and environmental rasters are not serialized.

## 10. Subsetting for smoke tests

Subsetting always operates on complete strata and reuses existing annotations:

```python
small = issf.subset(
    n_strata_per_id=100,
    seed=42,
)
```

The returned object preserves the model specification, including directional modifiers, but resets prepared designs and fits.

## Low-level API

The lower-level design API remains available for explicit workflows:

```python
from hsa.ssf import (
    ISSFDesign,
    prepare_issf_design,
    build_hierarchical_issf_model,
)
```

The public facade is intentionally thin: statistical preparation, scaling, proposal correction, dense choice arrays, and likelihood construction remain in the established iSSF core.

## Current validation boundary

The iSSF API still does not expose the existing SSF cross-validation schemes. Correct iSSF validation must fit environmental scaling inside each training fold and carry proposal correction into held-out choice sets. Until that fold-specific design preparation is implemented, `validate()` raises rather than silently leaking held-out information.
