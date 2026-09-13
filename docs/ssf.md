# Step-selection workflows

`hsa.ssf` implements movement-informed step-selection functions (SSFs) with a shared workflow for frequentist conditional logistic regression and hierarchical Bayesian categorical/softmax models.

The SSF workflow deliberately does **not** inherit from `RSFAnalysis`. Resource selection and step selection share data-handling ideas, but their availability processes and likelihoods are fundamentally different. SSF alternatives share a common step origin and are generated from the individual's movement kernel.

## Build movement-informed choice sets

```python
from hsa.ssf import BayesianSSF

analysis = BayesianSSF(
    reloc,
    env,
    predictors=["elevation", "slope", "vrm_2070m", "wind_support"],
    id_col="individual-local-identifier",
    timestamp_col="Timestamp",
    expected_interval_min=60,
    tolerance_min=10,
    n_available=20,
    speed_margin=1.05,
)

analysis.prepare_choice_sets(seed=42)
analysis.annotate_static(
    bands=["elevation", "slope", "vrm_2070m"],
    batch_size=100_000,
)
```

Step lengths are proposed from the fitted per-individual movement family. When a speed cap is used, the distribution is sampled by inverse-CDF truncation; draws are not clipped after sampling. Proposal log densities are retained in the choice table for later integrated-SSF extensions.

When an environmental raster is supplied, available endpoints outside raster coverage are regenerated rather than snapped to the raster edge. Observed endpoints are never moved. The safe default is to raise if an observed endpoint lies outside environmental support. If the raster cannot reasonably be expanded, complete unsupported strata can instead be excluded explicitly:

```python
analysis.prepare_choice_sets(
    seed=42,
    observed_outside="exclude",
)
analysis.choice_set_diagnostics_
```

`observed_outside="exclude"` removes the observed endpoint **and every alternative in that stratum**. This preserves the conditional-choice design but changes the target data set to strata for which environmental support exists, so the number and identity of excluded strata are recorded in `choice_set_diagnostics_` and should be reported. The default remains `"raise"` so this exclusion can never happen silently.

## Dynamic vector fields

Candidate endpoints may span multiple atmospheric grid cells, so dynamic wind is sampled independently at every alternative rather than broadcast from the step origin.

```python
analysis.annotate_dynamic_vectors(
    wind,
    time_col="end_time",
    u_var="u10",
    v_var="v10",
    method="linear",
)
analysis.add_vector_support(
    u_col="u10",
    v_col="v10",
    prefix="wind",
)
```

`wind_support > 0` is the tailwind component along the candidate's geodesic bearing; negative values indicate headwind. The standard wind names are `wind_support`, `crosswind`, `abs_crosswind`, and `wind_alignment`. Geodesic bearings are used because ERA5 vectors are true east/north components whereas projected raster axes need not align exactly with geographic east/north.

## Frequentist SSF

```python
from hsa.ssf import FrequentistSSF

freq = FrequentistSSF(
    reloc,
    env,
    predictors=["elevation", "slope", "vrm_2070m", "wind_support"],
    id_col="individual-local-identifier",
    choices=analysis.choices,
)
fit = freq.fit()
fit.coefficients()
fit.choice_scores()["summary"]
```

The model is a conditional logistic regression with one chosen endpoint per stratum and no intercept. Predictors are standardized once over the fitting choice table. Missing/non-finite predictor values are handled at the **whole-stratum** level: individual alternatives are never deleted from an otherwise retained choice set.

Individual no-pooling fits are available for heterogeneity diagnostics:

```python
individual = freq.fit_individuals()
individual.summary
individual.plot_selection_opportunity("elevation_z", label="Elevation")
```

## Selection opportunity and CIIF

For predictor `k`, the diagonal conditional information contributed by a choice set is

\[
I_{s,k}=\operatorname{Var}_{p_s}(x_{s,j,k}).
\]

This measures how much environmental contrast the animal could choose among. Low opportunity primarily limits coefficient precision; it does not imply a weak preference. For cross-individual comparisons, `fit_individuals()` evaluates these probabilities using the **same pooled reference beta for every individual**. This prevents differences in the animals' fitted coefficients from themselves redefining the opportunity metric.

The Conditional Information Inflation Factor is computed from each individual conditional-logit information matrix,

\[
\operatorname{CIIF}_k=I_{kk}[I^{-1}]_{kk}.
\]

`CIIF ~= 1` means most information for that predictor is unique. Values above one indicate that predictor information overlaps with other fitted covariates; `sqrt(CIIF)` is the corresponding standard-error inflation factor. CIIF is an identifiability/design diagnostic and is kept separate from temporal/bootstrap uncertainty.

## Hierarchical Bayesian SSF

```python
fit = analysis.fit(
    sample_kwargs={
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.95,
    }
)

fit.coefficients()
fit.individual_coefficients()
fit.heterogeneity()
fit.choice_scores()["summary"]
```

For individual `i`, predictor `k`, stratum `s`, and candidate `j`, the model is

\[
\beta_{ik}=\mu_k+\sigma_k z_{ik},\qquad z_{ik}\sim N(0,1),
\]

\[
\eta_{sj}=x_{sj}^{T}\beta_{i[s]},
\qquad
Y_s\sim\operatorname{Categorical}(\operatorname{softmax}(\eta_s)).
\]

There is no stratum intercept: a constant added to every candidate in a choice set cancels under the softmax.

A pilot can reuse the scaling of a trusted reference fit so its posterior coefficients stay on exactly the same standardized scale:

```python
pilot_fit = pilot.fit(
    scaling=fit.scaling,
    sample_kwargs={"draws": 1000, "tune": 1000},
)
```

Cross-validation never uses externally supplied scaling: it always learns scaling from the training partition only.

### Posterior diagnostics and forest plots

Bayesian SSF fits expose the same compact plotting workflow as Bayesian RSFs:

```python
diagnostics = fit.plot_diagnostics(
    forest_predictors=["elevation", "slope", "vrm_2070m", "wind_support"],
    ci_prob=0.95,
)
```

The returned dictionary contains:

- `trace`: posterior-distribution and chain-trace diagnostics for `mu_beta` and `sigma_beta`;
- `population_forest`: population-average selection coefficients `mu_beta` across predictors;
- `heterogeneity_forest`: between-individual standard deviations `sigma_beta`;
- `individual_forests`: one partially pooled individual-coefficient forest for each requested predictor.

The three forest levels have different meanings and are intentionally kept separate. `mu_beta` describes the population-average selection relationship, `beta` describes individual-specific partially pooled selection coefficients, and `sigma_beta` describes the magnitude of between-individual heterogeneity rather than a signed selection effect.

Forest plots can also be requested without constructing trace plots:

```python
forests = fit.plot_forest(
    predictors=["elevation", "slope"],
    ci_prob=0.89,
)

forests["population"]
forests["heterogeneity"]
forests["individual"]["elevation_z"]
```

Raw predictor names and standardized names are both accepted by fitted-object methods. Trace plots stay compact by default; individual traces are opt-in because a model with many animals can otherwise generate a very large panel:

```python
fit.plot_trace(
    include_individual=True,
    individuals=["BG1018_Kika", "BG1053_Bwindi"],
)
```

## Predictive validation

All SSF validation schemes use the same proper conditional log-score scale:

\[
G_s=\log p_{\text{model}}(y_s)-\log(1/J_s).
\]

`G > 0` is better than uniform choice among `J` alternatives and `exp(mean(G))` is reported as `predictive_advantage`.

```python
from hsa.ssf import LeaveOneIndividualOut, TemporalBlockCV

loio = analysis.validate(
    LeaveOneIndividualOut(n_train_per_id=1000, n_test_per_id=1000),
    sample_kwargs={"draws": 600, "tune": 800, "target_accept": 0.95},
)

temporal = analysis.validate(
    TemporalBlockCV(
        n_blocks=5,
        holdout_block=2,
        embargo="24h",
        n_train_per_id=1000,
        n_test_per_id=500,
    ),
    sample_kwargs={"draws": 800, "tune": 1000, "target_accept": 0.95},
)
```

For a fitted Bayesian model, `fit.psis_loo()` provides fast stratum-wise PSIS-LOO plus Pareto-k diagnostics. It is intentionally interpreted as new-stratum/known-individual prediction, whereas exact temporally blocked CV measures temporal stability and LOIO measures transfer to a new animal. On very large full-data fits, pointwise PSIS can require substantial memory because the log-likelihood matrix has one value per posterior draw and stratum; the package therefore leaves the decision to run full-data PSIS explicit.

## Toward iSSF

The conventional SSF uses the fitted movement kernel to construct availability and estimates habitat-selection effects conditional on that proposal. The choice table also retains `step_length`, `turn_angle`, movement transforms and `proposal_logpdf`. These columns are intentionally preserved for a subsequent integrated-SSF model in which movement and habitat selection are estimated jointly rather than treating the proposal kernel as fixed.
