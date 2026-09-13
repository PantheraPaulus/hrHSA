# Getting started

This guide follows one complete hrHSA analysis from telemetry to ecological inference. We will ask how environmental conditions shape the relative use of space, encode that hypothesis as a resource-selection function (RSF), evaluate whether the inferred relationship transfers among individuals, and finally project the fitted model back across the landscape.

The workflow is deliberately simple:

```text
telemetry + environment
          ↓
define the ecological hypothesis
          ↓
define availability
          ↓
fit the model
          ↓
inspect and validate
          ↓
spatial prediction
```

The purpose of this page is to introduce the scientific workflow rather than every available option. The model-specific documentation describes alternative availability definitions, hierarchical effects, dynamic environmental data, diagnostics, and large-scale execution in greater detail.

## Installation

Clone the repository and install hrHSA with its geospatial and distributed-computing dependencies:

```bash
git clone <repository-url>
cd hrHSA
python -m pip install -e ".[hpc]"
```

The examples below use a relocation table and a Zarr-backed environmental raster stack.

```python
import pandas as pd

from hsa import FeatureSpec
from hsa.compute import open_raster_stack_zarr
from hsa.movement.geometry import prepare_trajectory_data
from hsa.rsf import FrequentistRSF, LeaveOneIndividualOut
```

## 1. Prepare the data

hrHSA expects telemetry with an individual identifier, timestamp, and spatial coordinates. Relocations should be represented in a projected coordinate reference system appropriate for the study area whenever the analysis involves distance, area, or spatial availability.

```python
raw = pd.read_csv("data/relocations.csv")

reloc = prepare_trajectory_data(
    raw,
    id_col="Individual_ID",
    timestamp_col="Timestamp",
    lon_col="longitude",
    lat_col="latitude",
    source_crs="EPSG:4326",
    target_crs="EPSG:32733",
    round_freq="h",
    drop_duplicate_fixes=True,
)

reloc.head()
```

`prepare_trajectory_data()` parses timestamps, removes invalid records, orders trajectories, projects the coordinates, and can remove duplicate fixes within the requested temporal resolution.

The environmental predictors are opened as a labelled `xarray` object:

```python
env = open_raster_stack_zarr(
    "data/environment.zarr",
    name="env",
    chunks=None,
)

print(env)
```

The telemetry and environmental raster must refer to the same spatial coordinate system:

```python
assert reloc.crs == env.rio.crs
```

At this stage it is worth checking the spatial extent of the telemetry, the environmental variables and their units, and the amount of missing data. Errors discovered here are generally easier to interpret than failures encountered after model fitting.

## 2. Encode the ecological hypothesis

An RSF compares environmental conditions at observed locations with conditions considered available to the animal. Availability is therefore part of the ecological model rather than merely a computational sampling step: changing the available domain changes the comparison being made.

hrHSA keeps the availability configuration together with the fitted analysis. In this introductory workflow, we will sample ten available locations for each retained used location. More elaborate definitions of accessibility and availability are described in the RSF documentation.

The environmental hypothesis is expressed through a `FeatureSpec`. Suppose we expect selection to vary with vegetation productivity, terrain slope, and distance to water, with a nonlinear response to distance from water:

```python
spec = FeatureSpec(
    linear=[
        "ndvi",
        "slope",
        "distance_to_water",
    ],
    quadratic=[
        "distance_to_water",
    ],
    add_const=True,
)
```

This specification defines the statistical design explicitly. Continuous predictors are standardized during model fitting, and quadratic or interaction terms are subsequently constructed on that fitted scale. The scaling and design metadata remain attached to the fitted model so that validation and spatial prediction reproduce the same transformation.

The important point is that the model is defined **before** comparing fit statistics. Predictor sets, nonlinearities, and interactions should represent ecological hypotheses rather than combinations generated solely because they are computationally available.

## 3. Fit the resource-selection function

Create a frequentist RSF analysis from the telemetry, environmental data, and model specification:

```python
rsf = FrequentistRSF(
    reloc,
    env,
    spec=spec,
    id_col="Individual_ID",
)
```

The analysis object owns the scientific state of the workflow: telemetry, environmental predictors, feature definition, availability configuration, fitted scaling, and the information required for subsequent prediction and validation.

Fit the model:

```python
fit = rsf.fit(
    sampling_factor=10,
    thin_dt="12h",
)
```

Here, `sampling_factor=10` controls the numerical representation of the available environment, while `thin_dt="12h"` limits the temporal density of used relocations entering this particular analysis. Neither parameter should be interpreted as a purely computational setting: both affect the data supplied to the statistical comparison and should therefore be reported with the fitted model.

Inspect the estimated coefficients:

```python
fit.coefficients()
```

For a linear standardized predictor, a positive coefficient indicates increasing relative selection with increasing predictor values, conditional on the remaining terms; a negative coefficient indicates decreasing relative selection. Quadratic and interaction terms should be interpreted jointly with the corresponding main effects rather than coefficient by coefficient.

Because the number of available locations is chosen by the analyst, the fitted intercept and raw logistic probabilities generally do **not** represent absolute occurrence probabilities. RSFs estimate contrasts in **relative selection strength** under the specified availability design.

## 4. Inspect the fitted design

Before interpreting the model ecologically, examine the data that generated it. In particular, check whether the environmental range of used locations is represented within the available sample, whether predictors contain substantial missingness, and whether a fitted effect is supported by meaningful environmental variation rather than by a small number of extreme observations.

The lower-level diagnostic functions remain available when direct access to prepared used–available tables is required. They summarize differences between used and available predictor distributions using quantities such as differences in means and medians, the Kolmogorov–Smirnov statistic, Wasserstein distance, and common-language effect size.

These summaries are descriptive. Strong separation between used and available observations can indicate an informative ecological signal, but it can also reveal an implausible availability definition, incomplete environmental support, or influential outliers. Conversely, substantial overlap does not imply the absence of conditional selection once several predictors are considered jointly.

The fitted model should therefore be interpreted together with the ecological design, predictor distributions, coefficient uncertainty, and subsequent predictive validation.

## 5. Evaluate transfer among individuals

Model fit alone does not establish that an inferred relationship generalizes beyond the observations used to estimate it. Randomly splitting autocorrelated relocations can also produce misleadingly optimistic validation results.

For multi-individual datasets, a useful first question is therefore:

> **Does the inferred population-level selection relationship transfer to an individual that was not used for model fitting?**

hrHSA implements this through leave-one-individual-out validation:

```python
loio = LeaveOneIndividualOut(
    heldout="all",
    sampling_factor_train=10,
    n_background=100_000,
    n_bins=20,
    seed=42,
)

cv = rsf.validate(loio)
```

Each fold fits the model without one individual and evaluates the held-out animal against its own availability background. Predictor scaling is estimated independently from the training data in each fold, preventing information from the held-out individual from entering preprocessing.

The returned validation result retains fold-specific diagnostics rather than reducing all animals to one pooled statistic. These can be used to inspect whether performance is consistent across individuals and to identify animals for which the population-level relationship transfers poorly.

Boyce-type diagnostics ask whether observed locations become increasingly common relative to availability as predicted relative selection increases. They should be interpreted alongside calibration, fold-specific coefficients, sample sizes, and the ecological context of each held-out individual.

Leave-one-individual-out validation therefore answers a different question from conventional goodness of fit: it evaluates **transferability to new individuals**.

## 6. Project the fitted model

Once the model and its transferability have been examined, the fitted selection function can be projected over the environmental raster:

```python
surface = fit.predict_surface()

surface
```

The resulting `xarray` object can be plotted directly:

```python
surface.plot(
    robust=True,
    figsize=(9, 7),
)
```

For a log-linear RSF, the relative selection surface is proportional to

$$
w(\mathbf{x})
=
\exp
\left(
\beta_0
+
\boldsymbol{\beta}^{\mathsf T}\mathbf{x}
\right).
$$

The map should therefore be interpreted comparatively. Locations with larger values are predicted to have greater relative selection strength under the fitted environmental relationship and availability design; the surface is not, in general, a map of absolute occurrence probability.

Prediction is only meaningful within environmental conditions for which the fitted model is scientifically defensible. Extrapolation beyond the range represented in the fitting data should therefore be examined explicitly, particularly when nonlinear terms or interactions are present.

## 7. Extend the analysis hierarchically

The frequentist workflow above estimates one population-level selection relationship. Repeated observations from several individuals also allow variation among animals to be estimated explicitly through a hierarchical Bayesian RSF.

The ecological design remains the same. Only the inference model changes:

```python
from hsa.rsf import BayesianRSF

bayes = BayesianRSF(
    reloc,
    env,
    spec=spec,
    id_col="Individual_ID",
)

bayes_fit = bayes.fit(
    sampling_factor=10,
)
```

The hierarchical model separates population-average effects, partially pooled individual effects, and the among-individual variation in those effects. Partial pooling allows individuals with fewer informative observations to borrow information from the population while retaining evidence for genuine heterogeneity.

The Bayesian RSF documentation describes random slopes, prior specification, regularization, posterior diagnostics, prediction, and predictive validation in detail. The important conceptual point is that Bayesian and frequentist RSFs operate on the same ecological comparison; they differ in how uncertainty and individual variation are represented.

## Where to go next

This first analysis introduced the central hrHSA workflow:

```text
telemetry
   +
environment
   ↓
ecological hypothesis
   ↓
availability
   ↓
statistical model
   ↓
validation
   ↓
spatial inference
```

From here, the appropriate extension depends on the ecological question.

**Resource-selection functions** are appropriate when the primary question concerns relative use of environmental conditions across an availability domain. The RSF documentation covers alternative availability definitions, hierarchical effects, uncertainty, and model diagnostics.

**Step-selection functions** condition availability on the animal's current position and recent movement. They are appropriate when habitat selection is to be studied at the scale of individual movement decisions.

**Integrated step-selection functions** additionally estimate the movement process itself, allowing environmental conditions to influence both selection and movement behaviour.

Time-varying environmental fields, including directional quantities such as wind, can be incorporated when environmental conditions vary over the timescale of the observed movement.

For large raster archives, repeated cross-validation, or HPC execution, the same scientific workflow can be evaluated using hrHSA's prepared-data and accelerated execution layers. These computational choices do not redefine the ecological model; their implementation and performance are described separately in the **Implementation** and **Performance engineering and benchmarking** sections.