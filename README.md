<p align="center">
  <img src="logo/logo_rsf2.png" width="600" alt="hrHSA logo">
</p>

<h1 align="center">hrHSA</h1>

<p align="center">
  <strong>High-resolution Habitat Selection Analysis</strong><br>
  Scalable inference for animal movement, habitat selection, and environmental dynamics.
</p>

<p align="center">
  <a href="https://hrhsa-public.readthedocs.io/en/latest/">Documentation</a>
  ·
  <a href="https://hrhsa-public.readthedocs.io/en/latest/getting_started.html">Getting started</a>
  ·
  <a href="https://hrhsa-public.readthedocs.io/en/latest/performance.html">Performance</a>
  ·
  <a href="https://github.com/PantheraPaulus/hrHSA/issues">Issues</a>
</p>

---

**hrHSA** is a Python framework for modelling how animals use and move through heterogeneous environments.

It integrates animal telemetry with static and time-varying environmental data to fit **Resource Selection Functions (RSFs)**, **Step Selection Functions (SSFs)**, and **integrated Step Selection Functions (iSSFs)** using frequentist or hierarchical Bayesian inference.

The package is designed around a separation between the **ecological model** and the **execution model**. Availability, movement constraints, individual variation, and environmental hypotheses define the scientific analysis; raster access, parallelisation, and memory management determine how that analysis is executed. The same workflow can therefore move from a workstation to large out-of-core and HPC workloads without changing its ecological interpretation.

## What hrHSA provides

| | |
|---|---|
| **Resource selection** | RSFs with explicit availability, nonlinear terms, interactions, spatial prediction, and predictive validation |
| **Movement-conditioned selection** | SSFs with individual movement-informed choice sets and conditional-choice inference |
| **Integrated movement models** | iSSFs combining habitat selection, movement kernels, proposal correction, and environmentally modified movement |
| **Hierarchical inference** | Population and individual responses through partial pooling, heterogeneity estimation, and Bayesian shrinkage |
| **Dynamic environments** | Static and time-varying raster predictors, including directional environmental fields |
| **Validation** | Leave-one-individual-out, temporal blocking, bootstrap uncertainty, conditional-choice scoring, and posterior diagnostics |
| **Spatial computation** | Storage-aware raster sampling, blockwise prediction, and bounded-memory processing |
| **Scalable execution** | Local, Dask, and SLURM workflows with reproducible benchmarking tools |

## Installation

hrHSA requires Python 3.10 or newer.

```bash
git clone https://github.com/PantheraPaulus/hrHSA.git
cd hrHSA

pip install -e .
```

Optional functionality is installed through extras:

```bash
# Hierarchical Bayesian inference
pip install -e ".[bayesian]"

# Dask / SLURM execution
pip install -e ".[hpc]"

# Google Earth Engine integration
pip install -e ".[earthengine]"

# Combined scientific/HPC installation
pip install -e ".[hpc,bayesian,earthengine]"
```

For development with Conda:

```bash
conda env create -f environment.yml
conda activate hsa
```

The distribution is named **hrHSA**, while the Python import namespace remains:

```python
import hsa
```

## Quick start

A resource-selection analysis begins by defining the ecological hypothesis rather than the computational backend.

```python
from hsa import FeatureSpec
from hsa.rsf import FrequentistRSF

spec = FeatureSpec(
    linear=[
        "ndvi",
        "slope",
        "distance_to_water",
    ],
    quadratic=[
        "distance_to_water",
    ],
)

rsf = FrequentistRSF(
    reloc,
    env,
    spec=spec,
    id_col="Individual_ID",
)

fit = rsf.fit(
    sampling_factor=10,
    thin_dt="12h",
)
```

The fitted object retains the feature specification, scaling, model metadata, and availability design required for consistent inference and prediction:

```python
fit.coefficients()

surface = fit.predict_surface()
surface.plot()
```

Predictive transfer to new individuals can be evaluated without changing the fitted analysis definition:

```python
from hsa.rsf import LeaveOneIndividualOut

cv = rsf.validate(
    LeaveOneIndividualOut(
        heldout="all",
        sampling_factor_train=10,
        n_background=100_000,
        seed=42,
    )
)
```

The complete workflow, including data preparation and interpretation, is developed in the
[Getting started guide](https://hrhsa-public.readthedocs.io/en/latest/getting_started.html).

## From RSF to iSSF

The three principal model families differ primarily in how they define the alternatives against which observed movement is compared.

```text
RSF
broad spatial availability
        ↓
relative selection across a landscape

SSF
movement-constrained local alternatives
        ↓
relative choice among reachable endpoints

iSSF
movement-constrained alternatives
+ proposal correction
        ↓
joint habitat-selection and movement inference
```

This common structure allows analyses to become progressively more mechanistic without changing the fundamental interpretation of habitat selection as **observed use relative to biologically meaningful alternatives**.

The theoretical foundations are described in the
[Theory](https://hrhsa-public.readthedocs.io/en/latest/theory.html)
documentation.

## Hierarchical Bayesian inference

Animal telemetry typically contains many observations from comparatively few individuals. hrHSA therefore supports hierarchical Bayesian RSF, SSF, and iSSF models in which population-level effects, individual responses, and between-individual heterogeneity are estimated jointly.

Partial pooling allows strongly informed individuals to remain distinct while shrinking uncertain individual estimates toward the population distribution. Regularising priors can additionally stabilise larger correlated predictor sets, while posterior sampling propagates uncertainty into nonlinear predictions and derived movement responses.

This makes it possible to distinguish questions such as

```text
What is the population-average response?

How strongly do individuals differ?

Which responses are shared across animals?

How uncertain are derived ecological predictions?
```

rather than reducing a population to a single coefficient vector.

## High-resolution environmental data

hrHSA is designed for analyses in which environmental extraction and prediction can become as computationally demanding as statistical inference itself.

Environmental data remain in labelled `xarray` structures, while accelerated routines use storage-aware access, block-local sampling, and bounded-memory prediction to avoid materialising unnecessarily large intermediate arrays.

Optional remote-sensing utilities support the construction of predictor stacks from external environmental data sources, including Google Earth Engine.

## Scaling from workstation to HPC

The ecological analysis is independent of where it is executed. hrHSA provides reference and accelerated computational paths together with local, Dask, and SLURM execution utilities for workloads ranging from ordinary workstation analyses to very large raster and telemetry datasets.

The public repository includes the scripts used to profile and benchmark these execution strategies under:

```text
benchmarking/scripts/
├── 01_python/
├── 02_slurm/
├── 03_sh/
└── 04_config/
```

Execution geometry is deliberately treated as workload- and architecture-specific rather than assuming that one worker layout is universally optimal.

For transparency and reproducibility, the complete profiling methodology, scaling experiments, and practical guidance for calibrating hrHSA to a given system are documented in
[Performance engineering and benchmarking](https://hrhsa-public.readthedocs.io/en/latest/performance.html).

## Case studies

The documentation develops the framework through three ecological applications of increasing complexity:

**Ground pangolins** demonstrate how optical and radar Earth observation can be transformed into ecological predictors of habitat structure and analysed using population-level and hierarchical RSFs.

**African lions** illustrate seasonally varying habitat selection, where the response to a static landscape feature changes through time and among individuals.

**Bearded vultures** extend the framework to dynamic SSF and iSSF models in which atmospheric conditions, terrain, and movement interact to generate a changing energetic landscape.

See the
[Case studies](https://hrhsa-public.readthedocs.io/en/latest/case-studies.html)
for the complete analyses.

## Documentation

The full documentation covers the scientific and computational framework in substantially greater detail:

- [Getting started](https://hrhsa-public.readthedocs.io/en/latest/getting_started.html)
- [Theory](https://hrhsa-public.readthedocs.io/en/latest/theory.html)
- [Implementation](https://hrhsa-public.readthedocs.io/en/latest/implementation.html)
- [Case studies](https://hrhsa-public.readthedocs.io/en/latest/case-studies.html)
- [Performance engineering and benchmarking](https://hrhsa-public.readthedocs.io/en/latest/performance.html)

The public repository contains the package source, documentation, tests, and reproducibility tooling. Empirical telemetry datasets and project-specific research material are not distributed with the package.

## Citation

If you use hrHSA in scientific work, please cite the software and the corresponding methodological publication where appropriate. Machine-readable citation metadata are provided in `CITATION.cff`.

## Acknowledgements

Development of **hrHSA** was supported by the
[Competence Network for Scientific High Performance Computing in Bavaria (KONWIHR)](https://www.konwihr.de/).

Large-scale profiling and benchmarking used infrastructure operated by the
[Leibniz Supercomputing Centre (LRZ)](https://www.lrz.de/en/),
including **CoolMUC-4** and the **terrabyte** High Performance Data Analytics platform operated jointly with the German Aerospace Center (DLR).

## License

hrHSA is distributed under the **BSD 3-Clause License**. See [`LICENSE`](LICENSE) for details.