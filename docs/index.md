# hrHSA

**hrHSA** (_High-resolution Habitat Selection Analysis_) is a scalable Python framework for rigorous statistical inference from spatially and temporally resolved observations. It integrates Lagrangian trajectory data with heterogeneous, time-varying Eulerian environmental fields to quantify how spatial structure, individual variation, stochastic dynamics, and changing conditions shape the probability distribution of observed locations. The framework combines resource-selection analysis, spatial point-process models, hierarchical frequentist and Bayesian inference, mechanistic movement models, and predictive simulation, with explicit treatment of availability, observation processes, uncertainty, validation, and prediction. This provides a unified statistical framework for studying multiscale interactions between moving entities and dynamic spatial fields. 

Developed primarily for wildlife telemetry and spatial ecology, **hrHSA** addresses the more general problem of inferring interactions between actively or passively redistributing entities and their spatial context from a finite number of observations. In this broader perspective, trajectory-based habitat-selection problems belong to a wider class of spatiotemporal inference and inverse problems involving stochastic redistribution in heterogeneous environments.The underlying concepts therefore connect naturally to applications in geospatial and Earth-system science, epidemiology, environmental modelling, human mobility, transport and dispersal processes, and other fields concerned with spatial point patterns, trajectories, dynamic fields, and coupled movement–environment systems.

**hrHSA** provides reproducible, vectorized, out-of-core, and HPC-ready workflows that support analyses from high-resolution environmental annotation and parameter estimation to hierarchical inference, validation, prediction, and large-scale mechanistic simulation. Its computational architecture was designed to scale with the resources available, minimizing unnecessary memory traffic and computational overhead so that complex analyses remain feasible on conventional workstations while efficiently exploiting many-core and HPC environments when available.

```{thumbnail} assets/intro/hrHSA-summary.png
:width: 100%
:alt: Movement observations are combined with spatially and temporally resolved environmental data and transformed into model-ready resource- or step-selection datasets. hrHSA supports frequentist and hierarchical Bayesian RSF, SSF and iSSF estimation through a common analysis framework, from which fitted models can be used for spatial prediction, simulation and inference on individual and population-level effects. Integrated validation and diagnostic workflows include resampling, leave-one-individual-out and temporal cross-validation, as well as posterior predictive assessment.
:align: center
```

```{toctree}
:maxdepth: 1
:caption: Documentation
:hidden:
getting_started
theory
implementation
performance
case-studies
acknowledgements
```
