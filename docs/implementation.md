# Implementation

hrHSA is implemented as a scientific analysis framework in which the **scientific model** and the **execution model** are deliberately separated. The scientific model determines how telemetry, availability, environmental conditions, movement, individual variation, and validation enter an analysis; the execution model determines how those same operations are evaluated on a notebook, workstation, or distributed HPC system. This separation allows the statistical formulation to remain stable while data access, memory management, parallelization, and numerical kernels are optimized independently.

The package therefore extends beyond a collection of RSF, SSF, and iSSF fitting functions. It provides stateful frequentist and Bayesian workflows, movement-informed availability, static and dynamic environmental annotation, hierarchical individual effects, predictive validation and uncertainty diagnostics, persistent analytical datasets, and both reference and accelerated execution paths. High-level analysis and result objects retain the scientific state required to reproduce an analysis, whereas lower-level numerical and geospatial kernels remain public and independently testable.

Three principles guide the implementation. First, fitted analyses must retain the information required to reproduce their design: coefficients alone are insufficient without fitted scaling, feature definitions, availability rules, environmental support, movement definitions, and validation provenance. Second, biologically distinct analyses remain distinct abstractions. RSF, SSF, and iSSF workflows share infrastructure where their scientific assumptions overlap, but they are not forced into a common inheritance hierarchy. Third, computational optimization must preserve the inferential problem. Reference implementations therefore remain available beside accelerated kernels, prepared data are validated before reuse, and invalid shortcuts generally raise rather than silently changing the analysis.

## Part I — Scientific and software architecture

### 1. Analysis objects, functional kernels, and explicit state

The public API combines a stateful object layer with lower-level functional kernels. Analysis classes orchestrate complete workflows and own the associated scientific state, whereas functions perform the numerical and statistical operations beneath them. This keeps high-level analyses concise without making the object interface the only route through the implementation.

The class hierarchy is intentionally shallow. RSF analyses share a common analysis abstraction, with frequentist and Bayesian variants returning corresponding fitted objects. Validation is composed with an estimator rather than encoded through increasingly specific subclasses. Movement-based analyses follow the same architectural principle but do not inherit from the RSF hierarchy, because their availability processes and likelihoods differ fundamentally. An RSF compares observed locations with a broader representation of available space, an SSF conditions alternatives on the current position and recent movement, and an iSSF additionally estimates the movement process jointly with habitat selection. Shared software infrastructure is therefore reused without obscuring these biological distinctions.

A fitted object contains substantially more than a coefficient vector. It retains the fitted transformation of the predictors, the model specification, design-matrix metadata, environmental definitions, availability or choice-set configuration, individual identifiers, prediction metadata, and validation provenance. Bayesian fits additionally retain the probabilistic model, posterior `InferenceData`, hierarchical metadata, and diagnostic information. This state ownership prevents, for example, a prediction from using the scaler from another fit, a validation fold from inheriting full-data scaling, or a raster surface from being generated with a feature expansion different from the model that was fitted.

The corresponding architectural rule is simple:

> **Analysis and result objects own scientific state; functional kernels perform numerical work.**

Low-level functions remain public for testing, benchmarking, advanced workflows, and backwards compatibility.

### 2. Geospatial and analytical data model

Telemetry is represented primarily as `GeoDataFrame` objects, while environmental fields use labelled `xarray.DataArray` or `xarray.Dataset` objects that retain dimensions, coordinates, and spatial metadata. A static environmental stack is generally represented as

```text
band × y × x
```

whereas dynamic datasets add one or more temporal dimensions. Model specifications are propagated into the environmental operations so that only predictors required by the current analysis are read, sampled, or projected. A six-predictor model should not require every band in a much larger remote-sensing archive to enter memory or the execution graph.

Spatial operations enforce the assumptions implied by their geometry. Distance-based quantities require coordinate systems with meaningful distance units, whereas directional environmental fields are treated in geographic space when their interpretation is defined relative to true east and north. This distinction is particularly important for atmospheric vector fields such as wind, because the axes of a projected raster are not necessarily aligned with geodesic directions.

The same principle applies to temporal environmental information. Dynamic predictors are evaluated at the relevant candidate location and time rather than inherited from the observed endpoint or step origin. Environmental conditions can therefore remain genuinely candidate-specific within a choice set. Static, dynamic, scalar, and directional predictors ultimately enter a common model-ready representation, but their spatial and temporal semantics are preserved during annotation.

### 3. Scientific safeguards

hrHSA generally prefers an explicit error to a plausible result produced from an invalid design. Scientific assumptions are therefore encoded as implementation constraints rather than left to convention.

For step-selection analyses, available endpoints falling outside environmental support are regenerated rather than snapped to the raster boundary. Observed endpoints are never moved. If an observed endpoint lacks required environmental support, the default behaviour is to raise; when exclusion is explicitly requested, the entire stratum is removed and its identity retained in diagnostics. Non-finite predictor values are handled similarly: individual alternatives are not silently removed from an otherwise retained choice set because doing so would change the conditional choice experiment.

Validation is also isolated from information that belongs to the complete dataset. Predictor scaling is estimated within each training partition rather than inherited from the full fit, and held-out observations retain the availability or choice structure appropriate to the validation question. The same principle currently defines the validation boundary for iSSF analyses: a valid held-out iSSF fold must estimate scaling from the training data while preserving the proposal correction attached to the held-out choice sets. Until those operations can be performed together consistently, the public validation path raises rather than silently applying the simpler SSF procedure.

The broader implementation principle is that convenience does not override the inferential design. The software may refuse an operation that appears computationally possible when its statistical interpretation is not sufficiently well defined.

## Part II — Statistical model implementations

### 4. Resource-selection functions

RSF workflows estimate relative use of environmental conditions by comparing observed locations with sampled available locations. The implementation separates three stages: construction of the ecological design, statistical inference, and prediction or validation. Frequentist and Bayesian analyses share the same predictor semantics and environmental preparation but retain model-specific inference state.

Predictor transformations are declared through a feature specification containing linear, quadratic, interaction, and categorical terms. Continuous predictors are standardized during fitting, after which nonlinear terms are generated on the fitted scale. Column ordering, category levels, and scaling parameters are retained by the fit so that point prediction, raster projection, and validation reconstruct exactly the same design matrix.

The frequentist implementation fits the resulting logistic model using optimized numerical routines. The hierarchical Bayesian formulation extends the same design to population and individual effects using non-centred random effects. Repeated observations sharing the same predictor vector may be represented through an aggregated Binomial likelihood,

$$
y_g
\sim
\operatorname{Binomial}(n_g,p_g),
\qquad
\operatorname{logit}(p_g)
=
x_g^\top\beta,
$$

which can replace large numbers of identical Bernoulli contributions without changing the likelihood.

Regularization is treated as part of the statistical model rather than as an automatic feature-selection procedure. Small biologically defined predictor sets use weakly informative standardized priors by default, whereas larger correlated candidate sets can optionally use a regularized horseshoe. When an expected number $m_0$ of coefficients escaping strong shrinkage is supplied, the initial global scale is

$$
\tau_0
=
\frac{m_0}{p-m_0}
\frac{1}{\sqrt{n_{\mathrm{eff}}}}.
$$

For RSFs, $n_{\mathrm{eff}}$ is based on the number of used locations rather than the analyst-controlled number of used plus available observations. Increasing the availability sampling factor therefore does not automatically impose stronger prior shrinkage.

Validation is implemented compositionally. Leave-one-individual-out analysis evaluates transfer to a new animal by fitting on the remaining individuals and evaluating the held-out animal against its own fixed availability domain. Temporal block methods address temporal stability rather than individual transfer. Post-fit blocked bootstrap procedures resample complete temporal blocks to preserve local dependence; in frequentist analyses they quantify finite validation-sample uncertainty conditional on the fitted coefficients, whereas Bayesian analyses can combine bootstrap resampling with posterior coefficient draws.

### 5. Step-selection functions

SSF workflows condition availability on the current movement state. Each observed step is compared with alternatives originating at the same starting location and sampled from an estimated movement proposal. Choice-set construction, environmental annotation, inference, and validation are therefore implemented as distinct stages so that both movement and environmental assumptions can be inspected before fitting.

Available step lengths are sampled from fitted individual-specific movement distributions. Where a maximum speed is imposed, the proposal is truncated through its distribution rather than by generating impossible steps and clipping them afterwards. Each candidate retains its stratum and individual identity together with step length, turn angle, movement transforms, bearing, and proposal density. The resulting choice sets can subsequently be reused by both SSF and iSSF analyses.

Static predictors are evaluated at every candidate endpoint. Time-varying environmental fields are sampled independently for each candidate at the appropriate time, and vector fields can be transformed relative to the candidate movement bearing. For a wind vector, for example, this yields directional components such as support along the movement direction, crosswind, and alignment. Dynamic fields can also be inspected outside model fitting so that units, sign conventions, temporal evolution, and transformations can be evaluated before entering the statistical analysis.

Frequentist SSF inference uses a conditional logistic likelihood with one selected endpoint per stratum and no stratum-specific intercept. Predictor scaling is fitted from the choice table itself, and missing values are handled at the complete-stratum level. The amount of information available for a coefficient depends on contrast among alternatives within strata. For predictor $k$, the conditional information contribution can be written as

$$
I_{s,k}
=
\operatorname{Var}_{p_s}
\left(
x_{s,j,k}
\right),
$$

so low within-stratum environmental variation primarily limits coefficient precision rather than implying weak ecological preference.

The hierarchical Bayesian SSF represents the same one-of-$J$ choice process directly. Individual coefficients are written as

$$
\beta_{ik}
=
\mu_k+\sigma_k z_{ik},
\qquad
z_{ik}\sim\mathcal N(0,1),
$$

and the utility of candidate $j$ in stratum $s$ is

$$
\eta_{sj}
=
x_{sj}^{\top}\beta_{i[s]}.
$$

Choice probabilities follow the softmax over candidates within each stratum. The fitted object keeps population-average effects, partially pooled individual effects, and between-individual heterogeneity distinct because these quantities answer different ecological questions.

Predictive validation uses the conditional log score

$$
G_s
=
\log p_{\mathrm{model}}(y_s)
-
\log(1/J_s),
$$

where $J_s$ is the number of alternatives in the stratum. Leave-one-individual-out validation measures transfer to new individuals, temporal blocks measure stability through time, and fitted-model PSIS-LOO evaluates prediction to new strata for individuals already represented in the hierarchy. These validation questions are retained as distinct procedures rather than reduced to one generic score.

### 6. Integrated step-selection functions

iSSF analyses use the same movement-informed choice sets as SSFs but additionally estimate the movement process itself. The implementation therefore separates the proposal used to generate available alternatives from the movement kernel represented by the fitted model.

Candidate generation retains the log density of the proposal distribution. During fitting, the utility includes the fixed correction

$$
-\log q,
$$

where $q$ is the proposal density, preventing the sampling distribution used to generate alternatives from being conflated with the ecological movement process being estimated. Standard movement terms include step length, log step length, and the cosine of the turn angle.

Selection and movement effects are declared separately. Candidate-varying environmental predictors can enter directly as selection terms, whereas environmental conditions measured at the beginning of a step are constant within a choice set and cannot contribute identifiable conditional-logistic main effects. Such variables instead modify movement through interactions with step length or turning terms. The model specification retains whether each environmental quantity varies among candidates or is constant within a stratum and expands the corresponding movement interactions consistently.

The workflow is deliberately staged:

```text
telemetry
   ↓
movement-informed choice sets
   ↓
static / dynamic environmental annotation
   ↓
selection + movement model specification
   ↓
design preparation
   ↓
frequentist or hierarchical Bayesian inference
```

This allows the ecological model to be revised without regenerating candidate steps or repeatedly resampling expensive environmental predictors. Once the choice data are prepared, the same representation can be used by frequentist and hierarchical Bayesian inference.

For long-running Bayesian analyses, model-ready numeric design data and their metadata can be exported independently of large geospatial objects. External scheduler-managed MCMC can then write posterior `InferenceData` to NetCDF and reload the completed fit into the analysis context. Fitted iSSF objects additionally transform model-scale movement coefficients back into quantities such as expected displacement, step-length distributions, and turning-angle distributions under different environmental conditions, because simple linear effects on the utility scale can imply nonlinear responses on the ecological movement scale.

## Part III — Data access, execution, and computational reproducibility

### 7. Storage-aware environmental computation

At high spatial resolution, environmental extraction frequently becomes limited by memory traffic, storage access, and scheduler overhead before floating-point arithmetic becomes limiting. hrHSA therefore treats environmental data access as part of the algorithm.

The readable reference sampler uses labelled xarray indexing. The accelerated path instead maps requested locations to physical or computational raster blocks, groups requests touching the same block, loads the source block once, performs the local NumPy gather, and restores the original point order. This aligns the unit of computation with the unit of storage and prevents repeated access to the same chunk.

For $B$ selected predictor bands, spatial chunk dimensions $n_x,n_y$, and $d$ bytes per value, the first-order block memory requirement is approximately

$$
M_{\mathrm{chunk}}
\approx
B n_x n_y d.
$$

The qualifier *selected* is important: model-relevant bands are identified before chunk planning so unrelated environmental variables do not increase I/O or memory use. Physical storage chunks and computational chunks remain distinct execution parameters because the optimal relationship depends on the underlying storage system.

Dynamic environmental datasets add a temporal routing problem. Grouping requests by time alone can produce geographically large reads when individuals are spatially dispersed. Dynamic extraction can therefore partition requests jointly in time and space before applying the required interpolation semantics. This changes the routing of the environmental data without changing the ecological meaning of the predictor.

Expensive environmental transformations can also be persisted once and reused. Prepared analytical datasets store model-ready partitions in formats such as Parquet together with metadata describing predictor definitions, availability sampling, thinning, random seeds, and the extraction configuration. They are treated as persistent analytical products, not opaque caches: reuse is permitted only when the stored scientific configuration is compatible with the requested analysis. A validation run with a different availability ratio, thinning rule, or predictor definition must therefore regenerate the relevant preparation rather than silently reusing incompatible data.

This separation is especially useful for repeated fitting and cross-validation. Once environmental extraction has been performed correctly for each individual, model folds can operate on compact analytical partitions without repeatedly querying large raster archives.

### 8. Execution backends and accelerated analysis paths

Execution is configured independently of the ecological model. The same analysis can therefore be evaluated serially, on a local Dask cluster, with a caller-managed distributed client, or through a scheduler-backed HPC environment without changing its statistical specification.

Conceptually,

```python
from hsa.compute import ExecutionConfig

compute = ExecutionConfig(
    backend="local",
    n_workers=8,
    threads_per_worker=1,
    chunk_mb=256,
)
```

defines how the computation is executed rather than what scientific question is fitted. Scheduler accounts, partitions, module commands, worker placement, and native BLAS/OpenMP thread pools remain outside the model definition. External services are similarly initialized explicitly: importing `hsa` does not authenticate remote Earth-observation services or create a distributed cluster.

Dense surface prediction uses a fused blockwise execution path. The reference implementation expresses standardization, nonlinear transformations, and model evaluation as transparent labelled xarray operations. The accelerated implementation evaluates the complete fitted model within each spatial block, so temporary standardized predictors, interactions, and masks remain block-local rather than becoming raster-sized intermediate arrays. Linear standardization can additionally be absorbed algebraically into fitted coefficients,

$$
\beta
\frac{x-\mu}{\sigma}
=
\frac{\beta}{\sigma}x
-
\frac{\beta\mu}{\sigma},
$$

reducing memory traffic further. Temporary workspaces are reused where possible, while the reference engine remains available as a correctness target.

Independent scientific units provide a second level of coarse-grained parallelism. LOIO folds, Bayesian chains, candidate models, bootstrap replicates, and independent individual fits can be evaluated concurrently when their communication requirements are small relative to their computational cost. Distributed validation therefore returns compact diagnostics by default rather than large fitted objects or complete fold-specific prediction surfaces. Nested uncontrolled Dask execution is avoided: if a distributed validation task itself touches a Dask-backed environmental dataset, the inner spatial operation is executed within the task rather than recursively submitting another competing graph.

The performance consequences of chunk size, worker geometry, process/thread decomposition, storage layout, and parallelism level are treated separately in [Performance engineering and benchmarking](performance.md).

### 9. Correctness and computational reproducibility

Performance engineering is treated as part of the implementation rather than as an informal timing exercise. An accelerated path is accepted only after its scientific output has been compared with the corresponding reference formulation.

For environmental extraction, deterministic reference and accelerated results are compared directly. Frequentist model optimizations are checked using likelihoods, coefficient estimates, predictions, and validation summaries. Bayesian acceleration is evaluated using posterior agreement and predictive diagnostics in addition to runtime. A sampler that produces more draws per second but materially changes the posterior is not considered an optimization.

Bayesian performance is therefore assessed through quantities such as wall time, bulk and tail effective sample size per second, $\hat R$, divergences, NUTS depth or step counts, posterior agreement, and held-out predictive agreement rather than raw sampling throughput alone. Large posterior intermediates are retained only when scientifically necessary; quantities required transiently during likelihood evaluation are not automatically stored for every posterior draw if doing so would dominate posterior memory.

Scientific and computational reproducibility are treated as one provenance problem. A reproducible analysis should retain the package and dependency versions, source-data provenance, coordinate reference systems, telemetry filtering or thinning, availability definitions, random seeds, feature and model specifications, fitted scaling, environmental-support rules, validation partitions, Bayesian sampler settings, and - when performance is reported - the execution configuration itself. A faster result is not equivalent if the optimization also changed predictor precision, availability, proposal construction, environmental support, or validation folds without recording the change.

The source tree reflects the same separation of concerns:

```text
src/hsa/
├── sampling.py          # reference spatial sampling operations
├── features.py          # reproducible feature construction
├── types.py             # shared scientific specifications
├── rsf/                 # RSF analysis, inference, prediction, validation
├── ssf/                 # SSF/iSSF choice sets, inference, diagnostics
├── movement/            # trajectory and movement-kernel utilities
├── diagnostics/         # shared diagnostic infrastructure
├── compute/             # execution, chunking, prepared data, I/O
├── remote_sensing/      # optional environmental-data interfaces
└── simulation/          # forward-simulation components
```

Optional functionality is kept outside the core dependency set so that Bayesian samplers, distributed execution, HPC I/O, and remote-sensing services are installed only where required. Statistical modelling, environmental data preparation, and execution infrastructure can consequently evolve independently while remaining composable through the public analysis objects.

## Summary

hrHSA is organized around explicit scientific state and replaceable computational kernels. RSF, SSF, and iSSF retain the distinctions imposed by their availability processes and likelihoods while sharing environmental annotation, diagnostics, validation infrastructure, and execution services where those concepts are genuinely common. Frequentist and Bayesian fit objects preserve the scaling, design, hierarchical structure, and validation provenance required to reconstruct the analysis rather than exposing only numerical coefficients.

The same separation extends to computation. Environmental extraction can be prepared once and reused when scientifically compatible, raster operations are aligned with storage and evaluated blockwise, and distributed execution is configured independently of the ecological model. Accelerated kernels remain testable against readable reference implementations, allowing computational scale to increase without changing the inferential question.

The central implementation principle is therefore that increasing ecological realism and computational scale should not require sacrificing inspectability. Scientific state remains explicit, ecological design remains distinct from execution, and performance optimizations are accepted only when they preserve the statistical and biological meaning of the analysis.