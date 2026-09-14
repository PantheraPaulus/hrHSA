# Performance

Scientific-computing performance is not determined by core count alone. Runtime emerges from the interaction between an algorithm, its data representation, the physical organization of those data, the execution runtime, and the architecture on which the computation is performed. Increasing computational resources can therefore expose rather than remove bottlenecks: additional processes may increase memory replication and scheduler overhead, wider threading may saturate cache or memory bandwidth, and fine task decomposition may leave a large machine underutilized despite substantial unused memory. Conversely, datasets substantially larger than physical RAM can remain tractable when the computation operates on a bounded active working set.

These considerations are particularly important for high-resolution habitat-selection analysis because a single scientific workflow combines several computational regimes. Environmental information must first be gathered sparsely at observed and available locations; statistical models then repeatedly evaluate structured likelihoods, gradients, or posterior densities; and fitted models may subsequently be projected across complete environmental surfaces. Cross-validation, hierarchical inference, simulation, and individual-level analyses introduce further independent work above these kernels. There is consequently no single scalar notion of problem size, nor a single parallel configuration that can be expected to perform optimally throughout an hrHSA analysis.

hrHSA was therefore developed together with a systematic performance-engineering campaign. Complete workflows were first profiled to identify the dominant computational kernels. Each kernel was then redesigned to reduce unnecessary data movement, allocation, bookkeeping, and arithmetic before additional parallelism was introduced. Controlled experiments subsequently isolated numerical thread scaling, process/thread geometry, storage organization, task granularity, and memory behaviour. The resulting implementation was finally evaluated in large capacity experiments and complete RSF, SSF, and iSSF workflows. The objective was not only to determine which configuration was fastest, but to establish why performance changed and which conclusions could be transferred to other HPC architectures.

## Part I — Benchmark framework

### 1. Profiling performance as an experimental problem

Profiling of complete hrHSA workflows identified three dominant computational kernels: sparse raster sampling, statistical inference, and dense raster prediction. Their computational structures are sufficiently different that optimization must be kernel-specific. Sparse sampling is governed primarily by how irregular point requests map onto physical raster blocks and distributed tasks. Statistical inference repeatedly traverses comparatively compact but potentially large model representations and is sensitive to numerical formulation, cache use, and the placement of concurrency. Raster prediction evaluates essentially every cell and behaves predominantly as a dense streaming calculation in which repeated memory traffic and temporary allocation can outweigh floating-point arithmetic.

The benchmark strategy therefore followed a layered experimental design. We first optimized the algorithm within each kernel, then measured the numerical operation independently from the distributed runtime where this distinction was informative, and only afterwards optimized the mapping onto hardware. The experimental principle was to change one layer at a time: direct numerical-kernel experiments establish useful intraprocess scaling; CPU-placement tests expose cache and locality effects; fixed-core worker experiments characterize runtime geometry; and telemetry distinguishes useful computation from process management, scheduling, transfers, and memory pressure. A benchmark establishes *what* is fast, whereas a mechanism experiment is used to determine *why*.

Three complementary forms of scaling were considered. Capacity experiments increased logical problem size while keeping the available resources fixed and therefore tested whether memory consumption remained bounded. Strong-scaling experiments retained the scientific workload while increasing computational resources,

$$
S_{\mathrm{strong}}(p)
=
\frac{T_1(W)}
{T_p(W)},
\qquad
E_{\mathrm{strong}}(p)
=
\frac{S_{\mathrm{strong}}(p)}
{p},
$$

whereas weak scaling increased workload in proportion to physical resources,

$$
E_{\mathrm{weak}}(p)
=
\frac{T_1(W_1)}
{T_p(W_p)},
\qquad
W_p=pW_1.
$$

Execution-geometry experiments addressed a separate question: how should the same physical-core budget be divided between processes and threads? Keeping these questions distinct was important throughout the campaign because an implementation can have excellent out-of-core capacity while exhibiting poor weak scaling, or strong numerical scaling while performing poorly when embedded in a fragmented distributed runtime.

### 2. Reference systems and benchmark workloads

Most large-scale experiments were performed on a single LRZ CoolMUC-4 node exposing 112 physical cores and 224 logical CPUs. Large spatial campaigns typically used a 224-GiB memory allocation. A 12-core AMD Ryzen 9 3900X workstation with approximately 125 GiB RAM provided a contrasting shared-memory architecture and was used particularly to investigate single-analysis latency, numerical thread width, cache locality, and matched weak scaling.

The two systems were not treated as directly comparable machines unless workload and protocol were harmonized. The workstation contains several relatively small shared-cache domains and reaches its useful intraprocess scaling limit rapidly, whereas the CoolMUC-4 node contains two 56-core sockets and exposes a much wider shared-memory execution environment. Non-harmonized cross-machine experiments were therefore used to compare scaling mechanisms and response shapes rather than to claim an absolute workstation-versus-HPC speed-up.

| Characteristic | Reference workstation | CoolMUC-4 |
|---|---:|---:|
| Processor context | AMD Ryzen 9 3900X | LRZ CoolMUC-4 compute node |
| Physical cores | 12 | 112 |
| Logical CPUs | 24 | 224 |
| Important topology | four small L3/cache domains | two sockets × 56 physical cores |
| Memory | ~125 GiB installed | typically 224 GiB allocated |
| Useful direct-kernel thread width | ~3–4 threads | useful scaling extends through ~14–28 threads and beyond |
| Observed spatial-worker tendency | several narrow workers | fewer, wider workers |
| Principal early limitation | cache/locality pressure | process/runtime fragmentation and shared-memory limits |

The benchmark campaign combined kernel experiments with complete workflows.

| Campaign | Scientific workload | Experimental comparison | Repetition | Primary response |
|---|---|---|---|---|
| Workflow profiling | 20 individuals; ~220k candidates/workflow; 24-GiB raster | RSF, SSF, iSSF | 3 repeats | runtime decomposition |
| End-to-end A/B | 50k steps/individual; 384-GiB raster | reference vs optimized | 2 paired repeats | total wall time |
| Point capacity | $10^6$–$10^9$ queries; 24–768-GiB rasters | increasing point/raster volume | repeated | wall time, throughput, RSS |
| Raster A/B | 128–384-GiB logical rasters | reference vs fused implementation | paired/repeated | wall time, RSS |
| Raster capacity | up to 900.4-GiB six-band raster | fixed 224-GiB allocation | repeated | throughput, aggregate RSS |
| Worker geometry | fixed spatial workload and 112 cores | process × thread sweep | repeated | wall time, memory |
| Inference parallelism | 20 independent individual fits | 1–20 physical cores | repeat-paired | speed-up |
| Spatial weak scaling | 12 → 112 physical cores | workload increased 9.33× | matched | weak-scaling efficiency |
| Storage-layout experiment | identical 225.094-GiB raster | $1024^2$ vs $2048^2$ physical chunks | paired | wall time, task graph, occupancy |

The profiling configuration retained realistic model differences rather than forcing the three model classes into an artificial equal-cost design. RSF used three predictors, SSF six, and iSSF fourteen. Frequentist models were fitted using L-BFGS. Bayesian benchmarks used four chains with 500 warm-up and 500 retained posterior iterations per chain, with the inference backend chosen according to model class.

The matched spatial weak-scaling workload followed the physical-core ratio

$$
\frac{112}{12}=9.33.
$$

The workstation processed 96 million point queries against a nominal 24-GiB raster, while CoolMUC-4 processed 896 million points against a 225.094-GiB chunk-aligned raster,

$$
\frac{896}{96}=9.33,
$$

thereby preserving approximately constant point work per physical core.

### 3. Measurement, telemetry, and evidence boundaries

Benchmark timings were applied to the scientific operation under study rather than unrelated data preparation. Distributed raster-prediction benchmarks forced completion by persisting the distributed result but did not subsequently collect the complete output into one local array. The reported wall time therefore describes distributed computation and partitioned materialization, not an additional global assembly step that would be inappropriate for a several-hundred-gigabyte prediction.

Run-level results were retained in structured JSONL records rather than only as aggregated figures. Depending on the campaign, the recorded metadata include worker count, threads per worker, task count, input dimensions, point count, logical bytes processed, source and computational chunks, materialization policy, repeat identity, wall time, throughput, and process- or worker-level memory measurements. Geometry tests additionally validated that the requested worker topology was actually established before timing. Process-oriented runs constrained hidden OpenMP, MKL, OpenBLAS, and NumExpr thread pools where necessary to prevent nested oversubscription.

Discovery, diagnostic, and production measurements were kept conceptually separate. This distinction is important for the architecture experiments described below: highly instrumented diagnostic runs provide mechanistic evidence about cache behaviour or process fragmentation, whereas final performance claims are based preferentially on the later production-scale campaigns. Warm-up observations are similarly distinguished from measured repeats. Where only two paired production repeats were available, the individual observations remain part of the raw record rather than being hidden behind an unsupported confidence interval.

Benchmark provenance should therefore be regarded as part of the computational methods. At minimum, a reproducible performance result requires the hrHSA revision and software environment; CPU architecture, sockets, NUMA and physical/logical processor counts; scheduler allocation and affinity; worker and thread topology; native numerical-library threading; logical workload dimensions and data type; physical and computational chunking; memory allocation and peak RSS; warm-up and repetition rules; materialization strategy; and the raw run outcome.

The discovery and mechanism history is retained separately from the public summaries, including raw benchmark observations and task-stream telemetry. This separation is deliberate: it allows the development process to remain auditable without presenting every exploratory experiment as a production benchmark.

```{thumbnail} assets/performance/03_profiling.png
:width: 100%
:alt:
This experimental design identified three kernels with distinct computational bottlenecks and scaling behaviour. Point sampling is dominated by sparse access and task orchestration, statistical inference by repeated likelihood evaluation, and dense prediction by traversal of large predictor arrays. The following sections describe the algorithmic changes made to each kernel before returning to their scaling and execution geometry in Part III.
```

## Part II — Optimization of the computational kernels

### 4. Sparse raster sampling: storage-aware access

Habitat-selection analyses require environmental covariates to be extracted at large numbers of observed and available locations. Computationally, this is a sparse gather problem: only a small fraction of cells is requested from raster stores that may contain billions of values and exceed physical memory. A general nearest-neighbour implementation treats locations largely independently and can consequently access the same physical blocks repeatedly. At sufficiently large scale, repeated raster access and the movement of indexing information can become more expensive than the coordinate lookup itself.

hrHSA therefore reorganizes sampling around the physical storage geometry. For regular rasters, point coordinates are converted directly into integer cell positions,

$$
c_i
\approx
\operatorname{round}
\left(
\frac{x_i-x_0}{\Delta x}
\right),
\qquad
r_i
\approx
\operatorname{round}
\left(
\frac{y_i-y_0}{\Delta y}
\right),
$$

while irregular grids retain a general search-based fallback. Cell positions are then mapped to physical raster chunks, all requests touching the same block are grouped, and the requested values are extracted together. Within each block, local row and column positions are flattened,

$$
q_i
=
r_i^{\mathrm{local}} B_x
+
c_i^{\mathrm{local}},
$$

and can normally be represented as `uint32`. This reduced point-index storage from approximately 16 to 4 bytes per query and correspondingly reduced serialization and communication.

The sampler was subsequently made explicitly bounded. Large spatially tiled point collections are processed in batches rather than materialized as one global request, and only a limited number of batches are permitted to remain in flight. Point preparation, raster access, and result assembly can therefore overlap without allowing resident memory to grow with the complete dataset. The cumulative impact of these changes was substantial: arithmetic indexing alone reduced a 100-million-point benchmark from approximately 66.8 to 49.5 s, while the final partitioned block-local implementation processed 100 million queries against a 24-GiB raster stack in approximately 20 s, or about 5 million queries $s^{-1}$.

At HPC scale, profiling revealed a second limitation: even after individual sampling tasks had been optimized, a single global scheduler could become the coordination bottleneck. hrHSA therefore decomposes very large point workloads into independent spatially compact domains using recursive weighted bisection. Point counts are balanced between domains while their corresponding raster footprints are kept compact, reducing both centralized scheduler pressure and unnecessary source access.

On CoolMUC-4, this reduced a one-billion-point, approximately 194-GiB workload from roughly 61–64 s with one global execution domain to 22.5 s with 28 balanced four-core domains. Throughput increased from approximately 16 to 44.5 million point queries $s^{-1}$. Further subdivision to 56 domains slightly increased wall time, showing that the high-performance region had been bracketed and that per-domain orchestration had begun to dominate. The same bounded implementation subsequently processed one billion queries against a nominal 768-GiB raster stack in approximately 56 s, demonstrating that neither the point table nor complete raster stack must reside simultaneously in physical memory.

The optimization of point sampling therefore proceeded across several levels—coordinate arithmetic, compact indexing, storage-aware block grouping, bounded batching, asynchronous concurrency, and finally scheduler-domain decomposition. The final throughput is a consequence of all of these changes rather than of parallelism alone.

### 5. Statistical inference: exploiting known likelihood structure

Statistical inference formed the second dominant kernel. General-purpose statistical implementations are designed to accommodate arbitrary model structures, whereas hrHSA knows important properties of the underlying data in advance. SSF and iSSF strata contain a fixed number of candidate endpoints with exactly one observed choice; RSF datasets frequently contain repeated predictor combinations; and hierarchical analyses repeatedly fit closely related models across individuals. We therefore reduced each likelihood to the smallest representation required by its statistical structure before optimizing execution.

For SSF and iSSF, a stratum $s$ contains $J$ candidate endpoints and one observed choice $y_s$,

$$
P(Y_s=j)
=
\frac{\exp(\eta_{sj})}
{\sum_{k=1}^{J}\exp(\eta_{sk})},
\qquad
\eta_{sj}=x_{sj}^{\top}\beta.
$$

Rather than repeatedly grouping a long observation table, hrHSA reshapes the design once into a dense

```text
stratum × choice × predictor
```

representation and stores a single integer identifying the observed alternative in each stratum. The stratum log-likelihood reduces to

$$
\ell_s
=
\eta_{s,y_s}
-
\operatorname{logsumexp}_{j}(\eta_{sj}),
$$

allowing likelihoods, choice probabilities, analytic gradients, and information matrices to be evaluated directly through vectorized operations.

Bayesian SSF and iSSF models use the same compact representation. Hierarchical coefficients are expressed non-centrally,

$$
\beta_{ip}
=
\mu_p+\sigma_p z_{ip},
\qquad
z_{ip}\sim\mathcal{N}(0,1),
$$

providing partial pooling while improving posterior geometry for NUTS. Large design tensors are retained in `float32` where appropriate, posterior prediction is evaluated in bounded batches, and prediction for a new individual reuses one latent coefficient vector per posterior draw across that individual's strata so that the hierarchical dependence structure is preserved.

Several inference optimizations proved complementary rather than interchangeable. Reducing the memory footprint of the dense choice tensor substantially improved concurrent SSF/iSSF execution. Further restructuring of the coefficient contraction improved the raw log-probability-and-gradient kernel but yielded a smaller gain once embedded in full four-chain NUTS, indicating that another component had become limiting. At that stage, exploiting the known one-of-$J$ structure directly reduced full four-chain Bayesian SSF sampling by a further 15–16% relative to a generic categorical likelihood while retaining stratum-wise log-likelihoods required for posterior diagnostics and PSIS-LOO.

RSFs benefit from a different structural simplification. Observations sharing an identical predictor vector also share the same Bernoulli probability and can therefore be aggregated exactly,

$$
y_g
\sim
\operatorname{Binomial}(n_g,p_g),
\qquad
\operatorname{logit}(p_g)
=
x_g^\top\beta.
$$

Potentially millions of observation-level likelihood contributions can thus be replaced by a much smaller set of unique predictor combinations. Continuous predictors may optionally be discretized before aggregation when a controlled approximation is acceptable, and unnecessary observation-level posterior arrays are omitted unless explicitly requested.

Only after these structural optimizations did we benchmark inference parallelism. This ordering was important: increasing cores around a memory-heavy or unnecessarily general likelihood would have distributed inefficiency rather than removed it.

### 6. Raster prediction: fused dense evaluation

Raster prediction has the opposite data-access pattern to point sampling. Essentially every output cell must be evaluated, so performance is governed less by irregular lookup than by the number of times large predictor arrays are traversed and by the volume of temporary allocation. In the initial implementation, standardization, quadratic terms, interactions, and model evaluation were represented as separate labelled Xarray expressions. This formulation was convenient and transparent, but it could generate multiple raster-sized intermediate arrays and several complete passes through the predictor stack.

The optimized implementation fuses these transformations within storage-aligned spatial blocks. Algebraic simplification is applied before execution wherever possible. For a standardized linear term,

$$
\beta
\frac{x-\mu}{\sigma}
=
\frac{\beta}{\sigma}x
-
\frac{\beta\mu}{\sigma},
$$

allowing the constant contribution to be absorbed into the fitted intercept and avoiding construction of a separate standardized raster. Standardized buffers are retained only for predictors participating in active quadratic or interaction terms. Arithmetic is performed in place where practical, while thread-local workspaces reuse temporary standardization, interaction, and categorical-mask arrays between successive block tasks.

This redesign reduces both allocation pressure and sustained memory traffic. In paired reference-versus-optimized benchmarks, the fused implementation produced a remarkably stable improvement across increasingly large logical rasters:

| Logical raster | Optimized speed-up |
|---:|---:|
| 128 GiB | 1.73× |
| 160 GiB | 1.61× |
| 192 GiB | 1.64× |
| 224 GiB | 1.64× |
| 256 GiB | 1.69× |
| 320 GiB | 1.68× |
| 384 GiB | 1.67× |

At 384 GiB, wall time decreased from approximately **73.1 s to 43.7 s**. The near-constant speed-up across a threefold increase in logical data volume indicates that the optimization acts on sustained memory traffic and repeated allocation rather than only on a small cache-resident special case.

The block kernel was subsequently profiled under alternative worker geometries rather than assuming that one process per physical core should be optimal. On the final production-scale geometry benchmark, an `8×14` layout completed the reference 4.29-billion-cell workload in approximately 9.8–9.9 s, whereas `112×1` required roughly 16.2–16.4 s. The same 112 physical cores therefore completed the operation around 40% faster solely because the process/thread decomposition was changed. The calibrated `8×14` geometry was subsequently used for the largest raster-capacity experiments.

Under a 224-GiB single-node allocation, the optimized predictor successfully evaluated a 900.4-GiB six-band logical raster, approximately four times the allocated memory. Aggregate worker RSS remained near 121–122 GiB, and the two completed largest-scale repeats sustained approximately 333–344 million cells $s^{-1}$. Throughput declined gradually rather than collapsing as logical size crossed physical memory, demonstrating that the predictor behaves as a bounded streaming calculation whose active memory demand is governed by live blocks and output materialization rather than by total logical raster size.

## Part III — Scaling and architecture-aware execution

### 7. Capacity and scaling of the optimized kernels

The final benchmark campaign demonstrates three different scaling regimes corresponding to the three kernels.

Sparse point sampling becomes increasingly throughput-dominated as point count grows. At smaller workloads, raster opening, point routing, and scheduler activity contribute a relatively large fixed component; beyond approximately $10^8$ requests, these costs are increasingly amortized. The final one-billion-query result of 22.5 s and 44.5 million queries $s^{-1}$ represents the combination of storage-aware extraction and scheduler-domain decomposition. Capacity remained bounded as raster size increased, including the one-billion-query experiment against a nominal 768-GiB stack.

Inference instead exhibits nested parallelism. An individual Bayesian fit contains independent chains, whereas a population-scale workflow contains independent model fits above those chains. We benchmarked 20 independent per-individual fits under equal physical-core budgets of

$$
1,\;2,\;4,\;6,\;8,\;10,\;12,\;16,\;20
$$

cores and compared three strategies: independent fits in outer processes, independent fits scheduled through outer threads, and increased native threading within one fit. Every parallel observation was normalized against the serial measurement from the same model and repeat,

$$
S_{p,r}
=
\frac{T_{1,r}}
{T_{p,r}},
$$

before median speed-up was calculated.

| Strategy / model | Approx. best observed speed-up |
|---|---:|
| RSF, outer threads | **~3.4×** |
| SSF, outer threads | **~1.9×** |
| iSSF, outer threads | **~3.5×** |
| Outer processes | generally ~1.1–1.5× |
| Inner native threading | generally ~1.1–1.25× |

Outer threading across independent fits provided the strongest application-level throughput on the tested node. The earlier saturation of SSF is consistent with concurrent fits repeatedly traversing large `stratum × choice × predictor` tensors and competing for shared cache and memory bandwidth. Importantly, isolated four-chain SSF fits scaled well across the cores required by their chains; the efficiency loss appeared primarily when many fits were executed concurrently. Single-fit latency and whole-node model throughput are therefore distinct optimization problems.

Dense raster prediction exhibits a third regime. Once model evaluation is fused, the kernel behaves as a streaming computation and strong scaling continues only while additional execution units can be supplied with independent blocks and useful memory bandwidth. The final capacity experiment confirmed that this architecture remains memory-bounded well beyond RAM capacity: approximately 900 GiB of logical input could be processed on a 224-GiB allocation without a swap- or major-fault-style collapse.

The initial matched workstation-to-CoolMUC-4 weak-scaling experiment nevertheless produced only approximately 33% efficiency for point sampling and 50% for raster prediction. This apparent discrepancy was informative: capacity was already excellent, so poor weak scaling had to originate elsewhere. Subsequent profiling showed substantial unused CPU and memory capacity and redirected the investigation toward task supply, source-graph structure, and execution geometry. After source-layout optimization, raster weak scaling moved toward a diagnostic estimate of approximately 80% relative to the earlier workstation baseline. Because this final percentage combines experiments conducted during the optimization campaign, it should be treated as a mechanism-level estimate until reproduced in a fully harmonized final weak-scaling rerun.

The broader result is that capacity and scaling measure different properties. An algorithm can process data several times larger than RAM while still exposing insufficient useful work to a wider node.

### 8. Execution geometry and architectural mechanism

A worker geometry is written as

$$
G=(W,T),
$$

where $W$ is the number of worker processes and $T$ the number of threads per worker. Under full use of a physical-core budget $P$,

$$
WT=P.
$$

```{thumbnail} assets/performance/04_benchmarking.png
:width: 100%
:alt: 
Equal physical-core budgets therefore do not imply equivalent execution environments. We explicitly benchmarked this trade-off for the spatial kernels and compared it with the placement of parallelism during statistical inference. Intermediate process/thread geometries occupied the favourable time–memory region for both spatial kernels, whereas inference benefited most from coarse-grained concurrency across independent fits.
```

Configurations such as `112×1`, `28×4`, `16×7`, and `8×14` therefore expose the same nominal 112-way concurrency on CoolMUC-4, but they differ substantially in process state, serialization, scheduler endpoints, memory replication, cache sharing, and transfer behaviour.

The final production geometry benchmark established the practical consequence: the low-process hybrid `8×14` configuration completed the 4.29-billion-cell raster workload in approximately 9.8–9.9 s, while `112×1` required about 16.2–16.4 s. Earlier mechanism experiments were then used to determine why.

First, we removed Dask and measured only the reusable numerical prediction kernel with threads pinned to physical CPUs. On the workstation the kernel reached an early scaling knee:

| Pinned threads | Wall time | Speed-up |
|---:|---:|---:|
| 1 | 1.452 s | 1.00× |
| 2 | 0.839 s | 1.73× |
| 3 | 0.634 s | 2.29× |
| 4 | **0.561 s** | **2.59×** |
| 6 | 0.775 s | 1.87× |
| 12 | 1.469 s | 0.99× |

Thread placement further supported a cache/locality mechanism: distributing four threads across independent L3 domains improved throughput by approximately 27% relative to compact placement, and the corresponding improvement with three threads was approximately 34%. A single very wide worker is therefore intrinsically unattractive on this architecture.

The same numerical kernel behaved very differently on CoolMUC-4:

| Pinned threads | Speed-up | Parallel efficiency |
|---:|---:|---:|
| 1 | 1.0× | 100% |
| 2 | 1.9× | 95% |
| 4 | 3.6× | 90% |
| 7 | 5.8× | 83% |
| 14 | 8.7× | 62% |
| 28 | 10.0× | 36% |
| 56 | 9.3× | 17% |
| 112 | 16.0× | 14% |

Although the non-monotonic shape should not be interpreted as proof of one specific bandwidth mechanism, the direct kernel clearly retains useful scaling far beyond workstation-scale thread widths. The poor `112×1` Dask result therefore cannot be explained simply by an inability of the numerical operation to exploit a wide CoolMUC-4 node.

An earlier instrumented diagnostic run then compared `8×14` and `112×1` under the same 112-core allocation:

| Diagnostic quantity | `8×14` | `112×1` | Change |
|---|---:|---:|---:|
| Physical cores | 112 | 112 | same |
| Median wall time | 10.37 s | 15.58 s | 1.50× |
| Aggregate worker RSS | ~4.3 GiB | ~36.5 GiB | ~8.5× |
| Involuntary worker context switches $s^{-1}$ | ~142 | ~1,702 | ~12× |
| Summed Dask transfer-action time* | ~6.8 s | ~25.3 s | ~3.7× |
| Dask compute-task parallelism* | ~79 | ~32 | ~40% retained |
| Equivalent busy worker cores | ~35 | ~36 | essentially unchanged |

\* Task-stream quantities summarize the repetitions for which task-stream events were captured.

The process-heavy configuration therefore multiplied runtime state and operating-system scheduling without generating more equivalent useful CPU work. The direct kernel remained capable of wide-node scaling, but exposing every physical core as an independent Python/Dask worker increased memory consumption, context switching, transfers, and coordination. This provides a mechanistic explanation for the low-process hybrid region observed in the final production benchmark.

Storage geometry proved equally important. Increasing only the *computational* block size on a 225.094-GiB raster raised useful CPU occupancy but had almost no effect on wall time because the underlying Zarr store still consisted of fine $1024^2$ physical chunks:

| Computational block | Wall time | Mean useful cores |
|---:|---:|---:|
| $1024^2$ | 25.759 s | 33.1 |
| $2048^2$ | 25.313 s | 45.0 |
| $4096^2$ | 26.821 s | 41.5 |

The identical scientific raster was therefore rewritten using $2048^2$ physical chunks:

| Physical Zarr layout | Source chunks | Paired wall times | Mean wall time | Mean useful cores |
|---|---:|---:|---:|---:|
| $1024^2$ | 9,604 | 25.564, 27.220 s | 26.392 s | 32.48 |
| $2048^2$ | 2,401 | 16.270, 16.318 s | **16.294 s** | **57.45** |

Reducing the number of source blocks by 75% lowered wall time by approximately 38% and increased useful occupancy by about 77%:

$$
\frac{26.392}{16.294}
\approx
1.62.
$$

Major page faults and local I/O wait remained negligible, supporting the interpretation that the improvement arose from a smaller source graph and reduced scheduler overhead rather than from additional memory capacity.

The execution problem can therefore be summarized conceptually as

$$
T_{\mathrm{total}}
=
T_{\mathrm{kernel}}(T)
+
T_{\mathrm{memory}}(W,T)
+
T_{\mathrm{runtime}}(W)
+
T_{\mathrm{transfer}}(W,C)
+
T_{\mathrm{I/O}}(C),
$$

where $C$ represents task and storage granularity. This is not intended as a fitted timing model, but as a useful decomposition of the mechanisms encountered during calibration. Neither one process for the entire machine nor one process per physical core is universally optimal.

### 9. End-to-end performance and deployment on new systems

The final question is whether kernel-level improvements survive integration into complete scientific analyses. We therefore compared the reference and optimized implementations using 50,000 steps per individual, a 384-GiB environmental workload, and two paired repeats for each RSF, SSF, and iSSF workflow.

| Workflow | Reference | Optimized | End-to-end speed-up |
|---|---:|---:|---:|
| RSF | ~2.6 min | ~2.0 min | **1.34×** |
| SSF | ~3.3 min | ~2.6 min | **1.27×** |
| iSSF | ~4.5 min | ~3.0 min | **1.50×** |

These gains are smaller than the largest isolated kernel improvements because complete analyses also contain stages that were not modified or were not dominant. They nevertheless demonstrate that storage-aware sampling, structure-aware inference, fused prediction, and calibrated execution geometry reduce complete analysis time rather than only improving synthetic microbenchmarks.

Performance optimization was also evaluated against scientific output rather than wall time alone. Optimized formulations were compared with their reference implementations using fitted model results, predictions, and diagnostic outputs; Bayesian workflows retained pointwise log-likelihoods required for posterior assessment. The broader workflow was additionally exercised across empirical movement datasets representing different spatial scales and movement structures. Performance changes were therefore accepted as implementation improvements only when the scientific calculation remained consistent with the reference formulation.

The CoolMUC-4 settings should not, however, be transferred directly to another cluster. The transferable result of the campaign is the calibration procedure. Computational demand in hrHSA is conditional on trajectory dimensions, environmental dimensions, and the requested operation. A million tracked locations may expand into many millions of RSF or SSF candidate rows, while complete-surface prediction depends much more directly on raster cells, predictor bands, transformations, and source chunking. A useful calibration workload must therefore be smaller than the intended production workload while preserving the properties that determine its execution behaviour.

A practical site calibration can be organized as follows:

| Calibration stage | Held fixed | Varied | Principal response |
|---|---|---|---|
| Memory-capacity test | allocated hardware | logical workload | RSS, faults, throughput |
| Kernel-width test | numerical operation | pinned threads | useful intraprocess width |
| Worker-geometry sweep | workload + physical cores | processes × threads | wall time, RSS |
| Physical-storage test | scientific input + geometry | source chunk layout | task count, occupancy |
| Computational-block test | source layout | kernel block size | time, RSS, task density |
| Strong scaling | workload | physical resources | speed-up, efficiency |
| Weak scaling | work per physical core | machine width | runtime stability |
| Inference-concurrency test | model workload | outer vs inner parallelism | throughput, latency |

Memory safety should be treated as a constraint before runtime minimization. A useful conceptual requirement is

$$
M_{\mathrm{predicted}}
<
\alpha M_{\mathrm{allocated}},
\qquad
0<\alpha<1,
$$

where the reserve protects scheduler state, numerical libraries, filesystem cache, temporary allocations, and uncertainty in the memory estimate. Similarly, optimization must use the resources actually granted by the scheduler rather than the installed capacity of the physical node,

$$
\mathcal H_{\mathrm{allocated}}
\neq
\mathcal H_{\mathrm{physical}}.
$$

A job receiving 56 cores on a 112-core node should be calibrated within 56 cores.

The goal of a geometry sweep should normally be to identify a broad high-performance region rather than an exact mathematical optimum. Runtime varies with cache state, operating-system activity, scheduler behaviour, and shared-system conditions. Candidate configurations that are practically indistinguishable from the fastest should therefore be compared by memory use and stability as well as median wall time. This is particularly relevant when neighbouring layouts differ by only a few percent.

The repository provides dedicated benchmark drivers for this calibration. `benchmarking/scripts/01_python/run_worker_geometry.py` evaluates fixed-core spatial worker geometries, while `benchmarking/scripts/01_python/run_scaling.py` performs reproducible scaling experiments on a prepared Zarr workload. A representative geometry test is

```bash
python benchmarking/scripts/01_python/run_worker_geometry.py \
    --root /path/to/benchmark-data \
    --output geometry-8x14.jsonl \
    --scheduler-file scheduler.json \
    --workers 8 \
    --threads-per-worker 14 \
    --benchmarks sampling,surface \
    --repeats 3 \
    --warmup-repeats 1
```

where `8×14` should be understood as one candidate in a sweep rather than a universal default.

For process-oriented execution, nested native thread pools should normally be constrained unless native threading is itself being benchmarked:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

The scaling driver additionally checks whether the task graph contains sufficient independent work for the requested worker count, avoiding apparently poor CPU scaling that is in fact caused by task starvation.

Once calibrated, a machine-specific profile can retain the tested processor and NUMA topology, software environment, worker geometries, physical and computational chunking, memory behaviour, preferred performance region, and benchmark date. Such a profile should be invalidated when relevant conditions change—for example after a major change in the hrHSA kernel implementation, Dask, BLAS backend, CPU architecture, or storage layout.

The final benchmark campaign therefore supports a broader conclusion than any individual timing. hrHSA performance is a property of the interaction between workload, algorithm, storage, runtime, and machine architecture. Sparse sampling is governed primarily by locality and scheduler granularity; statistical inference by likelihood representation and the placement of concurrency; and dense prediction by memory traffic, source layout, and process/thread geometry. Because these mechanisms can be measured independently, execution policies can therefore be calibrated reproducibly for a target HPC system rather than hidden behind universal parallel defaults.