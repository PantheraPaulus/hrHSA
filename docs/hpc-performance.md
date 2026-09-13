# HPC execution and performance engineering

High-resolution habitat-selection analysis is not dominated by a single computational operation. A complete workflow alternates between sparse access to large environmental datasets, repeated statistical inference, and dense evaluation over complete spatial surfaces. These operations stress different parts of a computing system: sparse sampling is sensitive to storage locality and task granularity, inference to numerical representation and shared-memory contention, and raster prediction to memory traffic, temporary allocation, and source layout. Parallel performance therefore cannot be inferred from core count alone.

hrHSA treats high-performance execution as part of the algorithm rather than as a scheduler layer added to otherwise unchanged code. The performance-engineering strategy was to first identify the dominant kernels in complete analyses, simplify their numerical and data-access structure, and only then determine how the remaining work should be mapped onto available hardware. Reference implementations were retained throughout as correctness targets, allowing changes in execution strategy to be separated from changes in the scientific calculation.

The resulting benchmark campaign combined kernel profiling, memory-capacity experiments, fixed-core process/thread sweeps, strong and weak scaling, storage-layout experiments, resource telemetry, and complete end-to-end RSF, SSF, and iSSF analyses. The purpose was not simply to obtain minimum wall times, but to identify the mechanisms that limit each kernel and to provide a calibration procedure that can be repeated on other HPC architectures.

## Part I — Computational model and benchmark design

### 1. Kernel-specific performance architecture

Profiling showed that the computational cost of hrHSA is concentrated in three kernels with fundamentally different access patterns.

**Sparse raster sampling** extracts environmental values at observed and available locations. Only a small fraction of the environmental grid is requested, but those requests may be distributed across raster stores containing billions of cells. Performance is therefore governed by the relationship between points, physical storage chunks, scheduler tasks, and the amount of indexing information moved between processes.

**Statistical inference** repeatedly evaluates structured likelihoods, gradients, or posterior densities. The environmental data have already been reduced to model-ready arrays, so performance depends primarily on the representation of the likelihood, the size of the working tensors, and where concurrency is exposed. Additional threads within a fit and concurrent independent fits are not equivalent forms of parallelism.

**Dense raster prediction** reverses the point-sampling access pattern. Essentially every raster cell is evaluated, turning prediction into a streaming numerical calculation in which repeated traversals of large predictor arrays and temporary allocations can dominate floating-point arithmetic.

The optimization procedure therefore followed the sequence

```text
complete-workflow profiling
        ↓
identify dominant kernel
        ↓
simplify numerical / data representation
        ↓
reduce unnecessary allocation and movement
        ↓
establish bounded-memory behaviour
        ↓
calibrate process / thread / chunk geometry
        ↓
measure scaling
        ↓
validate in complete scientific workflows
```

This ordering is deliberate. Parallelizing an inefficient representation generally distributes its overhead rather than removing it.

A second distinction is made between the **scientific model** and the **execution model**. Worker count, thread count, chunk dimensions, scheduler backend, and temporary storage determine how a calculation is executed but do not belong to the ecological model itself. The same RSF, SSF, or iSSF specification can therefore be evaluated through reference, local, or distributed execution without changing its inferential definition.

### 2. Benchmark systems and workloads

Most large-scale experiments were performed on a single LRZ CoolMUC-4 node exposing 112 physical cores and 224 logical CPUs. Large spatial experiments generally used a 224-GiB memory allocation. A 12-core AMD Ryzen 9 3900X workstation with approximately 125 GiB RAM provided a smaller shared-memory reference for latency experiments, numerical thread scaling, and matched weak-scaling comparisons.

The two systems were used to characterize different execution regimes rather than to infer an absolute workstation-versus-HPC speed-up from unmatched workloads. The workstation reaches its useful intraprocess scaling limit over relatively few cores, whereas a CoolMUC-4 node exposes a substantially wider shared-memory architecture.

| System | Physical cores | Logical CPUs | Memory used in campaign | Main role |
|---|---:|---:|---:|---|
| Reference workstation | 12 | 24 | ~125 GiB installed | latency, thread-width and weak-scaling reference |
| CoolMUC-4 node | 112 | 224 | typically 224 GiB allocated | capacity, geometry and scaling |
| Calibrated spatial geometry | 112 total | — | workload dependent | 8 workers × 14 threads |

The campaign used representative rather than artificially equalized model workloads. The common workflow-profiling experiment contained 20 individuals and approximately 220,000 candidate rows per workflow against a nominal 24-GiB environmental stack. RSF used three predictors, SSF six, and iSSF fourteen. Frequentist models were fitted with L-BFGS. Bayesian benchmarks used four chains with 500 warm-up and 500 retained posterior draws per chain, with the sampling backend selected according to model class.

The principal benchmark campaigns were:

| Campaign | Workload | Comparison | Main response |
|---|---|---|---|
| Workflow profiling | 20 individuals; ~220k candidates; 24-GiB raster | RSF, SSF, iSSF | stage runtime |
| End-to-end A/B | 50k steps/individual; 384-GiB environment | reference vs optimized | total wall time |
| Point capacity | \(10^6\)–\(10^9\) queries; 24–768-GiB rasters | increasing workload | throughput, RSS |
| Raster A/B | 128–384-GiB logical rasters | reference vs fused prediction | runtime, RSS |
| Raster capacity | up to 900.4-GiB six-band raster | fixed 224-GiB node allocation | throughput, RSS |
| Worker geometry | fixed 112 physical cores | process × thread sweep | runtime, memory |
| Inference parallelism | 20 independent individual fits | 1–20 physical cores | repeat-paired speed-up |
| Weak scaling | 12 → 112 cores | workload increased 9.33× | scaling efficiency |
| Storage layout | identical 225.094-GiB raster | physical \(1024^2\) vs \(2048^2\) chunks | runtime, task supply |

The weak-scaling experiment preserved approximately constant work per physical core. The workstation processed 96 million point queries against a nominal 24-GiB raster, whereas CoolMUC-4 processed 896 million queries against a 225.094-GiB raster:

\[
\frac{112}{12}
=
\frac{896}{96}
=
9.33.
\]

This allows machine width and workload to increase together while approximately preserving the per-core scientific problem.

### 3. Measurement, correctness, and provenance

Benchmark timing was applied to the scientific operation under study rather than to unrelated data preparation. Distributed raster benchmarks forced completion by persisting the partitioned result but did not subsequently gather hundreds of gigabytes of output into a single local array. The reported prediction times therefore describe distributed computation and materialization rather than an artificial global collection step.

Reference and accelerated implementations were evaluated under a correctness gate before their runtime was interpreted. Raster values were compared directly on deterministic workloads. Frequentist model changes were checked through likelihoods, coefficient estimates, predictions, and validation outputs. Bayesian acceleration was evaluated through posterior agreement and predictive diagnostics as well as runtime. A faster sampler is not considered an optimization if it changes the inferred posterior.

For Bayesian workloads, useful performance measures therefore include

```text
wall time
bulk ESS / second
tail ESS / second
R-hat
divergences
NUTS depth / steps
posterior agreement
held-out predictive agreement
```

rather than raw draws per second alone.

The benchmark infrastructure stores run-level machine-readable records rather than only final plots. Depending on the campaign, these contain workload dimensions, physical-core allocation, workers, threads per worker, physical and computational chunks, task counts, materialization policy, wall time, throughput, memory measurements, platform information, software revision, and repeat identity. Instrumented experiments additionally recorded process-tree RSS, CPU time, useful-core equivalents, read/write activity, process count, thread count, and node-memory state at sub-second intervals.

Discovery, diagnostic, and final production experiments are kept conceptually distinct. Mechanism experiments may use additional instrumentation or slightly different workloads because their purpose is causal diagnosis; headline performance values are taken from the corresponding production benchmark wherever possible.

## Part II — Kernel optimization and final performance

### 4. Sparse raster sampling

Environmental extraction is a sparse gather problem. A naïve nearest-neighbour implementation evaluates point locations largely independently, so multiple requests touching the same raster block can repeatedly traverse the same underlying data. At large scale, raster access, indexing payloads, and scheduler coordination become more important than the coordinate lookup itself.

hrHSA first exploits regular raster geometry directly. For point \(i\),

\[
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
\]

allowing raster rows and columns to be obtained arithmetically rather than through repeated labelled nearest-neighbour searches. Irregular grids retain the general search-based fallback.

The requested cells are subsequently mapped to raster blocks, and all points touching the same block are evaluated together:

```text
point coordinates
        ↓
integer raster row / column
        ↓
physical / computational block
        ↓
group requests by block
        ↓
read block once
        ↓
local NumPy gather
        ↓
restore original point order
```

Within a block, local row and column positions are flattened,

\[
q_i
=
r_i^{\mathrm{local}}B_x
+
c_i^{\mathrm{local}},
\]

and normally represented as `uint32`. This reduced index storage from approximately 16 to 4 bytes per requested location while also reducing serialization and communication overhead.

Large point collections are not submitted as one global materialized request. They are processed through bounded batches, with only a limited amount of point and raster work allowed in flight at one time. This permits routing, raster access, and result assembly to overlap while keeping the active memory footprint bounded.

These changes progressively reduced point-sampling cost. Arithmetic indexing alone reduced a 100-million-point experiment from approximately 66.8 to 49.5 s. The final block-local implementation sampled 100 million points against a nominal 24-GiB raster stack in approximately 20 s, corresponding to about 5 million queries s\(^{-1}\).

At larger scale, the scheduler itself became a limiting resource. hrHSA therefore decomposes very large spatial point workloads into independent compact domains using recursive weighted bisection. Domain construction balances point counts while keeping the associated raster footprint spatially compact, reducing the amount of work routed through one global task graph.

For a one-billion-point workload against approximately 194 GiB of raster data, one global execution domain required roughly 61–64 s. Decomposition into 28 balanced four-core domains reduced wall time to **22.5 s**, increasing throughput from approximately 16 to **44.5 million point queries s\(^{-1}\)**. Increasing decomposition further to 56 domains slightly increased wall time, indicating that the benefit from additional concurrency had been exceeded by per-domain coordination costs.

The same bounded architecture subsequently processed one billion point queries against a nominal **768-GiB raster stack** in approximately 56 s. Logical raster size therefore does not need to fit in node memory, provided that the simultaneously active blocks and routing structures remain bounded.

### 5. Statistical inference

Inference was optimized by exploiting statistical structure before introducing additional concurrency.

For SSF and iSSF models, every stratum contains \(J\) alternatives and exactly one observed choice. Rather than repeatedly grouping a long tabular representation, hrHSA stores the design as a dense

```text
stratum × choice × predictor
```

tensor together with one integer identifying the observed alternative in every stratum.

For stratum \(s\),

\[
P(Y_s=j)
=
\frac{\exp(\eta_{sj})}
{\sum_{k=1}^{J}\exp(\eta_{sk})},
\qquad
\eta_{sj}=x_{sj}^{\top}\beta,
\]

and the contribution to the log-likelihood reduces to

\[
\ell_s
=
\eta_{s,y_s}
-
\operatorname{logsumexp}_{j}
(\eta_{sj}).
\]

Likelihoods, probabilities, gradients, and information matrices can therefore be evaluated through vectorized array operations without repeated grouping or generic categorical bookkeeping.

Bayesian SSF and iSSF models use the same representation with non-centred hierarchical coefficients,

\[
\beta_{ip}
=
\mu_p+\sigma_p z_{ip},
\qquad
z_{ip}\sim\mathcal N(0,1).
\]

Large design tensors are retained in `float32` where appropriate to reduce concurrent working-set size, posterior predictions are evaluated in bounded batches, and unnecessary observation-level posterior arrays are not retained merely because they were required transiently during likelihood evaluation.

The specialized one-of-\(J\) formulation reduced full four-chain Bayesian SSF runtime by approximately **15–16%** relative to a general categorical formulation while retaining stratum-level log-likelihoods required for posterior diagnostics and PSIS-LOO.

RSFs benefit from a complementary structural reduction. Observations sharing predictor vector \(x_g\) share the same Bernoulli probability and can therefore be aggregated exactly as

\[
y_g
\sim
\operatorname{Binomial}(n_g,p_g),
\qquad
\operatorname{logit}(p_g)=x_g^\top\beta.
\]

Large collections of repeated observation-level likelihood terms can consequently be replaced by a much smaller number of unique predictor combinations. Optional discretization of continuous covariates can provide additional controlled compression when appropriate.

Parallelism was then evaluated above these optimized likelihoods. Twenty independent per-individual fits were benchmarked using physical-core budgets of

\[
1,\;2,\;4,\;6,\;8,\;10,\;12,\;16,\;20.
\]

Three strategies were distinguished:

| Strategy | Parallel unit |
|---|---|
| Outer processes | independent model fits |
| Outer threads | independent model fits |
| Inner native threads | numerical work within one fit |

Each measurement was normalized against the serial runtime from the same repeat,

\[
S_{p,r}
=
\frac{T_{1,r}}
{T_{p,r}},
\]

before aggregation.

The best observed outer-thread speed-ups were approximately **3.4× for RSF**, **1.9× for SSF**, and **3.5× for iSSF**. Outer process parallelism generally produced only ~1.1–1.5×, while additional native threading within a fit typically provided ~1.1–1.25× at the application level.

The earlier saturation of SSF is consistent with several simultaneous fits repeatedly traversing large dense choice tensors and competing for shared cache and memory bandwidth. Importantly, isolated four-chain Bayesian SSF fits themselves scaled effectively across the small number of cores required by their chains. Single-fit latency and whole-machine throughput are therefore distinct optimization problems.

### 6. Dense raster prediction

Raster prediction evaluates essentially every cell and is consequently dominated by data traversal and temporary allocation rather than irregular lookup. In the readable reference implementation, standardization, nonlinear terms, interactions, and model evaluation are expressed as separate labelled Xarray operations. This representation is transparent, but can require several complete passes through the environmental stack and generate multiple raster-sized intermediates.

The accelerated implementation evaluates the complete fitted model within spatial blocks:

```text
read selected predictors
        ↓
standardize where required
        ↓
linear terms
        ↓
quadratic terms
        ↓
interactions
        ↓
categorical masks
        ↓
linear predictor
        ↓
relative intensity / choice value
        ↓
output block
```

Transformations are simplified before execution whenever possible. For a standardized linear term,

\[
\beta
\frac{x-\mu}{\sigma}
=
\frac{\beta}{\sigma}x
-
\frac{\beta\mu}{\sigma},
\]

allowing the standardization constants to be folded into the fitted coefficients instead of constructing a complete standardized raster. Separate standardized buffers are retained only when predictors participate in quadratic or interaction terms. Arithmetic is performed in place where practical, and temporary workspaces are reused between block evaluations.

The resulting fused kernel consistently outperformed the reference formulation across increasingly large logical rasters:

| Logical raster size | Optimized speed-up |
|---:|---:|
| 128 GiB | 1.73× |
| 160 GiB | 1.61× |
| 192 GiB | 1.64× |
| 224 GiB | 1.64× |
| 256 GiB | 1.69× |
| 320 GiB | 1.68× |
| 384 GiB | 1.67× |

For the **384-GiB** workload, wall time decreased from approximately **73.1 s to 43.7 s**. The near-constant speed-up across the tested range indicates that the improvement reflects reduced sustained memory traffic and temporary allocation rather than an effect confined to small cache-resident workloads.

Worker topology was subsequently optimized at fixed physical-core count. On a **4.29-billion-cell** reference workload, an `8×14` configuration—eight worker processes with fourteen threads each—completed in approximately **9.8–9.9 s**, whereas `112×1` required approximately **16.2–16.4 s**. The same 112 physical cores therefore completed the calculation around 40% faster solely through a different process/thread geometry.

The final capacity experiment processed a **900.4-GiB six-band logical raster** on a node allocation of **224 GiB**. Aggregate worker RSS remained approximately **121–122 GiB**, and the largest completed runs sustained roughly **333–344 million cells s\(^{-1}\)**.

Raster capacity is therefore controlled principally by the active working set,

\[
M_{\mathrm{working}}
\ll
M_{\mathrm{logical}}
\]

when execution remains blockwise. Logical raster size can substantially exceed physical memory provided that intermediate blocks and final output are not globally materialized.

## Part III — Scaling and deployment

### 7. Capacity, strong scaling, and weak scaling

Capacity and scaling answer different questions. A workload may remain safely out of core while still scaling poorly across a wider node, and a numerically scalable kernel may fail to exploit a distributed runtime if its task graph is too fine.

For a fixed workload \(W\), strong scaling is described by

\[
S_{\mathrm{strong}}(p)
=
\frac{T_1(W)}
{T_p(W)},
\]

with efficiency

\[
E_{\mathrm{strong}}(p)
=
\frac{S_{\mathrm{strong}}(p)}
{p}.
\]

Weak scaling instead increases the workload in proportion to available resources,

\[
E_{\mathrm{weak}}(p)
=
\frac{T_1(W_1)}
{T_p(W_p)},
\qquad
W_p=pW_1.
\]

The early matched workstation-to-CoolMUC-4 weak-scaling experiment produced approximately **33% efficiency for point sampling** and **50% for raster prediction**. These values did not indicate a memory-capacity failure: the same architecture was already capable of operating on logical environmental datasets several times larger than physical RAM.

Resource telemetry instead showed substantial unused CPU and memory capacity. An early 896-million-point sampling workload generated approximately 55,000 scheduler-visible tasks while keeping only about 19 of 112 physical cores usefully occupied. A representative task decomposition included approximately 9,900 source-opening tasks, 896 point partitions, and more than 30,000 block-selection and extraction tasks. The problem was therefore insufficiently coarse useful work relative to scheduler coordination.

This diagnosis motivated both spatial domain decomposition for point sampling and the storage-layout experiments for raster prediction.

Following physical source-layout optimization, the raster weak-scaling result moved toward a diagnostic estimate of approximately **80%** relative to the earlier matched workstation baseline. Because this percentage combines experiments performed during successive stages of optimization, it should be interpreted as a mechanism-level result unless reproduced by a final fully harmonized weak-scaling campaign.

The broader conclusion is that memory capacity, useful concurrency, and scaling efficiency must be measured separately.

### 8. Execution geometry and bottleneck diagnosis

At a fixed physical-core budget \(C\), a worker geometry can be written as

\[
P\times T,
\qquad
PT=C,
\]

where \(P\) is the number of worker processes and \(T\) the number of threads per worker.

The following configurations all nominally expose 112 worker threads:

| Geometry | Processes | Threads / process |
|---|---:|---:|
| `112×1` | 112 | 1 |
| `56×2` | 56 | 2 |
| `28×4` | 28 | 4 |
| `16×7` | 16 | 7 |
| `14×8` | 14 | 8 |
| `8×14` | 8 | 14 |
| `4×28` | 4 | 28 |
| `2×56` | 2 | 56 |

They are nevertheless not equivalent execution environments. Process count changes memory replication, scheduler endpoints, serialization, and operating-system scheduling, while thread width changes cache sharing and memory-bandwidth contention.

The final production raster benchmark identified `8×14` as a favourable CoolMUC-4 geometry, but earlier diagnostic experiments were retained to explain the mechanism.

#### Direct numerical-kernel scaling

Dask was first removed entirely and the reusable numerical raster kernel was evaluated with pinned physical threads.

On the reference workstation, useful scaling ended rapidly:

| Threads | Wall time | Speed-up |
|---:|---:|---:|
| 1 | 1.452 s | 1.00× |
| 2 | 0.839 s | 1.73× |
| 3 | 0.634 s | 2.29× |
| 4 | **0.561 s** | **2.59×** |
| 6 | 0.775 s | 1.87× |
| 12 | 1.469 s | 0.99× |

Distributing three or four threads across separate cache domains further improved throughput relative to compact placement, supporting an interpretation based on cache locality.

The same numerical kernel scaled substantially farther on CoolMUC-4:

| Threads | Speed-up | Efficiency |
|---:|---:|---:|
| 1 | 1.0× | 100% |
| 2 | 1.9× | 95% |
| 4 | 3.6× | 90% |
| 7 | 5.8× | 83% |
| 14 | 8.7× | 62% |
| 28 | 10.0× | 36% |
| 56 | 9.3× | 17% |
| 112 | 16.0× | 14% |

The exact non-monotonic shape should not be interpreted as evidence for one unique bandwidth mechanism, but it establishes that the numerical kernel itself can exploit substantially more than one core per process on CoolMUC-4.

#### Process fragmentation

A separate instrumented mechanism experiment held the 112-core allocation fixed and compared the low-process `8×14` geometry with `112×1`.

| Diagnostic quantity | `8×14` | `112×1` |
|---|---:|---:|
| Median wall time | 10.37 s | 15.58 s |
| Aggregate worker RSS | ~4.3 GiB | ~36.5 GiB |
| Involuntary worker context switches s\(^{-1}\) | ~142 | ~1,702 |
| Summed transfer-action time | ~6.8 s | ~25.3 s |
| Task compute parallelism | ~79 | ~32 |
| Equivalent busy worker cores | ~35 | ~36 |

The process-heavy configuration consumed substantially more memory and generated much more operating-system and runtime coordination while producing essentially no increase in equivalent useful CPU occupancy. Together with the direct-kernel experiment, this associates the degradation at `112×1` with process/runtime fragmentation rather than inability of the numerical kernel to exploit the node.

These diagnostic timings are not used as substitutes for the final production geometry benchmark; they provide the mechanism underlying it.

#### Physical versus computational chunking

Worker geometry is only one component of execution geometry. Physical raster chunks determine how many source objects enter the execution graph, whereas computational chunks determine the amount of data evaluated by one kernel task.

This distinction was isolated on an identical **225.094-GiB** raster.

Changing only the computational block size while retaining \(1024^2\) physical Zarr chunks produced:

| Computational block | Wall time | Mean useful cores |
|---:|---:|---:|
| \(1024^2\) | 25.759 s | 33.1 |
| \(2048^2\) | 25.313 s | 45.0 |
| \(4096^2\) | 26.821 s | 41.5 |

The larger compute block increased CPU occupancy but produced almost no wall-time improvement because the underlying source graph remained unchanged.

The same scientific raster was then rewritten with a different **physical** layout:

| Physical layout | Source chunks | Mean wall time | Mean useful cores |
|---:|---:|---:|---:|
| \(1024^2\) | 9,604 | 26.392 s | 32.48 |
| \(2048^2\) | 2,401 | **16.294 s** | **57.45** |

Reducing the number of source chunks by 75% lowered wall time by approximately 38% and increased useful occupancy by roughly 77%:

\[
\frac{26.392}{16.294}
\approx
1.62.
\]

Major page faults and local I/O wait remained negligible. The gain therefore resulted principally from reducing source-task and scheduler overhead rather than from increasing physical memory.

The execution hierarchy can be summarized as

```text
physical storage chunk
        ↓
source objects
        ↓
source task graph
        ↓
scheduler / storage overhead

computational chunk
        ↓
kernel working set
        ↓
task granularity / memory locality
```

Both dimensions require calibration.

### 9. End-to-end validation and site calibration

Kernel improvements were finally evaluated within complete scientific workflows.

The reference and optimized implementations were compared using **50,000 steps per individual** and a **384-GiB environmental workload**, with two paired repeats for each model class.

| Workflow | Reference | Optimized | Speed-up |
|---|---:|---:|---:|
| RSF | ~2.6 min | ~2.0 min | **1.34×** |
| SSF | ~3.3 min | ~2.6 min | **1.27×** |
| iSSF | ~4.5 min | ~3.0 min | **1.50×** |

The smaller end-to-end gains relative to some isolated kernel optimizations are expected because complete analyses also contain operations that were not modified or were not computationally dominant. The paired comparison nevertheless confirms that improvements to sampling, likelihood evaluation, prediction, and execution geometry translate into shorter complete analyses rather than only faster microbenchmarks.

Scientific equivalence was checked alongside runtime. Optimized implementations were compared against their reference formulations through fitted coefficients, likelihoods, prediction outputs, and diagnostic quantities; Bayesian workflows additionally retained the pointwise log-likelihood information needed for posterior diagnostics and model assessment.

#### Calibrating another HPC system

The CoolMUC-4 geometry should not be copied directly to another architecture. What transfers is the calibration procedure.

A representative calibration should preserve the numerical and storage characteristics of the production problem while remaining sufficiently small for repeated experiments. The following sequence is generally sufficient:

| Stage | Hold fixed | Vary | Measure |
|---|---|---|---|
| Memory capacity | allocated hardware | logical workload | RSS, faults, throughput |
| Kernel-width test | numerical kernel | pinned threads | intraprocess scaling |
| Worker geometry | workload + physical cores | processes × threads | wall time, RSS |
| Physical layout | workload + workers | storage chunks | task count, occupancy |
| Compute blocks | storage layout | computational chunks | time, RSS, task density |
| Strong scaling | workload | compute resources | speed-up, efficiency |
| Weak scaling | work per core | machine width | runtime stability |
| Inference concurrency | statistical workload | outer vs inner parallelism | throughput, latency |

The benchmark should use the **allocated** machine rather than the installed machine. A job receiving 56 physical cores on a 112-core node should be optimized within those 56 cores.

Memory safety should also be established before minimizing wall time. Conceptually,

\[
M_{\mathrm{predicted}}
<
\alpha M_{\mathrm{allocated}},
\qquad
0<\alpha<1,
\]

where the reserve allows for scheduler state, temporary arrays, filesystem cache, native libraries, and uncertainty in the memory estimate.

The preferred configuration should normally be selected from a broad high-performance region rather than from one apparently exact minimum. Small timing differences between neighbouring geometries may be less important than substantially lower memory use or more stable execution.

#### Execution configuration

The scientific analysis remains independent of scheduler settings. Execution can be configured separately:

```python
from hsa.compute import ExecutionConfig

compute = ExecutionConfig(
    backend="local",
    n_workers=8,
    threads_per_worker=14,
    chunk_mb=256,
    point_batch_mb=128,
)
```

Supported execution modes include serial reference execution, local Dask clusters, caller-managed distributed clients, and SLURM-backed clusters. The values above illustrate a configuration and should not be interpreted as a universal default.

For process-oriented runs, hidden native thread pools should normally be constrained unless native threading is itself part of the experiment:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

This prevents a process-level geometry from being silently multiplied by nested BLAS or OpenMP threads.

The repository provides dedicated benchmark utilities for fixed-core worker-geometry and scaling experiments. Raw JSONL benchmark records should be retained together with the scripts used to aggregate them so that future hrHSA releases or machine profiles can be compared against the same experimental definition.

A reproducible site profile should record at least the processor and NUMA topology, physical-core and memory allocation, software environment, process/thread geometry, native thread policy, physical storage chunks, computational chunks, workload definition, benchmark date, and observed high-performance region.

## Summary

The performance-engineering campaign demonstrates that hrHSA is not limited by one universal computational resource. Sparse raster sampling is governed by locality, compact indexing, task granularity, and scheduler topology. Statistical inference benefits primarily from representations that exploit the known structure of the likelihood and from placing concurrency at the level where independent scientific work exists. Dense raster prediction behaves as a streaming calculation whose performance depends on memory traffic, process/thread geometry, and physical source layout.

On CoolMUC-4, the optimized sampling architecture processed **one billion point queries in 22.5 s**, corresponding to approximately **44.5 million queries s\(^{-1}\)**. The optimized raster predictor reduced a 384-GiB workload from approximately **73.1 to 43.7 s**, and processed a **900.4-GiB logical raster** within a 224-GiB node allocation while aggregate worker RSS remained near **121–122 GiB**. Rewriting an identical 225.094-GiB raster from \(1024^2\) to \(2048^2\) physical chunks reduced its source graph by 75% and prediction time by approximately **1.62×**. Application-level inference scaling reached approximately **3.4× for RSF**, **1.9× for SSF**, and **3.5× for iSSF** when concurrency was placed across independent fits. Complete optimized workflows were approximately **1.27–1.50× faster** than their reference implementations.

These results do not define one universal worker count, thread count, or chunk size. They show instead that the dominant kernels can be isolated sufficiently well to identify their limiting resources and map them deliberately onto a target architecture. High-performance execution in hrHSA is therefore treated as a reproducible calibration problem rather than as a fixed parallel default.