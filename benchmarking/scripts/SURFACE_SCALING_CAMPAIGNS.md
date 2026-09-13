# Public surface-scaling campaigns

This runbook defines the first public workload-scaling campaign for hrHSA surface prediction. It deliberately separates three questions:

1. **Capacity scaling** — fixed machine and calibrated geometry, increasing raster size. This answers: *how large a problem can this machine handle, and how does the regime change as the raster approaches/exceeds RAM?*
2. **Strong scaling** — fixed raster, increasing resources. This answers: *how much faster does the same problem become when more cores/nodes are added?*
3. **Weak scaling** — raster size grows in proportion to physical resources. This answers: *can hrHSA absorb a proportionally larger problem while keeping runtime approximately constant?*

Performance profiling / geometry discovery remains a separate campaign. The commands below therefore take a **calibrated geometry** as an input rather than silently re-optimizing it at every problem size.

## Why nested raster families?

`prepare_surface_family.py` stores only the **largest** deterministic Zarr raster required by a campaign. Smaller workload points are storage-chunk-aligned nested windows of the same raster.

This has three advantages:

- hundreds of GiB/TiB are not duplicated on disk;
- every size point comes from the same deterministic raster realization;
- workload windows end on complete storage chunks, avoiding disproportionate tail-chunk effects.

For six `float32` bands and 1024×1024 storage chunks, common nominal targets resolve to:

| Requested logical input | Chunk-aligned logical input | Side length |
| ---: | ---: | ---: |
| 24 GiB | 24.000 GiB | 32,768 |
| 48 GiB | 47.461 GiB | 46,080 |
| 96 GiB | 96.000 GiB | 65,536 |
| 192 GiB | 194.086 GiB | 93,184 |
| 224 GiB | 225.094 GiB | 100,352 |
| 448 GiB | 446.344 GiB | 141,312 |
| 896 GiB | 900.375 GiB | 200,704 |

The benchmark records both requested and exact logical sizes; plots should use the exact value when precision matters.

## Workstation: prepare once

Choose a fast local SSD/NVMe root. The example below prepares the maximum workstation capacity raster (~194 GiB actual) once:

```bash
BASE=/fast/local/hrhsa-surface-scaling

python benchmarking/scripts/01_python/prepare_surface_family.py \
    --root "$BASE" \
    --raster-gib 24,48,96,192 \
    --bands 6 \
    --storage-chunk 1024 \
    --workers 8
```

Do not use `--force` unless the existing prepared family is intentionally being replaced.

## Workstation: capacity scaling

Use the calibrated full-machine production geometry. On the current 12-core reference workstation, `12x1` is a convenient candidate for the capacity pilot; substitute the final calibrated geometry if profiling selects another member of the high-performance region.

```bash
python benchmarking/scripts/01_python/run_surface_scaling.py \
    --root "$BASE" \
    --output "$BASE/results-capacity.jsonl" \
    --mode capacity \
    --backend local \
    --geometry 12x1 \
    --raster-gib 24,48,96,192 \
    --warmup-repeats 1 \
    --repeats 3
```

The geometry remains fixed while the workload grows. This is the curve to use for statements such as “what happens when a raster approaches or exceeds the RAM of a fixed workstation?”

## Workstation: strong scaling

Keep one scientific problem fixed while increasing the physical-core budget. The runner preserves the calibrated **threads per worker** and changes the worker count accordingly.

For a `12x1` base geometry:

```bash
python benchmarking/scripts/01_python/run_surface_scaling.py \
    --root "$BASE" \
    --output "$BASE/results-strong.jsonl" \
    --mode strong \
    --backend local \
    --geometry 12x1 \
    --fixed-raster-gib 24 \
    --core-counts 1,2,4,6,8,10,12 \
    --warmup-repeats 1 \
    --repeats 3
```

If the calibrated geometry uses two threads/worker (for example `6x2`), request only core counts divisible by two (for example `2,4,6,8,10,12`). The runner refuses a core count that would silently change worker thread width.

## Workstation: weak scaling

Hold logical raster input per physical core approximately constant:

```bash
python benchmarking/scripts/01_python/run_surface_scaling.py \
    --root "$BASE" \
    --output "$BASE/results-weak.jsonl" \
    --mode weak \
    --backend local \
    --geometry 12x1 \
    --core-counts 1,2,4,6,8,10,12 \
    --gib-per-core 2 \
    --warmup-repeats 1 \
    --repeats 3
```

At ideal weak scaling, wall time remains approximately constant while raster size and physical cores grow together. Because storage-chunk alignment slightly perturbs the exact requested GiB, use `metadata.logical_gib_per_physical_core` in the analysis rather than assuming the nominal density was exact.

## CoolMUC-4: prepare the shared family once

The CoolMUC family includes both the one-node capacity windows and the 1/2/4-node weak-scaling windows. Only the largest (~900 GiB logical) raster is physically stored.

Submit from the exact source commit that will later run the benchmarks:

```bash
SHA=$(git rev-parse HEAD)

sbatch \
  benchmarking/scripts/02_slurm/coolmuc4_prepare_surface_scaling.sbatch \
  "$SHA"
```

By default the family is stored under:

```text
$SCRATCH_DSS/hrhsa-surface-scaling
```

A different shared-data root can be supplied as the second sbatch argument.

## CoolMUC-4 `cm4_inter`: one-node capacity scaling

After preparation finishes:

```bash
SHA=$(git rev-parse HEAD)

sbatch \
  benchmarking/scripts/02_slurm/coolmuc4_inter_capacity_scaling.sbatch \
  "$SHA"
```

The default diagnostic campaign uses the current candidate `8x14` geometry and nested 24/48/96/192-GiB windows on one node. Geometry and raster targets are optional third/fourth sbatch arguments.

## CoolMUC-4 `cm4_inter`: strong and weak scaling

The four-node job runs both curves inside the same allocation/time window:

```bash
SHA=$(git rev-parse HEAD)

sbatch \
  benchmarking/scripts/02_slurm/coolmuc4_inter_scaling.sbatch \
  "$SHA"
```

Defaults:

```text
per-node geometry        8x14
strong-scaling workload  224 GiB nominal (~225.1 GiB exact)
node counts              1, 2, 4
weak-scaling density     2 GiB / physical core
weak workloads           ~225.1, 446.3, 900.4 GiB
```

`run_surface_scaling.py` deliberately accepts a node subset smaller than the enclosing allocation through `slurm_allocation_client()`. This lets 1-, 2-, and 4-node observations share one four-node allocation and substantially reduces temporal/environmental confounding.

`cm4_inter` remains **diagnostic**. Do not pool these timings statistically with later `cm4_std` production measurements.

## What every record contains

In addition to the normal benchmark wall-time/throughput/memory fields, the new campaign adds:

```text
scaling_mode
backend
resource_kind
resource_count
physical_cores_used
base_geometry
executed_geometry
target_raster_gib
actual_raster_gib
raster_logical_bytes
raster_cells
raster_size
raster_size_error_pct
logical_gib_per_physical_core
memory_gib_available_to_campaign
raster_to_memory_ratio
```

Slurm observations additionally retain the execution plan, strict topology-validation result, allocation node count, selected node subset, partition, job ID, and node list.

## Interpretation rules

- Capacity scaling changes **problem size only**; geometry/resources remain fixed.
- Strong scaling changes **resources only**; the scientific raster remains fixed.
- Weak scaling changes **problem size and resources together** while preserving the requested workload density.
- Geometry is an input to these campaigns. Do not re-optimize geometry independently at every size point in the primary curves.
- Warm-ups are not publication observations.
- Requested GiB and exact chunk-aligned GiB are both provenance; use exact GiB for quantitative scaling calculations.
- Workstation and CoolMUC curves should be presented as separate hardware regimes unless an explicit common-workload bridge point is being compared.

Point-count and Bayesian/MCMC capacity adapters will follow the same separation of capacity, profiling, and scaling, but should not be forced through the surface-raster abstraction.
