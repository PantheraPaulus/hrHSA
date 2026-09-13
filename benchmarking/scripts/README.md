# Benchmark tooling

The benchmark tree is organized by file role rather than by a mixture of machine, language, and campaign names. The numbered directories keep the execution layers in a stable, readable order while leaving script filenames descriptive and import-friendly.

```text
benchmarks/
├── 01_python/      # benchmark drivers, validators, probes and plotting helpers
├── 02_slurm/       # LRZ/CoolMUC-4 sbatch job definitions
├── 03_sh/          # shell bootstrap and submission helpers
├── 04_config/      # example machine/site configuration
├── 05_notebooks/   # benchmark reports and exploratory summaries
├── COOLMUC4_TESTING.md
└── README.md
```

For a first run of the topology-aware execution path, follow **`COOLMUC4_TESTING.md`** in order. It covers login-node preparation, policy tests, strict CPU-affinity validation, one-node and two-node benchmarks, run manifests, accounting, and failure triage.

The current CoolMUC policy profile is based on the LRZ CoolMUC Cheat Sheet released 2026-03-31 (best-before 2026-09-30). Site-policy constants are deliberately separated from hardware facts and benchmark evidence so they can be reviewed independently when LRZ guidance changes.

## 01_python

Reusable benchmark logic lives here. Important entry points include:

- `prepare_data.py` — prepare deterministic Zarr/Parquet benchmark inputs;
- `run_scaling.py` — raster strong-scaling driver;
- `run_worker_geometry.py` — fixed-core process/thread geometry driver;
- `run_planned_execution.py` — public planner/allocation path with strict topology validation and run manifests;
- `probe_coolmuc4.py` — runtime CPU/NUMA/network/profile probe;
- `run_workstation.py` and `run_production_workstation_suite.py` — workstation campaigns;
- `benchmark_inference.py` — frequentist and Bayesian inference benchmarks;
- `benchmark_cv_scaling.py` — outer-fold validation scaling;
- `run_production_smoke.py` — end-to-end production smoke gate;
- `plot_results.py`, `summarize_cv_scaling.py`, and validation helpers — machine-readable reporting and checks.

## 02_slurm

These are concrete scheduler job definitions. Files prefixed with `coolmuc4_` encode LRZ partition/QoS policy, CPU binding, memory requests, and data-root conventions. They source the shared bootstrap from `../03_sh/` and invoke Python drivers from `../01_python/` through repository-relative paths.

The raster campaign distinguishes exploratory, historical-control, and planner-driven jobs:

- `coolmuc4_large_worker_geometry.sbatch` — exploratory large-raster worker-geometry sweep;
- `coolmuc4_large_geometry_confirm.sbatch` — confirmatory geometry benchmark with an untimed warm-up, repeated timed measurements, and strict CPU-affinity validation before timing;
- `coolmuc4_multinode_surface.sbatch` — historical pure-process multi-node control;
- `coolmuc4_planned_surface.sbatch` — one-node regression benchmark through `coolmuc4_plan()` and `slurm_allocation_client()`;
- `coolmuc4_planned_multinode.sbatch` — two-node hybrid-worker benchmark through the same public execution path;
- `coolmuc4_inference.sbatch` — frequentist BLAS-scaling and hierarchical Bayesian backend campaign;
- `coolmuc4_cv_scaling.sbatch` — cross-validation scaling campaign.

The current large-raster fixed-112-core profile ranks `8x14` first by median wall time, with `16x7` close behind. This is a dated **provisional** benchmark profile, not a universal property of CoolMUC-4; confirm it when workload shape, storage layout, Dask/runtime versions, or hardware characteristics change.

Planner-driven jobs make worker placement explicit with one Slurm task per Dask worker, an explicit worker count per node, `--cpus-per-task=<threads>`, `block:block` distribution and core binding. The benchmark path then observes every worker's cpuset and fails strict validation if the physical layout differs from the requested geometry.

## 03_sh

`coolmuc4_common.sh` owns shared LRZ environment bootstrap and benchmark-root conventions. `submit_*.sh` helpers create/submit concrete scheduler jobs while retaining source provenance and avoiding submission-time resource overrides.

For interactive research workflows the same principle is available from Python: `hsa.compute.submit_slurm_job()` calls `sbatch` but intentionally does not accept resource overrides. This makes notebook submission convenient while keeping CPU, memory, partition and wall-time requests reviewable in the checked-in `.sbatch` file. Long-running waits use conservative scheduler polling defaults; manual status snapshots are preferable during interactive work.

Typical shell commands are:

```bash
bash benchmarking/scripts/03_sh/submit_coolmuc4_tiny_scaling.sh
bash benchmarking/scripts/03_sh/submit_coolmuc4_large_geometry_confirm.sh
bash benchmarking/scripts/03_sh/submit_coolmuc4_cv_scaling.sh frequentist
```

## 04_config

`coolmuc4.env.example` documents optional persistent site/user overrides. Copy it outside the repository to the location expected by `coolmuc4_common.sh`; do not commit machine-specific credentials or private paths.

## 05_notebooks

Human-facing benchmark reports belong here. Executable benchmark logic should remain in `01_python/` rather than being hidden in notebooks.

- `workstation_performance_report.ipynb` — workstation publication/report notebook;
- `hpc_execution_and_inference.ipynb` — living CoolMUC-4 report, execution-plan demonstration, guarded notebook job submission, raster scaling, and frequentist/Bayesian inference summaries.

The HPC notebook is a control/report plane. It may inspect plans, submit checked-in jobs and analyze finished results, but parallel raster execution belongs inside Slurm allocations rather than on an LRZ login node.

## Profiles, topology and provenance

The package deliberately separates three kinds of HPC knowledge:

- `SitePolicy` — scheduler/site rules such as partitions, node limits, memory ceilings and scheduler polling etiquette;
- `HardwareTopology` — node hardware such as physical/logical cores, sockets, RAM and scratch conventions;
- `BenchmarkProfile` — dated empirical tuning evidence such as the current provisional `8x14` raster preference.

Planner-driven runs emit a JSON manifest next to the JSONL benchmark output. It records the execution plan, benchmark profile, policy/hardware profile, git SHA/dirty state, Python and package versions, Slurm allocation, dataset/chunk metadata, driver topology, per-worker CPU affinity/NUMA facts, Dask thread/memory settings and topology validation result.

The current hrHSA raster benchmark default uses a conservative 224 GiB working budget with 32 GiB reserved for unmanaged/native memory and page cache, leaving 192 GiB Dask-managed per node. That is a reproducibility/tuning choice, **not** an LRZ policy ceiling; the current CoolMUC-4 policy profile records a 488 GiB node maximum.

## Results and provenance

Executable benchmark machinery lives in `benchmarks/`. Curated benchmark outputs retained for reproducibility or publication provenance live separately in `benchmark-runs/`; scratch JSONL, prepared Zarr stores, scheduler logs, Dask state and transient manifests should normally remain outside Git or in ignored scratch locations.

Publication comparisons should only combine records with compatible scientific kernels, materialization semantics, input sizes, chunking, execution layouts and benchmark profiles. Warm-up observations are not publication repeats, and failed/partial jobs should not be mixed into performance summaries.
