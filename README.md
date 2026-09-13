<p align="center">
  <img src="logo/logo_rsf2.png" width="600" alt="hrHSA logo">
</p>

# hrHSA

**hrHSA** *(high-resolution Habitat Selection Analysis)* is a scalable Python toolkit for statistical inference on movement and habitat selection in heterogeneous spatial systems. It combines positional data with environmental covariates to estimate how spatial structure, individual variation, and changing conditions influence observed locations through resource-selection, step-selection, spatial point-process, and movement models.

The package emphasizes reproducible workflows, computational efficiency, and scalable analyses of large telemetry and raster datasets, from workstation analyses to distributed HPC execution.

```text
Core capabilities
• Resource Selection Functions (RSFs)
• Step Selection Functions (SSFs) and integrated SSFs (iSSFs)
• Frequentist and hierarchical Bayesian inference
• Spatial point-process and movement modelling
• Large-scale telemetry and raster processing
• Model validation and cross-validation
• Movement simulation and prediction
• Reproducible Dask / SLURM workflows
```

## Package structure

```text
src/hsa/
├── sampling.py          # availability domains, available points, raster sampling
├── features.py          # feature specification and design-matrix construction
├── _time.py             # shared timezone validation helpers
├── diagnostics/         # model and posterior diagnostics
├── movement/            # steps, turning angles, trajectory regularization, kernels
├── rsf/                 # RSF fitting, prediction, validation and CV
├── ssf/                 # SSF / iSSF workflows, choice sets and inference
├── remote_sensing/      # optional remote-sensing / predictor-stack helpers
├── compute/             # Dask, storage and HPC execution utilities
└── simulation/          # movement-simulation components
```

The installable/importable Python package remains `hsa`:

```python
import hsa
```

## Development install

```bash
conda env create -f environment.yml
conda activate hsa
pip install -e .
```

For a pip-based HPC install with optional Dask/SLURM tools:

```bash
pip install -e ".[hpc,earthengine]"
```

For Bayesian workflows:

```bash
pip install -e ".[bayesian]"
```

## Minimal RSF pattern

```python
from hsa import FeatureSpec
from hsa.sampling import sample_available_points, sample_raster_stack
from hsa.rsf import fit_rsf, predict_rsf_surface

spec = FeatureSpec(linear=["ndvi_mean_30m"], add_const=True)

# samples = sample_available_points(domain, n=10_000, used=relocations)
# df = sample_raster_stack(samples, env)
# model, scaler, spec, meta = fit_rsf(df, spec)
# rsf = predict_rsf_surface(env, model, scaler, spec, meta)
```

## Dask / HPC pattern

Local workstation or notebook:

```python
from hsa.compute import make_local_dask_client

client = make_local_dask_client(
    n_workers=8,
    threads_per_worker=1,
    local_directory="/tmp/hsa-dask",
)
```

SLURM-backed execution can be configured through `hsa.compute`. Reproducible benchmark tooling lives under `benchmarks/`, organized as numbered role directories: `01_python/`, `02_slurm/`, `03_sh/`, `04_config/`, and `05_notebooks/`; see `benchmarks/README.md` for the execution and provenance conventions.

Earth Engine must be initialized separately on each Dask worker when Earth Engine calls happen inside distributed tasks:

```python
from hsa.compute import initialize_earth_engine_on_workers

initialize_earth_engine_on_workers(client, project="your-ee-project")
```

## Documentation and examples

The main documentation lives in `docs/` and is built with Sphinx/MyST. Maintained examples and exploratory notebooks live under `notebooks/`; see `notebooks/README.md` for the directory convention.

Project-specific workflows should remain outside the scientific core wherever possible: local paths, reserve-specific exclusions, credentials, Earth Engine project names, and study-specific predictor choices should not be hard-coded into `src/hsa/`.
