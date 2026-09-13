# CoolMUC-4-specific tuning

hrHSA keeps its scientific API portable, but the execution layer can exploit known
properties of LRZ CoolMUC-4. The purpose of the CoolMUC profile is not to hard-code
one configuration; it narrows the tuning space using machine topology and then lets
benchmarks choose the best execution policy.

## Machine profile

The built-in profile is available as

```python
from hsa.compute import COOLMUC4, get_site_profile

print(COOLMUC4)
site = get_site_profile("auto")
```

It records the CPU-node geometry used by the standard CoolMUC-4 nodes:

- 112 physical cores per node;
- two CPU sockets with 56 cores each;
- 512 GiB memory per node;
- `$SCRATCH_DSS` for large shared temporary data;
- `$TMPDIR` for node-local Dask spill data;
- `cm4_tiny` for single-node/shared parallel work;
- `cm4_std` / QoS `cm4_std` for full-node parallel work;
- HDR InfiniBand as the high-speed interconnect.

Site profiles are advisory. They do not silently change the statistical model or
select a Dask worker geometry without benchmark evidence.

## Probe the allocated compute node first

Before the first serious benchmark, run the topology probe inside a CoolMUC
allocation:

```bash
python benchmarks/hpc/probe_coolmuc4.py > coolmuc4-probe.json
```

It records:

- CPU model;
- Slurm CPU affinity;
- `lscpu` topology;
- `numactl --hardware` output when available;
- visible IPv4 network interfaces;
- Slurm allocation metadata;
- the currently selected `HRHSA_DASK_INTERFACE`.

This is particularly important for multi-node Dask. Do not assume the high-speed
interface is named `ib0`; verify the interface on the actual compute nodes.

Persistent user/site overrides can be stored in

```text
~/.config/hrhsa/coolmuc4.env
```

using `benchmarks/hpc/coolmuc4.env.example` as the template. The batch scripts
source this file after LRZ's recommended `--export=NONE` isolation.

## Physical cores, not accidental hyperthreads

The supplied CoolMUC batch scripts use

```text
--hint=nomultithread
```

and topology-sensitive worker steps use

```text
--distribution=block:block
--cpu-bind=cores
```

The objective is reproducible placement on physical Sapphire Rapids cores. Native
BLAS/OpenMP thread teams remain constrained to one thread because Dask provides the
outer task-level concurrency.

## Worker-process geometry is a benchmark parameter

A 112-core node does **not** imply that 112 one-thread Python worker processes are
optimal. hrHSA's fused surface kernel and chunk-local extraction spend much of their
time in NumPy/Zarr code, where a Dask worker can execute several independent blocks
through threads while sharing process memory and storage handles.

The CoolMUC profile therefore exposes candidate full-node geometries:

```python
from hsa.compute import COOLMUC4

for geometry in COOLMUC4.node_geometry_candidates():
    print(geometry.label)
```

Candidate layouts include

```text
112 x 1
56 x 2
28 x 4
16 x 7
14 x 8
8 x 14
4 x 28
2 x 56
```

where the first value is Dask worker processes and the second is threads per worker.
Every layout exposes exactly 112 Dask execution threads.

Run the topology sweep after preparing the medium Zarr input:

```bash
sbatch benchmarks/hpc/coolmuc4_worker_geometry.sbatch
```

Slurm binds each worker process to a compact physical-core set. The experiment
measures both point extraction and fused surface prediction three times for each
geometry and records worker memory, throughput and task density.

Do not select one geometry for the entire package automatically. The optimum for
surface prediction can differ from point sampling, LOIO fold execution or Bayesian
sampling.

## Worker memory and chunk size belong together

Moving from many processes to fewer threaded workers increases the memory available
to each worker but also increases the number of concurrent tasks inside it. The site
profile provides a conservative bound:

```python
geometry = COOLMUC4.node_geometry_candidates()[4]

memory_gib = COOLMUC4.memory_per_worker_gib(geometry.workers)
chunk_mb = COOLMUC4.recommend_chunk_mb(geometry)
```

The chunk recommendation includes a working-set multiplier for input, output and
temporary block arrays. It is a safety bound; the final chunk size should still be
chosen by a benchmark sweep.

## Exploit `$TMPDIR` and `$SCRATCH_DSS` differently

These filesystems serve different roles in hrHSA:

```text
$TMPDIR
    Dask spill / transient worker-local files

$SCRATCH_DSS
    shared environmental Zarr stores
    prepared Parquet datasets
    benchmark JSONL output
```

Do not spill Dask intermediates to shared DSS when node-local temporary storage is
available. Conversely, multi-node workers must not depend on another node's local
`$TMPDIR` for shared environmental input.

## Zarr v3 sharding on DSS

Logical Dask chunks and physical files do not have to be identical. Zarr v3 can
store several independently readable chunks inside a larger shard. This can reduce
filesystem metadata operations and file counts on shared storage.

hrHSA keeps sharding opt-in:

```python
from hsa.compute import write_raster_stack_zarr

write_raster_stack_zarr(
    env,
    "$SCRATCH_DSS/project/environment.zarr",
    chunks={"band": -1, "y": 1024, "x": 1024},
    target_shard_mb=512,
    zarr_format=3,
)
```

or inspect the proposed layout first:

```python
from hsa.compute import plan_zarr_shards, zarr_storage_units

chunks = {"band": -1, "y": 1024, "x": 1024}
shards = plan_zarr_shards(env, chunks, target_shard_mb=512)
print(shards)
print(zarr_storage_units(env, chunks), zarr_storage_units(env, shards))
```

Benchmark at least unsharded, roughly 256 MiB, 512 MiB and 1 GiB shard targets on
DSS. Sharding can reduce metadata pressure, but excessively large shards can hurt
random access; it is therefore not a universal default.

## Use the high-speed interface deliberately for multi-node Dask

CoolMUC-4 provides HDR InfiniBand. Once the probe identifies the appropriate
IP-capable high-speed interface, put it in

```bash
export HRHSA_DASK_INTERFACE="<verified-interface>"
```

inside `~/.config/hrhsa/coolmuc4.env`.

The supplied multi-node and worker-geometry launchers pass that interface to both
Dask scheduler and workers. If the variable is empty they retain Dask's default
network-interface selection.

For a multi-node performance result, archive the probe output alongside the JSONL
benchmark records so the network path is auditable.

## Recommended tuning order

Use this order rather than tuning everything simultaneously:

1. verify topology and network interfaces with `probe_coolmuc4.py`;
2. use physical-core binding and one native numerical thread per Dask task;
3. run `coolmuc4_worker_geometry.sbatch` to choose process/thread geometry;
4. sweep logical chunk sizes with the winning geometry;
5. compare unsharded and Zarr-v3 sharded DSS layouts;
6. only then run 1/2/4-node strong scaling;
7. run end-to-end prepared-dataset/LOIO and Bayesian benchmarks after the storage
   and worker geometry are fixed.

This produces a defensible hierarchy of evidence: CPU topology first, memory/task
geometry second, storage third, inter-node scaling last.
