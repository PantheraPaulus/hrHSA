"""HPC site policy, hardware topology and benchmark-evidence profiles.

These concepts deliberately live separately:

* :class:`SitePolicy` describes scheduler/site rules that can change independently
  of hardware or hrHSA performance.
* :class:`HardwareTopology` records static machine characteristics.
* :class:`BenchmarkProfile` records dated empirical tuning evidence and is allowed
  to become stale when kernels or dependencies change.

Keeping the three layers separate prevents a benchmark observation such as
"8x14 was fastest in August 2026" from becoming an undocumented hardware fact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Workload = Literal[
    "point_sampling",
    "surface_prediction",
    "validation",
    "inference",
    "generic",
]


@dataclass(frozen=True)
class SitePolicy:
    """Scheduler and resource-policy facts for one HPC site."""

    name: str
    scheduler: str
    cluster: str
    shared_partition: str
    shared_qos: str
    full_node_partition: str
    full_node_qos: str
    serial_partition: str
    min_parallel_cores_per_node: int
    max_parallel_nodes: int
    max_memory_per_node_gib: float
    interactive_cluster: str | None = None
    interactive_partition: str | None = None
    scheduler_poll_seconds: float = 600.0
    short_job_threshold_seconds: float = 120.0
    max_job_steps: int | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareTopology:
    """Static hardware characteristics of one compute-node type."""

    name: str
    physical_cores_per_node: int
    logical_cores_per_node: int
    sockets_per_node: int
    cores_per_socket: int
    memory_gib_per_node: float
    scratch_env: str
    local_scratch_env: str
    interconnect: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkProfile:
    """Dated empirical execution preference for a hardware/workload combination."""

    name: str
    site: str
    workloads: tuple[Workload, ...]
    cores_per_node: int
    preferred_geometries: tuple[str, ...]
    chunk_mb: int | None
    evidence_date: str
    evidence: str
    confidence: Literal["provisional", "confirmed"] = "provisional"
    repeat_count: int | None = None

    def applies_to(self, *, site: str, workload: Workload, cores_per_node: int) -> bool:
        return (
            self.site == site
            and workload in self.workloads
            and self.cores_per_node == int(cores_per_node)
        )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["workloads"] = list(self.workloads)
        payload["preferred_geometries"] = list(self.preferred_geometries)
        return payload


# CoolMUC-4 policy facts below mirror the LRZ CoolMUC Cheat Sheet released
# 2026-03-31 (best-before 2026-09-30). The policy object therefore represents a
# dated site interpretation, not an eternal cluster specification.
COOLMUC4_POLICY = SitePolicy(
    name="lrz-coolmuc4-2026-03",
    scheduler="slurm",
    cluster="cm4",
    shared_partition="cm4_tiny",
    shared_qos="cm4_tiny",
    full_node_partition="cm4_std",
    full_node_qos="cm4_std",
    serial_partition="serial_std",
    min_parallel_cores_per_node=17,
    max_parallel_nodes=4,
    max_memory_per_node_gib=488.0,
    interactive_cluster="inter",
    interactive_partition="cm4_inter",
    scheduler_poll_seconds=600.0,
    short_job_threshold_seconds=120.0,
    max_job_steps=40_000,
)


COOLMUC4_HARDWARE = HardwareTopology(
    name="coolmuc4-spr-node",
    physical_cores_per_node=112,
    logical_cores_per_node=224,
    sockets_per_node=2,
    cores_per_socket=56,
    memory_gib_per_node=488.0,
    scratch_env="SCRATCH_DSS",
    local_scratch_env="TMPDIR",
    interconnect="NVIDIA/Mellanox HDR InfiniBand",
)


# Large-raster fixed-112-core campaign supplied with hrHSA's August 2026 HPC
# report. 8x14 had the lowest observed median wall time (10.79 s), while 16x7
# was close (11.21 s). These measurements exercised surface prediction, so this
# profile must not be used as point-sampling evidence.
COOLMUC4_RASTER_112_2026_08 = BenchmarkProfile(
    name="coolmuc4-raster-112-2026-08-v1",
    site="coolmuc4",
    workloads=("surface_prediction",),
    cores_per_node=112,
    preferred_geometries=("8x14", "16x7", "14x8", "28x4"),
    chunk_mb=256,
    evidence_date="2026-08-30",
    evidence=(
        "Large-raster fixed-112-core surface-prediction sweep: 8x14 median "
        "10.79 s, 16x7 median 11.21 s, 14x8 median 14.22 s. The low-process "
        "hybrid region clearly outperformed one-process-per-core, but 8x14 "
        "versus 16x7 remains close."
    ),
    confidence="provisional",
    repeat_count=3,
)


# Partition-native point-sampling tuning on one CoolMUC-4 node, using 1 billion
# points against a ~194 GiB logical raster. The initial geometry/window sweep
# found 8x14/256 fastest at 62.33 s. A second independent geometry sweep at a
# fixed 256-partition graph window bracketed the optimum: 8x14 took 61.06 s,
# 7x16 61.21 s, 4x28 72.78 s, 2x56 149.74 s and 1x112 222.90 s. The profile
# therefore treats 7-8 low-process hybrid workers as the optimum region and
# keeps 8x14 as the preferred symmetric four-workers-per-socket geometry.
COOLMUC4_POINT_112_2026_09 = BenchmarkProfile(
    name="coolmuc4-point-112-2026-09-v1",
    site="coolmuc4",
    workloads=("point_sampling",),
    cores_per_node=112,
    preferred_geometries=("8x14", "7x16", "16x7", "28x4", "4x28"),
    chunk_mb=256,
    evidence_date="2026-09-04",
    evidence=(
        "Partition-native 1B-point/~194-GiB cm4_inter tuning brackets the "
        "point-sampling optimum around 7-8 workers: at a 256-partition graph "
        "window, 8x14 took 61.06 s and 7x16 61.21 s, while 4x28 took 72.78 s, "
        "2x56 149.74 s and 1x112 222.90 s. An independent earlier 8x14/256 "
        "run took 62.33 s; 8x14 is retained for symmetric socket placement."
    ),
    confidence="provisional",
    repeat_count=2,
)


BENCHMARK_PROFILES: tuple[BenchmarkProfile, ...] = (
    COOLMUC4_RASTER_112_2026_08,
    COOLMUC4_POINT_112_2026_09,
)


def benchmark_profile_for(
    *,
    site: str,
    workload: Workload,
    cores_per_node: int,
) -> BenchmarkProfile | None:
    """Return the first registered profile matching an execution request."""
    for profile in BENCHMARK_PROFILES:
        if profile.applies_to(
            site=site,
            workload=workload,
            cores_per_node=cores_per_node,
        ):
            return profile
    return None
