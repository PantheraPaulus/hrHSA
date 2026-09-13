from __future__ import annotations

from types import SimpleNamespace

from hsa.compute.partitioned import _default_graph_partitions


class _Client:
    def __init__(self, nthreads: list[int]):
        self._workers = {
            f"worker-{index}": {"nthreads": threads}
            for index, threads in enumerate(nthreads)
        }

    def scheduler_info(self):
        return {"workers": self._workers}


def test_default_graph_partitions_preserves_workstation_scale():
    client = _Client([2, 2, 2, 2, 2, 2])
    assert _default_graph_partitions(client, 1000) == 96


def test_default_graph_partitions_caps_large_hpc_nodes_at_256():
    client = _Client([14] * 8)
    assert _default_graph_partitions(client, 1000) == 256


def test_default_graph_partitions_never_exceeds_available_partitions():
    client = _Client([14] * 8)
    assert _default_graph_partitions(client, 10) == 10
