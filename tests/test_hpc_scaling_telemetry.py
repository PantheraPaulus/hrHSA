from __future__ import annotations

import pytest

from hsa.compute.scaling_telemetry import (
    distributed_node_runtime_delta,
    distributed_node_runtime_snapshot,
    linux_vmstat_delta,
    linux_vmstat_snapshot,
    scheduler_process_delta,
)


def test_linux_vmstat_snapshot_and_delta(tmp_path):
    before_path = tmp_path / "vmstat-before"
    after_path = tmp_path / "vmstat-after"

    before_path.write_text(
        "\n".join(
            [
                "pgfault 1000",
                "pgmajfault 10",
                "pgscan_kswapd 20",
                "pgscan_direct 3",
                "pgsteal_kswapd 15",
                "pgsteal_direct 2",
                "pswpin 4",
                "pswpout 5",
                "workingset_refault_anon 7",
                "workingset_refault_file 11",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after_path.write_text(
        "\n".join(
            [
                "pgfault 1300",
                "pgmajfault 14",
                "pgscan_kswapd 29",
                "pgscan_direct 7",
                "pgsteal_kswapd 22",
                "pgsteal_direct 5",
                "pswpin 6",
                "pswpout 8",
                "workingset_refault_anon 12",
                "workingset_refault_file 20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    before = linux_vmstat_snapshot(before_path)
    after = linux_vmstat_snapshot(after_path)
    delta = linux_vmstat_delta(before, after)

    assert delta["page_faults"] == pytest.approx(300)
    assert delta["major_page_faults"] == pytest.approx(4)
    assert delta["page_scans"] == pytest.approx(13)
    assert delta["page_steals"] == pytest.approx(10)
    assert delta["swap_pages_in"] == pytest.approx(2)
    assert delta["swap_pages_out"] == pytest.approx(3)
    assert delta["workingset_refaults"] == pytest.approx(14)


def test_distributed_node_snapshot_deduplicates_workers_on_same_host():
    class FakeClient:
        def run(self, function):
            return {
                "tcp://a:1": {"hostname": "node-a", "system": {}, "vmstat": {}},
                "tcp://a:2": {"hostname": "node-a", "system": {}, "vmstat": {}},
                "tcp://b:1": {"hostname": "node-b", "system": {}, "vmstat": {}},
            }

    observed = distributed_node_runtime_snapshot(FakeClient())
    assert sorted(observed) == ["node-a", "node-b"]


def test_distributed_node_delta_aggregates_counts_but_means_cpu_fractions():
    before = {
        "node-a": {
            "system": {
                "available": True,
                "timestamp": 1.0,
                "cpu_times": {"user": 10, "system": 5, "idle": 20, "iowait": 1},
                "cpu_stats": {"ctx_switches": 100},
                "disk_io": {"read_bytes": 1000, "write_bytes": 2000},
            },
            "vmstat": {
                "available": True,
                "counters": {"pgfault": 100, "pgmajfault": 2, "pswpin": 0, "pswpout": 0},
            },
        },
        "node-b": {
            "system": {
                "available": True,
                "timestamp": 1.0,
                "cpu_times": {"user": 10, "system": 5, "idle": 20, "iowait": 1},
                "cpu_stats": {"ctx_switches": 200},
                "disk_io": {"read_bytes": 3000, "write_bytes": 4000},
            },
            "vmstat": {
                "available": True,
                "counters": {"pgfault": 200, "pgmajfault": 3, "pswpin": 0, "pswpout": 0},
            },
        },
    }
    after = {
        "node-a": {
            "system": {
                "available": True,
                "timestamp": 3.0,
                "cpu_times": {"user": 18, "system": 9, "idle": 24, "iowait": 3},
                "cpu_stats": {"ctx_switches": 160},
                "disk_io": {"read_bytes": 5000, "write_bytes": 2500},
            },
            "vmstat": {
                "available": True,
                "counters": {"pgfault": 150, "pgmajfault": 4, "pswpin": 0, "pswpout": 0},
            },
        },
        "node-b": {
            "system": {
                "available": True,
                "timestamp": 3.0,
                "cpu_times": {"user": 16, "system": 7, "idle": 26, "iowait": 2},
                "cpu_stats": {"ctx_switches": 280},
                "disk_io": {"read_bytes": 9000, "write_bytes": 4500},
            },
            "vmstat": {
                "available": True,
                "counters": {"pgfault": 280, "pgmajfault": 8, "pswpin": 0, "pswpout": 0},
            },
        },
    }

    delta = distributed_node_runtime_delta(before, after)
    aggregate = delta["aggregate"]
    assert aggregate["disk_read_bytes"] == pytest.approx(10_000)
    assert aggregate["ctx_switches"] == pytest.approx(140)
    assert aggregate["page_faults"] == pytest.approx(130)
    assert aggregate["major_page_faults"] == pytest.approx(7)
    assert 0 <= aggregate["mean_cpu_busy_fraction"] <= 1
    assert 0 <= aggregate["mean_cpu_iowait_fraction"] <= 1


def test_scheduler_process_delta_adds_fraction_of_one_core():
    before = {
        "available": True,
        "pid": 1,
        "cpu_total_seconds": 10.0,
        "context_switches": 100,
    }
    after = {
        "available": True,
        "pid": 1,
        "cpu_total_seconds": 12.0,
        "context_switches": 130,
        "hostname": "node-a",
    }

    delta = scheduler_process_delta(before, after, wall_seconds=4.0)
    assert delta["cpu_total_seconds"] == pytest.approx(2.0)
    assert delta["context_switches"] == pytest.approx(30)
    assert delta["cpu_fraction_of_one_core"] == pytest.approx(0.5)
    assert delta["hostname"] == "node-a"
