from __future__ import annotations

import time

from hsa.compute import benchmark_timer, make_benchmark_record


class _FakeClient:
    def run(self, function):
        return {"worker-a": 100.0, "worker-b": 200.0}

    def scheduler_info(self):
        mib = 1024**2
        return {
            "workers": {
                "worker-a": {"metrics": {"memory": 100 * mib}},
                "worker-b": {"metrics": {"memory": 200 * mib}},
            }
        }


def test_benchmark_timer_records_operation_local_memory():
    with benchmark_timer(memory_interval_seconds=0.01) as timer:
        payload = bytearray(2_000_000)
        time.sleep(0.03)
        assert len(payload) == 2_000_000

    assert timer["wall_seconds"] > 0
    assert timer["operation_peak_rss_mb"] > 0
    assert timer["operation_peak_process_tree_rss_mb"] >= timer["operation_peak_rss_mb"]
    assert timer["memory_samples"] >= 2


def test_benchmark_timer_tracks_total_and_max_worker_rss():
    client = _FakeClient()
    with benchmark_timer(client=client, memory_interval_seconds=0.01) as timer:
        time.sleep(0.02)

    assert timer["worker_operation_peak_rss_total_mb"] == 300.0
    assert timer["worker_operation_peak_rss_max_mb"] == 200.0

    record = make_benchmark_record(
        "memory-smoke",
        timer["wall_seconds"],
        workers=2,
        operation_memory=timer,
        client=client,
    )
    assert record.operation_peak_rss_mb == timer["operation_peak_rss_mb"]
    assert record.operation_peak_process_tree_rss_mb == timer[
        "operation_peak_process_tree_rss_mb"
    ]
    assert record.worker_operation_peak_rss_total_mb == 300.0
    assert record.worker_operation_peak_rss_max_mb == 200.0
    assert record.memory_samples == timer["memory_samples"]
