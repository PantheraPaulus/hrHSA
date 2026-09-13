from __future__ import annotations

import pytest

from hsa.compute.telemetry import (
    aggregate_worker_runtime_delta,
    process_runtime_delta,
    summarize_task_stream,
    system_runtime_delta,
)


def test_process_runtime_delta_subtracts_counters_and_keeps_end_memory():
    before = {
        "available": True,
        "pid": 10,
        "cpu_total_seconds": 1.5,
        "cpu_user_seconds": 1.0,
        "cpu_system_seconds": 0.5,
        "context_switches": 100,
        "read_bytes": 1000,
    }
    after = {
        "available": True,
        "pid": 10,
        "cpu_total_seconds": 4.0,
        "cpu_user_seconds": 3.0,
        "cpu_system_seconds": 1.0,
        "context_switches": 160,
        "read_bytes": 5000,
        "rss_bytes": 1024,
        "pss_bytes": 768,
        "uss_bytes": 512,
        "num_threads": 8,
        "cpu_affinity": [0, 1, 2, 3],
    }

    delta = process_runtime_delta(before, after)
    assert delta["cpu_total_seconds"] == pytest.approx(2.5)
    assert delta["context_switches"] == pytest.approx(60)
    assert delta["read_bytes"] == pytest.approx(4000)
    assert delta["end_rss_bytes"] == 1024
    assert delta["end_pss_bytes"] == 768
    assert delta["cpu_affinity"] == [0, 1, 2, 3]


def test_aggregate_worker_runtime_delta_sums_matched_workers():
    before = {
        "a": {"available": True, "pid": 1, "cpu_total_seconds": 1.0, "context_switches": 10},
        "b": {"available": True, "pid": 2, "cpu_total_seconds": 2.0, "context_switches": 20},
    }
    after = {
        "a": {
            "available": True,
            "pid": 1,
            "cpu_total_seconds": 3.0,
            "context_switches": 20,
            "rss_bytes": 100,
            "pss_bytes": 80,
            "uss_bytes": 60,
        },
        "b": {
            "available": True,
            "pid": 2,
            "cpu_total_seconds": 5.0,
            "context_switches": 40,
            "rss_bytes": 200,
            "pss_bytes": 160,
            "uss_bytes": 120,
        },
    }

    delta = aggregate_worker_runtime_delta(before, after)
    assert delta["workers_matched"] == 2
    assert delta["cpu_total_seconds"] == pytest.approx(5.0)
    assert delta["context_switches"] == pytest.approx(30)
    assert delta["end_pss_bytes_total"] == pytest.approx(240)
    assert delta["end_uss_bytes_max"] == pytest.approx(120)


def test_system_runtime_delta_reports_busy_iowait_context_switches_and_io():
    before = {
        "available": True,
        "timestamp": 10.0,
        "cpu_times": {"user": 100.0, "system": 50.0, "idle": 300.0, "iowait": 10.0},
        "cpu_stats": {"ctx_switches": 1000.0},
        "disk_io": {"read_bytes": 100.0, "write_bytes": 200.0},
    }
    after = {
        "available": True,
        "timestamp": 12.0,
        "cpu_times": {"user": 108.0, "system": 54.0, "idle": 304.0, "iowait": 12.0},
        "cpu_stats": {"ctx_switches": 1300.0},
        "disk_io": {"read_bytes": 1124.0, "write_bytes": 712.0},
    }

    delta = system_runtime_delta(before, after)
    assert delta["elapsed_seconds"] == pytest.approx(2.0)
    assert delta["cpu_busy_fraction"] == pytest.approx(12 / 18)
    assert delta["cpu_iowait_fraction"] == pytest.approx(2 / 18)
    assert delta["ctx_switches"] == pytest.approx(300)
    assert delta["disk_read_bytes"] == pytest.approx(1024)
    assert delta["disk_write_bytes"] == pytest.approx(512)


def test_summarize_task_stream_accumulates_actions_and_compute_parallelism():
    events = [
        {
            "nbytes": 100,
            "startstops": [
                {"action": "transfer", "start": 0.0, "stop": 0.5},
                {"action": "compute", "start": 0.5, "stop": 2.5},
            ],
        },
        {
            "nbytes": 200,
            "startstops": [
                {"action": "compute", "start": 1.0, "stop": 3.0},
            ],
        },
    ]

    summary = summarize_task_stream(events)
    assert summary["tasks"] == 2
    assert summary["nbytes_observed"] == pytest.approx(300)
    assert summary["compute_seconds"] == pytest.approx(4.0)
    assert summary["transfer_seconds"] == pytest.approx(0.5)
    assert summary["span_seconds"] == pytest.approx(3.0)
    assert summary["compute_parallelism"] == pytest.approx(4 / 3)
    assert summary["compute_task_median_seconds"] == pytest.approx(2.0)
