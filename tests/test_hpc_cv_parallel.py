from __future__ import annotations

from hsa.rsf.bayesian_hpc import _sampling_options
from hsa.rsf.cv_parallel import execute_fold_calls


class _FakeClient:
    def __init__(self):
        self.submitted = []

    def submit(self, function, pure=False, **kwargs):
        self.submitted.append((function, pure, kwargs))
        return (function, kwargs)

    def gather(self, futures):
        # Intentionally reverse completion order; helper must restore fold order.
        return [function(**kwargs) for function, kwargs in reversed(futures)]


def _fold(*, fold_id, value):
    return {"row": {"fold": fold_id}, "value": value}


def test_execute_fold_calls_parallel_restores_fold_order():
    client = _FakeClient()
    calls = [
        {"fold_id": 0, "value": "a"},
        {"fold_id": 1, "value": "b"},
        {"fold_id": 2, "value": "c"},
    ]

    results = execute_fold_calls(_fold, calls, client=client)

    assert [result["row"]["fold"] for result in results] == [0, 1, 2]
    assert len(client.submitted) == 3
    assert all(pure is False for _, pure, _ in client.submitted)


def test_distributed_bayesian_folds_default_to_one_inner_core():
    options = _sampling_options(None, distributed=True)
    assert options["chains"] == 4
    assert options["cores"] == 1


def test_explicit_inner_core_budget_is_preserved():
    options = _sampling_options({"cores": 2, "chains": 4}, distributed=True)
    assert options["cores"] == 2


def test_serial_bayesian_sampling_does_not_force_cores():
    options = _sampling_options(None, distributed=False)
    assert "cores" not in options
