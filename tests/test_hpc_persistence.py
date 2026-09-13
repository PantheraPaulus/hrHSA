from __future__ import annotations

import numpy as np
import pytest

from hsa.compute.persistence import persist_distributed, release_distributed


def test_persist_distributed_keeps_dask_array_partitioned():
    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("distributed")

    cluster = distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        source = da.arange(10_000, chunks=1_000).reshape((100, 100))
        persisted = persist_distributed(client, source)

        assert isinstance(persisted, da.Array)
        assert not isinstance(persisted, np.ndarray)
        assert persisted.npartitions > 1
        assert persisted.chunks == source.chunks
        assert np.asarray(persisted.sum().compute()).item() == np.arange(10_000).sum()

        release_distributed(
            client,
            persisted,
            suppress_worker_errors=False,
        )
        persisted = None

        # Cleanup must not retire or poison the worker pool; the next benchmark
        # repeat should be able to use the same scheduler immediately.
        assert client.submit(lambda: 7).result() == 7
        assert len(client.scheduler_info().get("workers", {})) == 2
    finally:
        try:
            release_distributed(client, locals().get("persisted"))
        finally:
            client.close()
            cluster.close()
