"""Small execution helpers shared by parallel cross-validation kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def execute_fold_calls(
    function: Callable[..., dict[str, Any]],
    calls: Sequence[dict[str, Any]],
    *,
    client=None,
) -> list[dict[str, Any]]:
    """Execute independent fold calls serially or through a Dask client.

    Dask ``Client.gather`` preserves the order of the submitted future list. For
    production CV results that additionally expose ``result['row']['fold']``, fold
    order is restored explicitly as a defensive measure. Generic benchmark fold
    results can therefore reuse this executor without having to mimic the full
    production result schema.
    """
    if client is None:
        results = [function(**call) for call in calls]
    else:
        futures = [client.submit(function, pure=False, **call) for call in calls]
        results = client.gather(futures)

    if all(
        isinstance(result.get("row"), dict) and "fold" in result["row"]
        for result in results
    ):
        return sorted(results, key=lambda result: int(result["row"]["fold"]))
    return list(results)


__all__ = ["execute_fold_calls"]
