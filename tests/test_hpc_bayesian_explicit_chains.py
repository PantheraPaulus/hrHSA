from __future__ import annotations

import numpy as np
import pytest

from hsa.rsf._inference_tree import combine_single_chain_inference


def _idata(*, chains: int, draws: int, alpha_values):
    xr = pytest.importorskip("xarray")
    if not hasattr(xr, "DataTree"):
        pytest.skip("xarray.DataTree is required for this compatibility regression test")

    alpha = np.broadcast_to(
        np.asarray(alpha_values, dtype=float).reshape(chains, 1),
        (chains, draws),
    ).copy()
    coords = {
        "chain": np.arange(chains, dtype=int),
        "draw": np.arange(draws, dtype=int),
    }
    posterior = xr.Dataset(
        {"alpha": (("chain", "draw"), alpha)},
        coords=coords,
    )
    sample_stats = xr.Dataset(
        {
            "diverging": (
                ("chain", "draw"),
                np.zeros((chains, draws), dtype=bool),
            )
        },
        coords=coords,
    )
    return xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "sample_stats": sample_stats,
        }
    )


def test_explicit_chain_recombination_restores_chain_axis():
    idatas = [
        _idata(chains=1, draws=20, alpha_values=[float(chain)])
        for chain in range(4)
    ]

    combined = combine_single_chain_inference(idatas)

    assert int(combined.posterior.sizes["chain"]) == 4
    assert int(combined.posterior.sizes["draw"]) == 20
    assert combined.posterior["chain"].values.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(
        combined.posterior["alpha"].mean("draw").values,
        np.arange(4, dtype=float),
    )
    assert int(combined.sample_stats["diverging"].sum()) == 0


def test_explicit_chain_recombination_rejects_non_single_chain_inputs():
    bad = _idata(chains=2, draws=5, alpha_values=[0.0, 0.0])

    with pytest.raises(ValueError, match="Expected one chain"):
        combine_single_chain_inference([bad])
