from __future__ import annotations

import pytest

from hsa.bayesian_sampling import configure_pymc_sampling


def test_jax_chain_method_uses_modern_pymc_nuts_container():
    sampling = configure_pymc_sampling(
        {"draws": 500},
        nuts_sampler="blackjax",
        chain_method="vectorized",
        pymc_version="6.3.1",
    )
    assert sampling["nuts_sampler"] == "blackjax"
    assert sampling["nuts"] == {"chain_method": "vectorized"}
    assert sampling["draws"] == 500


def test_pymc_62_rejects_chain_method_that_pm_sample_cannot_route():
    with pytest.raises(RuntimeError, match="PyMC 6.0--6.2"):
        configure_pymc_sampling(
            nuts_sampler="blackjax",
            chain_method="vectorized",
            pymc_version="6.2.0",
        )


def test_jax_chain_method_uses_legacy_external_sampler_container():
    sampling = configure_pymc_sampling(
        nuts_sampler="blackjax",
        chain_method="parallel",
        pymc_version="5.25.1",
    )
    assert sampling["nuts_sampler"] == "blackjax"
    assert sampling["nuts_sampler_kwargs"] == {"chain_method": "parallel"}


def test_existing_nuts_options_are_preserved():
    sampling = configure_pymc_sampling(
        {"nuts": {"max_treedepth": 12}},
        nuts_sampler="numpyro",
        chain_method="vectorized",
        pymc_version="6.3.1",
    )
    assert sampling["nuts"] == {
        "max_treedepth": 12,
        "chain_method": "vectorized",
    }


def test_chain_method_requires_jax_sampler():
    with pytest.raises(ValueError, match="JAX NUTS"):
        configure_pymc_sampling(
            nuts_sampler="pymc",
            chain_method="parallel",
            pymc_version="6.3.1",
        )


def test_conflicting_sampler_and_chain_method_are_rejected():
    with pytest.raises(ValueError, match="Conflicting NUTS samplers"):
        configure_pymc_sampling(
            {"nuts_sampler": "numpyro"},
            nuts_sampler="blackjax",
            pymc_version="6.3.1",
        )

    with pytest.raises(ValueError, match="Conflicting JAX chain methods"):
        configure_pymc_sampling(
            {"nuts_sampler": "blackjax", "nuts": {"chain_method": "parallel"}},
            chain_method="vectorized",
            pymc_version="6.3.1",
        )
