from __future__ import annotations

import pytest

from hsa.compute import BayesianCPUPlan, plan_bayesian_cpu_execution


def test_workstation_outer_first_geometries_match_measured_policy():
    plan = plan_bayesian_cpu_execution(
        analysis="rsf",
        n_models=24,
        available_cores=24,
        chains=4,
    )
    assert isinstance(plan, BayesianCPUPlan)
    assert plan.objective == "throughput"
    assert plan.sampler == "nutpie"
    assert plan.outer_workers == 24
    assert plan.cores_per_fit == 1
    assert plan.active_cores == 24
    assert plan.unused_cores == 0
    assert plan.profile == "generic"

    plan = plan_bayesian_cpu_execution(
        analysis="rsf",
        n_models=12,
        available_cores=24,
        chains=4,
    )
    assert (plan.outer_workers, plan.cores_per_fit) == (12, 2)

    plan = plan_bayesian_cpu_execution(
        analysis="rsf",
        n_models=6,
        available_cores=24,
        chains=4,
    )
    assert (plan.outer_workers, plan.cores_per_fit) == (6, 4)


def test_generic_full_node_geometries_keep_portable_outer_first_rule():
    expected = {
        28: (28, 4),
        56: (56, 2),
        112: (112, 1),
    }
    for n_models, geometry in expected.items():
        plan = plan_bayesian_cpu_execution(
            analysis="rsf",
            n_models=n_models,
            available_cores=112,
            chains=4,
            objective="throughput",
        )
        assert (plan.outer_workers, plan.cores_per_fit) == geometry
        assert plan.active_cores == 112
        assert plan.unused_cores == 0
        assert plan.sampler == "nutpie"


def test_coolmuc4_profile_uses_measured_model_specific_throughput_geometry():
    rsf = plan_bayesian_cpu_execution(
        analysis="rsf",
        n_models=112,
        available_cores=112,
        chains=4,
        objective="throughput",
        profile="coolmuc4",
    )
    ssf = plan_bayesian_cpu_execution(
        analysis="ssf",
        n_models=112,
        available_cores=112,
        chains=4,
        objective="throughput",
        profile="coolmuc4",
    )
    issf = plan_bayesian_cpu_execution(
        analysis="issf",
        n_models=112,
        available_cores=112,
        chains=4,
        objective="throughput",
        profile="coolmuc4",
    )

    assert (rsf.outer_workers, rsf.cores_per_fit) == (56, 2)
    assert (ssf.outer_workers, ssf.cores_per_fit) == (28, 4)
    assert (issf.outer_workers, issf.cores_per_fit) == (28, 4)
    assert rsf.profile == ssf.profile == issf.profile == "coolmuc4"
    assert rsf.active_cores == ssf.active_cores == issf.active_cores == 112


def test_coolmuc4_profile_falls_back_outside_measured_full_node_case():
    partial = plan_bayesian_cpu_execution(
        analysis="ssf",
        n_models=24,
        available_cores=24,
        chains=4,
        objective="throughput",
        profile="coolmuc4",
    )
    assert (partial.outer_workers, partial.cores_per_fit) == (24, 1)

    fewer_chains = plan_bayesian_cpu_execution(
        analysis="ssf",
        n_models=112,
        available_cores=112,
        chains=2,
        objective="throughput",
        profile="coolmuc4",
    )
    assert (fewer_chains.outer_workers, fewer_chains.cores_per_fit) == (112, 1)


def test_single_fit_backend_recommendations_are_analysis_specific():
    rsf = plan_bayesian_cpu_execution(
        analysis="rsf",
        n_models=1,
        available_cores=24,
    )
    ssf = plan_bayesian_cpu_execution(
        analysis="ssf",
        n_models=1,
        available_cores=24,
    )
    issf = plan_bayesian_cpu_execution(
        analysis="issf",
        n_models=1,
        available_cores=24,
    )

    assert rsf.objective == "latency"
    assert rsf.sampler == "blackjax"
    assert rsf.outer_workers == 1
    assert rsf.cores_per_fit == 4

    assert ssf.sampler == "nutpie"
    assert issf.sampler == "nutpie"
    assert ssf.cores_per_fit == 4
    assert issf.cores_per_fit == 4


def test_plan_can_emit_portable_pymc_sample_kwargs():
    plan = plan_bayesian_cpu_execution(
        analysis="ssf",
        n_models=12,
        available_cores=24,
        chains=4,
    )
    assert plan.pymc_sample_kwargs() == {
        "nuts_sampler": "nutpie",
        "chains": 4,
        "cores": 2,
    }
    assert plan.pymc_sample_kwargs(draws=2000, tune=1500)["draws"] == 2000
    assert plan.pymc_sample_kwargs(draws=2000, tune=1500)["tune"] == 1500


def test_spare_cores_flow_into_chains_but_never_exceed_chain_count():
    plan = plan_bayesian_cpu_execution(
        analysis="ssf",
        n_models=20,
        available_cores=112,
        chains=4,
        objective="throughput",
    )
    assert plan.outer_workers == 20
    assert plan.cores_per_fit == 4
    assert plan.active_cores == 80
    assert plan.unused_cores == 32


def test_invalid_bayesian_execution_requests_are_rejected():
    with pytest.raises(ValueError):
        plan_bayesian_cpu_execution(analysis="other", n_models=1, available_cores=4)
    with pytest.raises(ValueError):
        plan_bayesian_cpu_execution(analysis="rsf", n_models=0, available_cores=4)
    with pytest.raises(ValueError):
        plan_bayesian_cpu_execution(analysis="rsf", n_models=1, available_cores=0)
    with pytest.raises(ValueError):
        plan_bayesian_cpu_execution(
            analysis="rsf",
            n_models=1,
            available_cores=4,
            objective="fastest",
        )
    with pytest.raises(ValueError):
        plan_bayesian_cpu_execution(
            analysis="rsf",
            n_models=1,
            available_cores=4,
            profile="other",
        )
