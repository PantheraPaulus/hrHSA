"""Shared sampling-option helpers for Bayesian hrHSA workflows.

The statistical model and the sampler execution geometry are deliberately kept
separate. In particular, PyMC's ``cores`` argument controls native PyMC chain
multiprocessing, but it is not the chain-execution switch used by JAX-backed
BlackJAX/NumPyro samplers. Those backends expose ``chain_method`` instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


JAX_NUTS_SAMPLERS = frozenset({"blackjax", "numpyro"})
JAX_CHAIN_METHODS = frozenset({"parallel", "vectorized"})


def _pymc_version_tuple(version: str | None) -> tuple[int, int, int]:
    if version is None:
        try:
            import pymc as pm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyMC is required; install hsa[bayesian].") from exc
        version = str(pm.__version__)

    parts: list[int] = []
    for raw in str(version).split(".")[:3]:
        digits = "".join(character for character in raw if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        raise ValueError(f"Could not parse PyMC version {version!r}.")
    return tuple((parts + [0, 0, 0])[:3])


def configure_pymc_sampling(
    sample_kwargs: Mapping[str, Any] | None = None,
    *,
    nuts_sampler: str | None = None,
    chain_method: Literal["parallel", "vectorized"] | None = None,
    pymc_version: str | None = None,
) -> dict[str, Any]:
    """Return PyMC sampling kwargs with explicit external-NUTS geometry.

    ``chain_method`` is deliberately separate from PyMC's ``cores`` argument.
    PyMC 6.3+ lifts JAX driver options supplied through ``nuts={...}`` to
    ``sample_jax_nuts``. PyMC 6.0--6.2 does not: placing ``chain_method`` in the
    nested NUTS dictionary leaves the JAX driver on its default chain method and
    forwards the value only to kernel kwargs. hrHSA therefore rejects explicit
    chain-method configuration on those releases rather than silently running a
    different geometry than requested.

    PyMC 5 uses the historical ``nuts_sampler_kwargs={...}`` convention.
    """
    sampling = {} if sample_kwargs is None else dict(sample_kwargs)

    existing_sampler = sampling.get("nuts_sampler")
    if nuts_sampler is not None:
        if existing_sampler is not None and existing_sampler != nuts_sampler:
            raise ValueError(
                "Conflicting NUTS samplers: sample_kwargs specifies "
                f"{existing_sampler!r}, but nuts_sampler={nuts_sampler!r}."
            )
        sampling.setdefault("nuts_sampler", nuts_sampler)

    effective_sampler = sampling.get("nuts_sampler")
    if chain_method is None:
        return sampling

    if chain_method not in JAX_CHAIN_METHODS:
        raise ValueError("chain_method must be 'parallel' or 'vectorized'.")
    if effective_sampler not in JAX_NUTS_SAMPLERS:
        raise ValueError(
            "chain_method applies only to JAX NUTS backends "
            "('blackjax' or 'numpyro'); set nuts_sampler explicitly."
        )

    version = _pymc_version_tuple(pymc_version)
    major, minor, _ = version
    if major == 6 and minor < 3:
        raise RuntimeError(
            "PyMC 6.0--6.2 does not route JAX chain_method from pm.sample() "
            "to sample_jax_nuts(). Upgrade to PyMC >= 6.3 or call the PyMC JAX "
            "driver directly for a controlled diagnostic."
        )

    container_key = "nuts" if major >= 6 else "nuts_sampler_kwargs"

    supplied_containers = [
        key for key in ("nuts", "nuts_sampler_kwargs") if key in sampling
    ]
    if len(supplied_containers) > 1:
        raise ValueError(
            "Specify external NUTS options in only one of 'nuts' or "
            "'nuts_sampler_kwargs'."
        )
    if supplied_containers:
        container_key = supplied_containers[0]

    nested = dict(sampling.get(container_key) or {})
    existing_method = nested.get("chain_method")
    if existing_method is not None and existing_method != chain_method:
        raise ValueError(
            "Conflicting JAX chain methods: sampling kwargs specify "
            f"{existing_method!r}, but chain_method={chain_method!r}."
        )
    nested.setdefault("chain_method", chain_method)
    sampling[container_key] = nested
    return sampling


__all__ = [
    "JAX_CHAIN_METHODS",
    "JAX_NUTS_SAMPLERS",
    "configure_pymc_sampling",
]
