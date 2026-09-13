"""Compatibility helpers for recombining independently sampled Bayesian chains."""

from __future__ import annotations

from typing import Any, Iterable


def _group_names(container: Any) -> list[str]:
    """Return top-level ArviZ group names for old InferenceData or new DataTree."""
    groups = getattr(container, "groups", None)
    if groups is None:
        raise TypeError("Inference container exposes no groups information.")
    raw_groups = groups() if callable(groups) else groups

    names: list[str] = []
    for raw in raw_groups:
        name = str(raw).strip("/")
        if not name:
            continue
        # ArviZ inference groups are top-level. Ignore any nested descendant paths
        # should a future DataTree expose them through ``groups`` as well.
        if "/" in name:
            continue
        names.append(name)
    return names


def _group_dataset(container: Any, group: str, xr):
    """Extract an xarray.Dataset from an InferenceData group or DataTree node."""
    node = getattr(container, group)
    if isinstance(node, xr.Dataset):
        return node

    to_dataset = getattr(node, "to_dataset", None)
    if callable(to_dataset):
        try:
            return to_dataset(inherit=False)
        except TypeError:  # older xarray DataTree API
            return to_dataset()

    raise TypeError(
        f"Inference group {group!r} is neither an xarray.Dataset nor a DataTree node."
    )


def combine_single_chain_inference(idatas: Iterable[Any]):
    """Combine one-chain inference containers into one multi-chain container.

    Accepts both legacy ArviZ ``InferenceData`` objects and modern xarray
    ``DataTree`` objects. Chain-bearing groups are concatenated along ``chain``;
    non-chain groups such as ``observed_data`` and ``constant_data`` are retained
    once. A modern ``xarray.DataTree`` is returned whenever available, otherwise
    the legacy ArviZ ``InferenceData`` constructor is used as a fallback.
    """
    idatas = list(idatas)
    if not idatas:
        raise ValueError("At least one inference container is required.")

    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - Bayesian optional dependency
        raise ImportError("xarray is required for chain recombination.") from exc

    reference_groups = _group_names(idatas[0])
    for idata in idatas[1:]:
        if _group_names(idata) != reference_groups:
            raise ValueError("One-chain inference containers expose different groups.")

    combined_groups: dict[str, Any] = {}
    for group in reference_groups:
        datasets = [_group_dataset(idata, group, xr) for idata in idatas]
        chain_flags = ["chain" in dataset.dims for dataset in datasets]

        if all(chain_flags):
            normalized = []
            for chain_index, dataset in enumerate(datasets):
                if int(dataset.sizes["chain"]) != 1:
                    raise ValueError(
                        f"Expected one chain in group {group!r}; "
                        f"got {dataset.sizes['chain']}."
                    )
                normalized.append(dataset.assign_coords(chain=[chain_index]))
            combined_groups[group] = xr.concat(
                normalized,
                dim="chain",
                combine_attrs="override",
            )
        elif any(chain_flags):
            raise ValueError(f"Inconsistent chain dimension in group {group!r}.")
        else:
            combined_groups[group] = datasets[0].copy(deep=False)

    data_tree = getattr(xr, "DataTree", None)
    if data_tree is not None:
        return data_tree.from_dict(combined_groups)

    # Compatibility fallback for environments predating xarray.DataTree.
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover - Bayesian optional dependency
        raise ImportError("ArviZ is required for legacy chain recombination.") from exc
    return az.InferenceData(**combined_groups)


__all__ = ["combine_single_chain_inference"]
