"""Availability-domain constraints for SSF choice sets."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
import xarray as xr

from hsa.ssf.choice_sets import movement_speed_caps, sample_available_steps
from hsa.ssf.environment import check_raster_coverage


def redraw_available_steps_inside_raster(
    choices: gpd.GeoDataFrame,
    movement: dict[str, Any],
    env: xr.DataArray | xr.Dataset,
    *,
    id_col: str,
    n_available: int,
    speed_margin: float = 1.05,
    stratum_col: str = "stratum_id",
    seed: int = 42,
    max_rounds: int = 20,
    observed_outside: str = "raise",
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Constrain SSF choice sets to environmental-raster support.

    Available alternatives outside the raster are replaced by rejection sampling:
    each unsupported candidate is redrawn from the same movement proposal until a
    raster-covered endpoint is obtained. Already-valid alternatives are retained.
    This avoids nearest-neighbour edge snapping while preserving one observed plus
    exactly ``n_available`` alternatives per retained stratum.

    Observed endpoints are never moved. ``observed_outside='raise'`` (the safe
    default) stops when an observed endpoint lacks raster support.
    ``observed_outside='exclude'`` instead removes the *entire affected stratum*
    before available alternatives are redrawn. Whole-stratum exclusion is
    required because retaining alternatives while deleting the observed endpoint
    would destroy the conditional-choice design.

    Candidate-wise rejection samples from the movement proposal conditional on
    raster support. Its additional normalizing factor is constant for all
    alternatives in a stratum and therefore cancels from the conditional-choice
    likelihood. Excluding unsupported observed strata changes the analysis
    population to movements for which environmental support exists and should
    therefore always be reported.
    """
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")
    if choices.crs is None:
        raise ValueError("choices.crs is None.")
    if observed_outside not in {"raise", "exclude"}:
        raise ValueError("observed_outside must be 'raise' or 'exclude'.")

    current = choices.copy()
    speed_caps = movement_speed_caps(
        movement,
        id_col=id_col,
        margin=speed_margin,
    )

    inside = check_raster_coverage(current, env)
    outside = current.loc[~inside]
    n_initial = int(len(outside))
    n_initial_strata = int(outside[stratum_col].nunique()) if n_initial else 0
    n_available_outside_initial = int(
        ((~inside) & ~current["used"].astype(bool)).sum()
    )

    observed_outside_rows = current.loc[
        (~inside) & current["used"].astype(bool)
    ].copy()
    n_observed_outside_initial = int(len(observed_outside_rows))
    excluded_stratum_ids: list[Any] = []
    excluded_by_id: dict[Any, int] = {}

    if not observed_outside_rows.empty:
        excluded_stratum_ids = (
            observed_outside_rows[stratum_col]
            .drop_duplicates()
            .tolist()
        )
        excluded_by_id = (
            observed_outside_rows.groupby(id_col, sort=False)[stratum_col]
            .nunique()
            .astype(int)
            .to_dict()
        )

        if observed_outside == "raise":
            raise ValueError(
                f"{n_observed_outside_initial:,} observed SSF endpoints fall outside "
                "the environmental raster. Observed endpoints cannot be redrawn; "
                "expand the raster or call prepare_choice_sets(..., "
                "observed_outside='exclude') to exclude the complete unsupported "
                f"strata. Counts by individual: {excluded_by_id}"
            )

        current = current.loc[
            ~current[stratum_col].isin(excluded_stratum_ids)
        ].copy()
        if current.empty:
            raise ValueError(
                "Excluding observed endpoints outside the raster removed every SSF "
                "stratum; the environmental raster does not support this analysis."
            )
        current = current.reset_index(drop=True)
        inside = check_raster_coverage(current, env)

    n_strata_redrawn_initial = (
        int(current.loc[~inside, stratum_col].nunique())
        if not inside.all()
        else 0
    )

    rounds = 0
    n_available_replaced_total = 0

    while not inside.all():
        if rounds >= max_rounds:
            remaining_mask = (~inside) & ~current["used"].astype(bool)
            remaining = int(remaining_mask.sum())
            remaining_by_stratum = (
                current.loc[remaining_mask]
                .groupby(stratum_col, sort=False)
                .size()
                .astype(int)
                .to_dict()
            )
            raise RuntimeError(
                "Could not obtain raster-covered replacements for all SSF "
                f"alternatives after {max_rounds} rejection rounds; {remaining} "
                "endpoints remain outside. Remaining counts by stratum: "
                f"{remaining_by_stratum}"
            )

        observed_outside_rows = current.loc[
            (~inside) & current["used"].astype(bool)
        ]
        if not observed_outside_rows.empty:
            raise RuntimeError(
                "Observed endpoints remain outside the raster after applying the "
                "requested observed_outside policy."
            )

        invalid_mask = (~inside) & ~current["used"].astype(bool)
        bad_strata = pd.Index(
            current.loc[invalid_mask, stratum_col].drop_duplicates()
        )
        observed = current.loc[
            current[stratum_col].isin(bad_strata)
            & current["used"].astype(bool)
        ].copy()

        invalid_counts = (
            current.loc[invalid_mask]
            .groupby(stratum_col, sort=False)
            .size()
        )
        max_invalid = int(invalid_counts.max())

        # Draw more than the number of missing candidates so strata close to the
        # raster edge usually obtain enough accepted replacements in one round.
        proposal_draws = max(n_available, 4 * max_invalid)
        proposal_batch = sample_available_steps(
            observed,
            movement["summary"],
            id_col=id_col,
            stratum_col=stratum_col,
            n_available=proposal_draws,
            max_speed_kmh=speed_caps,
            seed=seed + rounds + 1,
        )
        proposal_avail = proposal_batch.loc[
            ~proposal_batch["used"].astype(bool)
        ].copy()
        proposal_inside = check_raster_coverage(proposal_avail, env)
        proposal_avail = proposal_avail.loc[proposal_inside].copy()

        drop_indices: list[int] = []
        replacement_frames: list[gpd.GeoDataFrame] = []

        for sid in bad_strata:
            invalid_indices = current.index[
                invalid_mask & current[stratum_col].eq(sid)
            ].tolist()
            accepted = proposal_avail.loc[
                proposal_avail[stratum_col].eq(sid)
            ]
            take = min(len(invalid_indices), len(accepted))
            if take == 0:
                continue

            target_indices = invalid_indices[:take]
            replacements = accepted.iloc[:take].copy()

            # candidate_id is only a label within a stratum. Reuse the labels of
            # the rejected slots so valid alternatives and choice-set ordering are
            # otherwise untouched.
            replacements["candidate_id"] = current.loc[
                target_indices, "candidate_id"
            ].to_numpy()

            drop_indices.extend(target_indices)
            replacement_frames.append(replacements)

        if drop_indices:
            kept = current.drop(index=drop_indices)
            current = gpd.GeoDataFrame(
                pd.concat([kept, *replacement_frames], ignore_index=True),
                geometry="geometry",
                crs=choices.crs,
            )
            current = current.sort_values(
                [stratum_col, "candidate_id"]
            ).reset_index(drop=True)
            n_available_replaced_total += len(drop_indices)

        inside = check_raster_coverage(current, env)
        rounds += 1

    check = current.groupby(stratum_col, sort=False)["used"].agg(
        n_choices="size",
        n_used="sum",
    )
    if not check["n_choices"].eq(n_available + 1).all():
        raise RuntimeError(
            "Raster-domain filtering changed SSF choice-set size unexpectedly."
        )
    if not check["n_used"].eq(1).all():
        raise RuntimeError(
            "Raster-domain filtering produced a stratum without exactly one used choice."
        )

    diagnostics: dict[str, Any] = {
        "observed_outside_policy": observed_outside,
        "n_outside_initial": n_initial,
        "n_strata_outside_initial": n_initial_strata,
        "n_available_outside_initial": n_available_outside_initial,
        "n_observed_outside_initial": n_observed_outside_initial,
        "n_observed_strata_excluded": len(excluded_stratum_ids),
        "observed_strata_excluded_by_id": excluded_by_id,
        "excluded_stratum_ids": excluded_stratum_ids,
        "n_strata_redrawn_initial": n_strata_redrawn_initial,
        "n_available_replaced_total": n_available_replaced_total,
        "redraw_rounds": rounds,
        "n_outside_final": 0,
    }
    return current, diagnostics


__all__ = ["redraw_available_steps_inside_raster"]
