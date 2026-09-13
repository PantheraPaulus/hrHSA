"""Construction of movement-informed step-selection choice sets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import Point

from hsa._time import require_timezone_aware


def wrap_angle(angle):
    """Wrap angle(s) to ``[-pi, pi)``."""
    values = np.asarray(angle)
    return (values + np.pi) % (2 * np.pi) - np.pi


def build_observed_ssf_steps(
    movement: Mapping[str, Any],
    *,
    id_col: str = "Individual_ID",
    stratum_col: str = "stratum_id",
    burst_gap: str | pd.Timedelta = "2h",
) -> gpd.GeoDataFrame:
    """Build one observed SSF row per valid consecutive turning step."""
    if "angle_df" not in movement:
        raise KeyError("movement must contain an 'angle_df' table.")

    source = movement["angle_df"].copy()
    if id_col not in source.columns:
        raise KeyError(f"{id_col!r} not found in movement['angle_df'].")
    if "geometry" not in source.columns:
        raise KeyError("movement['angle_df'] must contain a 'geometry' column.")
    if source.crs is None:
        raise ValueError("movement['angle_df'].crs is None.")

    previous_col = next(
        (
            col
            for col in ("previous_location", "prev_position", "previous_position")
            if col in source
        ),
        None,
    )
    if previous_col is None:
        raise KeyError(
            "movement['angle_df'] must contain one of 'previous_location', "
            "'prev_position' or 'previous_position'."
        )
    if "next_position" not in source.columns:
        raise KeyError("movement['angle_df'] must contain 'next_position'.")

    start_time_col = next(
        (col for col in ("Timestamp", "timestamp", "start_time") if col in source),
        None,
    )
    end_time_col = next(
        (col for col in ("next_timestamp", "end_time") if col in source),
        None,
    )
    if start_time_col is None or end_time_col is None:
        raise KeyError(
            "movement['angle_df'] must contain a start timestamp and "
            "'next_timestamp'/'end_time'."
        )

    d = source.dropna(
        subset=[
            id_col,
            previous_col,
            "geometry",
            "next_position",
            start_time_col,
            end_time_col,
        ]
    ).copy()
    d["start_time"] = require_timezone_aware(
        d[start_time_col],
        name=start_time_col,
    )
    d["end_time"] = require_timezone_aware(
        d[end_time_col],
        name=end_time_col,
    )
    d = d.loc[d["end_time"] > d["start_time"]].copy()
    d = d.sort_values([id_col, "start_time"]).reset_index(drop=True)

    prev_x = np.asarray([point.x for point in d[previous_col]], dtype=float)
    prev_y = np.asarray([point.y for point in d[previous_col]], dtype=float)
    start_x = d.geometry.x.to_numpy(dtype=float)
    start_y = d.geometry.y.to_numpy(dtype=float)
    end_x = np.asarray([point.x for point in d["next_position"]], dtype=float)
    end_y = np.asarray([point.y for point in d["next_position"]], dtype=float)

    incoming = np.arctan2(start_y - prev_y, start_x - prev_x)
    heading = np.arctan2(end_y - start_y, end_x - start_x)
    step_length = np.hypot(end_x - start_x, end_y - start_y)
    turn_angle = wrap_angle(heading - incoming)

    observed = gpd.GeoDataFrame(
        {
            id_col: d[id_col].to_numpy(),
            "start_time": d["start_time"].to_numpy(),
            "end_time": d["end_time"].to_numpy(),
            "dt_h": (
                (d["end_time"] - d["start_time"])
                .dt.total_seconds()
                .to_numpy()
                / 3600.0
            ),
            "start_x": start_x,
            "start_y": start_y,
            "start_geometry": d.geometry.to_numpy(),
            "step_length": step_length,
            "incoming_heading": incoming,
            "heading": heading,
            "turn_angle": turn_angle,
            "used": np.ones(len(d), dtype=np.int8),
            "candidate_id": np.zeros(len(d), dtype=np.int32),
        },
        geometry=gpd.GeoSeries(
            d["next_position"].to_numpy(),
            crs=source.crs,
        ),
        crs=source.crs,
    )
    observed[stratum_col] = np.arange(len(observed), dtype=np.int64)

    gap = pd.Timedelta(burst_gap)
    previous_start = observed.groupby(id_col, sort=False)["start_time"].shift()
    new_burst = previous_start.isna() | (
        (observed["start_time"] - previous_start) > gap
    )
    observed["burst_id"] = (
        new_burst.groupby(observed[id_col], sort=False)
        .cumsum()
        .astype("int64")
        - 1
    )
    return observed


def draw_step_lengths(
    distribution: str,
    params,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw step lengths using the package movement-family parameterization."""
    if distribution == "exp":
        (scale,) = params
        return rng.exponential(scale=scale, size=size)
    if distribution == "gamma":
        shape, scale = params
        return rng.gamma(shape=shape, scale=scale, size=size)
    if distribution == "weibull":
        shape, scale = params
        return scale * rng.weibull(shape, size=size)
    if distribution == "lognorm":
        sigma, scale = params
        return rng.lognormal(mean=np.log(scale), sigma=sigma, size=size)
    raise ValueError(f"Unsupported step distribution: {distribution!r}")


def draw_turn_angles(
    distribution: str,
    params,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw turning angles using the package movement-family parameterization."""
    if distribution == "vonmises":
        kappa, mu = params
        return rng.vonmises(mu=mu, kappa=kappa, size=size)
    if distribution == "vm_uniform":
        kappa, weight = params
        use_vm = rng.random(size) < weight
        out = rng.uniform(-np.pi, np.pi, size=size)
        n_vm = int(use_vm.sum())
        if n_vm:
            out[use_vm] = rng.vonmises(mu=0.0, kappa=kappa, size=n_vm)
        return out
    raise ValueError(f"Unsupported turn-angle distribution: {distribution!r}")


def step_cdf(values, distribution: str, params) -> np.ndarray:
    """CDF of one fitted step-length family."""
    values = np.asarray(values, dtype=float)
    if distribution == "exp":
        (scale,) = params
        return stats.expon.cdf(values, scale=scale)
    if distribution == "gamma":
        shape, scale = params
        return stats.gamma.cdf(values, a=shape, scale=scale)
    if distribution == "weibull":
        shape, scale = params
        return stats.weibull_min.cdf(values, c=shape, scale=scale)
    if distribution == "lognorm":
        sigma, scale = params
        return stats.lognorm.cdf(values, s=sigma, scale=scale)
    raise ValueError(f"Unsupported step distribution: {distribution!r}")


def step_ppf(probabilities, distribution: str, params) -> np.ndarray:
    """Quantile function of one fitted step-length family."""
    probabilities = np.asarray(probabilities, dtype=float)
    if distribution == "exp":
        (scale,) = params
        return stats.expon.ppf(probabilities, scale=scale)
    if distribution == "gamma":
        shape, scale = params
        return stats.gamma.ppf(probabilities, a=shape, scale=scale)
    if distribution == "weibull":
        shape, scale = params
        return stats.weibull_min.ppf(probabilities, c=shape, scale=scale)
    if distribution == "lognorm":
        sigma, scale = params
        return stats.lognorm.ppf(probabilities, s=sigma, scale=scale)
    raise ValueError(f"Unsupported step distribution: {distribution!r}")


def step_logpdf(values, distribution: str, params) -> np.ndarray:
    """Log-density of one fitted step-length family."""
    values = np.asarray(values, dtype=float)
    if distribution == "exp":
        (scale,) = params
        return stats.expon.logpdf(values, scale=scale)
    if distribution == "gamma":
        shape, scale = params
        return stats.gamma.logpdf(values, a=shape, scale=scale)
    if distribution == "weibull":
        shape, scale = params
        return stats.weibull_min.logpdf(values, c=shape, scale=scale)
    if distribution == "lognorm":
        sigma, scale = params
        return stats.lognorm.logpdf(values, s=sigma, scale=scale)
    raise ValueError(f"Unsupported step distribution: {distribution!r}")


def angle_logpdf(values, distribution: str, params) -> np.ndarray:
    """Log-density of one fitted turn-angle family."""
    values = np.asarray(values, dtype=float)
    if distribution == "vonmises":
        kappa, mu = params
        return stats.vonmises.logpdf(values, kappa=kappa, loc=mu)
    if distribution == "vm_uniform":
        kappa, weight = params
        log_vm = np.log(weight) + stats.vonmises.logpdf(
            values,
            kappa=kappa,
            loc=0.0,
        )
        log_uniform = np.log1p(-weight) - np.log(2 * np.pi)
        return np.logaddexp(log_vm, log_uniform)
    raise ValueError(f"Unsupported turn-angle distribution: {distribution!r}")


def draw_truncated_step_lengths(
    distribution: str,
    params,
    max_length,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw exactly from a step distribution truncated at row-specific maxima."""
    max_length = np.asarray(max_length, dtype=float)
    if np.any(~np.isfinite(max_length)) or np.any(max_length <= 0):
        raise ValueError("max_length must be finite and strictly positive.")

    fmax = np.clip(
        step_cdf(max_length, distribution, params),
        np.finfo(float).tiny,
        1.0,
    )
    u = rng.random(max_length.shape) * fmax
    return step_ppf(u, distribution, params)


def truncated_step_logpdf(
    values,
    distribution: str,
    params,
    max_length,
) -> np.ndarray:
    """Log-density matching the exact truncated step proposal."""
    values = np.asarray(values, dtype=float)
    max_length = np.asarray(max_length, dtype=float)
    fmax = np.clip(
        step_cdf(max_length, distribution, params),
        np.finfo(float).tiny,
        1.0,
    )
    out = step_logpdf(values, distribution, params) - np.log(fmax)
    return np.where(
        (values >= 0) & (values <= max_length),
        out,
        -np.inf,
    )


def movement_speed_caps(
    movement: Mapping[str, Any],
    *,
    id_col: str = "Individual_ID",
    margin: float = 1.05,
) -> pd.Series:
    """Return per-individual maximum observed speed multiplied by a margin."""
    if margin < 1:
        raise ValueError("margin must be at least 1.")

    step_df = movement.get("step_df")
    if step_df is None:
        raise KeyError("movement must contain 'step_df'.")
    if id_col not in step_df or "speed_kmh" not in step_df:
        raise KeyError(
            f"movement['step_df'] must contain {id_col!r} and 'speed_kmh'."
        )

    caps = (
        step_df.groupby(id_col, sort=False)["speed_kmh"].max()
        * float(margin)
    )
    caps.name = "max_speed_kmh"
    return caps


def sample_available_steps(
    observed_steps: gpd.GeoDataFrame,
    movement_summary: pd.DataFrame,
    *,
    id_col: str = "Individual_ID",
    stratum_col: str = "stratum_id",
    n_available: int = 20,
    max_speed_kmh: Mapping[Any, float] | pd.Series | None = None,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Append movement-informed alternatives to each observed SSF step."""
    if n_available <= 0:
        raise ValueError("n_available must be positive.")
    if observed_steps.crs is None:
        raise ValueError("observed_steps.crs is None.")

    required = (
        id_col,
        stratum_col,
        "start_x",
        "start_y",
        "start_geometry",
        "incoming_heading",
        "dt_h",
        "step_length",
        "turn_angle",
    )
    missing = [col for col in required if col not in observed_steps]
    if missing:
        raise KeyError(f"Missing observed-step columns: {missing}")

    summary = movement_summary.copy()
    summary_id_col = id_col if id_col in summary.columns else "id"
    if summary_id_col not in summary:
        raise KeyError(
            f"Movement summary must contain {id_col!r} or the canonical 'id' column."
        )
    summary = summary.set_index(summary_id_col, drop=False)

    caps = None
    if max_speed_kmh is not None:
        caps = pd.Series(max_speed_kmh, dtype=float)

    rng = np.random.default_rng(seed)
    frames: list[gpd.GeoDataFrame] = []
    observed = observed_steps.copy()
    if "used" not in observed:
        observed["used"] = 1
    if "candidate_id" not in observed:
        observed["candidate_id"] = 0

    for animal_id, obs_i in observed.groupby(id_col, sort=False):
        if animal_id not in summary.index:
            raise KeyError(f"No movement-kernel summary for {animal_id!r}.")

        kernel = summary.loc[animal_id]
        if isinstance(kernel, pd.DataFrame):
            if len(kernel) != 1:
                raise ValueError(
                    f"Movement summary has duplicate rows for {animal_id!r}."
                )
            kernel = kernel.iloc[0]

        step_distribution = kernel["step_distribution"]
        step_params = kernel["step_params"]
        angle_distribution = kernel["angle_distribution"]
        angle_params = kernel["angle_params"]

        obs_i = obs_i.copy()
        if caps is not None:
            if animal_id not in caps.index:
                raise KeyError(f"No max-speed cap supplied for {animal_id!r}.")
            vmax = float(caps.loc[animal_id])
            max_step = (
                vmax
                * 1000.0
                * obs_i["dt_h"].to_numpy(dtype=float)
            )
            observed_length = obs_i["step_length"].to_numpy(dtype=float)
            if np.any(observed_length > max_step + 1e-8):
                raise ValueError(
                    f"Observed steps for {animal_id!r} exceed the supplied speed cap."
                )
            obs_i["max_speed_kmh"] = vmax
            obs_i["max_step_length"] = max_step
            obs_i["proposal_step_logpdf"] = truncated_step_logpdf(
                observed_length,
                step_distribution,
                step_params,
                max_step,
            )
        else:
            obs_i["max_speed_kmh"] = np.nan
            obs_i["max_step_length"] = np.inf
            obs_i["proposal_step_logpdf"] = step_logpdf(
                obs_i["step_length"],
                step_distribution,
                step_params,
            )

        obs_i["proposal_angle_logpdf"] = angle_logpdf(
            obs_i["turn_angle"],
            angle_distribution,
            angle_params,
        )
        obs_i["proposal_logpdf"] = (
            obs_i["proposal_step_logpdf"]
            + obs_i["proposal_angle_logpdf"]
        )
        obs_i["step_distribution"] = step_distribution
        obs_i["angle_distribution"] = angle_distribution
        frames.append(obs_i)

        n_obs = len(obs_i)
        repeated = (
            obs_i.loc[obs_i.index.repeat(n_available)]
            .copy()
            .reset_index(drop=True)
        )
        repeated["candidate_id"] = np.tile(
            np.arange(1, n_available + 1, dtype=np.int32),
            n_obs,
        )
        repeated["used"] = np.int8(0)

        if caps is None:
            lengths = draw_step_lengths(
                step_distribution,
                step_params,
                len(repeated),
                rng,
            )
            repeated["proposal_step_logpdf"] = step_logpdf(
                lengths,
                step_distribution,
                step_params,
            )
        else:
            vmax = float(caps.loc[animal_id])
            max_length = (
                vmax
                * 1000.0
                * repeated["dt_h"].to_numpy(dtype=float)
            )
            lengths = draw_truncated_step_lengths(
                step_distribution,
                step_params,
                max_length,
                rng=rng,
            )
            repeated["proposal_step_logpdf"] = truncated_step_logpdf(
                lengths,
                step_distribution,
                step_params,
                max_length,
            )
            repeated["max_speed_kmh"] = vmax
            repeated["max_step_length"] = max_length

        turns = draw_turn_angles(
            angle_distribution,
            angle_params,
            len(repeated),
            rng,
        )
        headings = wrap_angle(
            repeated["incoming_heading"].to_numpy(dtype=float)
            + turns
        )
        x = (
            repeated["start_x"].to_numpy(dtype=float)
            + lengths * np.cos(headings)
        )
        y = (
            repeated["start_y"].to_numpy(dtype=float)
            + lengths * np.sin(headings)
        )

        repeated["step_length"] = lengths
        repeated["turn_angle"] = turns
        repeated["heading"] = headings
        repeated["proposal_angle_logpdf"] = angle_logpdf(
            turns,
            angle_distribution,
            angle_params,
        )
        repeated["proposal_logpdf"] = (
            repeated["proposal_step_logpdf"]
            + repeated["proposal_angle_logpdf"]
        )
        repeated["step_distribution"] = step_distribution
        repeated["angle_distribution"] = angle_distribution
        repeated = gpd.GeoDataFrame(
            repeated,
            geometry=gpd.GeoSeries(
                [Point(xi, yi) for xi, yi in zip(x, y)],
                crs=observed.crs,
            ),
            crs=observed.crs,
        )
        frames.append(repeated)

    out = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=observed.crs,
    )
    out = out.sort_values([stratum_col, "candidate_id"]).reset_index(drop=True)

    check = out.groupby(stratum_col, sort=False)["used"].agg(
        n_choices="size",
        n_used="sum",
    )
    if not check["n_choices"].eq(n_available + 1).all():
        raise RuntimeError("Generated SSF strata do not have the requested choice-set size.")
    if not check["n_used"].eq(1).all():
        raise RuntimeError("Generated SSF strata do not contain exactly one used choice.")
    return out


def build_movement_choice_sets(
    movement: Mapping[str, Any],
    *,
    id_col: str = "Individual_ID",
    n_available: int = 20,
    speed_margin: float = 1.05,
    stratum_col: str = "stratum_id",
    burst_gap: str | pd.Timedelta = "2h",
    seed: int = 42,
) -> gpd.GeoDataFrame:
    """Build observed and movement-informed available alternatives end to end."""
    observed = build_observed_ssf_steps(
        movement,
        id_col=id_col,
        stratum_col=stratum_col,
        burst_gap=burst_gap,
    )
    caps = movement_speed_caps(
        movement,
        id_col=id_col,
        margin=speed_margin,
    )
    return sample_available_steps(
        observed,
        movement["summary"],
        id_col=id_col,
        stratum_col=stratum_col,
        n_available=n_available,
        max_speed_kmh=caps,
        seed=seed,
    )


__all__ = [
    "wrap_angle",
    "build_observed_ssf_steps",
    "draw_step_lengths",
    "draw_turn_angles",
    "step_cdf",
    "step_ppf",
    "step_logpdf",
    "angle_logpdf",
    "draw_truncated_step_lengths",
    "truncated_step_logpdf",
    "movement_speed_caps",
    "sample_available_steps",
    "build_movement_choice_sets",
]
