"""Exact cross-validation kernels for frequentist and Bayesian SSFs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hsa._time import require_timezone_aware

from hsa.ssf.bayesian import (
    build_hierarchical_ssf_model,
    posterior_choice_probabilities_known_individual,
    posterior_choice_probabilities_new_individual,
)
from hsa.ssf.data import (
    apply_ssf_scaling,
    build_ssf_choice_arrays,
    complete_ssf_strata,
    fit_ssf_scaling,
    score_choice_probabilities,
    stable_softmax,
)
from hsa.ssf.frequentist import fit_conditional_ssf
from hsa.ssf.schemes import SSFValidationResult


def _complete_choices(analysis) -> pd.DataFrame:
    """Apply the same whole-stratum completeness rule used by fit()."""
    analysis.validate_predictors()
    return complete_ssf_strata(
        analysis.choices,
        predictors=analysis.predictors,
        id_col=analysis.id_col,
        expected_n_choices=analysis.n_available + 1,
    )


def _sample_strata(df, *, id_col, n_per_id, seed):
    """Subsample whole choice sets independently within each individual."""
    if n_per_id is None:
        return df.copy()
    keys = df[[id_col, "stratum_id"]].drop_duplicates()
    parts = []
    for i, (_, group) in enumerate(keys.groupby(id_col, sort=False)):
        n = min(n_per_id, len(group))
        parts.append(group.sample(n=n, random_state=seed + i))
    selected = pd.concat(parts, ignore_index=True)
    return df.merge(
        selected,
        on=[id_col, "stratum_id"],
        how="inner",
        validate="many_to_one",
    )


def make_temporal_block_split(
    df,
    *,
    id_col,
    time_col="start_time",
    n_blocks=5,
    holdout_block=2,
    embargo="24h",
    n_train_per_id=None,
    n_test_per_id=None,
    seed=42,
):
    """Create one contiguous chronological holdout per individual.

    Training may use periods before and after the withheld block. Therefore this
    scheme measures temporal stability/interpolation rather than forecasting.
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2.")
    if not 0 <= holdout_block < n_blocks:
        raise ValueError("holdout_block must index one of n_blocks.")

    meta = df[[id_col, "stratum_id", time_col]].drop_duplicates().copy()
    time_counts = df.groupby([id_col, "stratum_id"], sort=False)[time_col].nunique()
    if not time_counts.eq(1).all():
        raise ValueError("Every stratum must have exactly one start time.")
    meta[time_col] = require_timezone_aware(meta[time_col], name=time_col)

    embargo_td = pd.Timedelta(embargo)
    if embargo_td < pd.Timedelta(0):
        raise ValueError("embargo cannot be negative.")

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    summaries = []

    for animal, group in meta.groupby(id_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)
        if len(group) < n_blocks:
            raise ValueError(
                f"{animal!r} has only {len(group)} strata for {n_blocks} blocks."
            )

        blocks = np.empty(len(group), dtype="int16")
        for block, positions in enumerate(
            np.array_split(np.arange(len(group)), n_blocks)
        ):
            blocks[positions] = block
        group["_block"] = blocks

        heldout = group.loc[group["_block"] == holdout_block].copy()
        start = heldout[time_col].min()
        end = heldout[time_col].max()
        train_candidates = group.loc[
            (group[time_col] < start - embargo_td)
            | (group[time_col] > end + embargo_td)
        ].copy()

        if n_train_per_id is not None and len(train_candidates) > n_train_per_id:
            positions = np.sort(
                rng.choice(len(train_candidates), n_train_per_id, replace=False)
            )
            train_selected = train_candidates.iloc[positions].copy()
        else:
            train_selected = train_candidates

        if n_test_per_id is not None and len(heldout) > n_test_per_id:
            first = (len(heldout) - n_test_per_id) // 2
            test_selected = heldout.iloc[first : first + n_test_per_id].copy()
        else:
            test_selected = heldout

        train_parts.append(train_selected)
        test_parts.append(test_selected)
        summaries.append(
            {
                id_col: animal,
                "n_total_strata": len(group),
                "n_heldout_block": len(heldout),
                "n_train_candidates_after_embargo": len(train_candidates),
                "n_train": len(train_selected),
                "n_test": len(test_selected),
                "holdout_start": start,
                "holdout_end": end,
                "embargo": embargo_td,
            }
        )

    train_keys = pd.concat(train_parts, ignore_index=True)
    test_keys = pd.concat(test_parts, ignore_index=True)
    key_cols = [id_col, "stratum_id"]
    train = df.merge(
        train_keys[key_cols],
        on=key_cols,
        how="inner",
        validate="many_to_one",
    )
    test = df.merge(
        test_keys[key_cols],
        on=key_cols,
        how="inner",
        validate="many_to_one",
    )
    return {
        "train": train,
        "test": test,
        "train_keys": train_keys,
        "test_keys": test_keys,
        "summary": pd.DataFrame(summaries),
    }


def _score_frequentist_test(test, result, predictors, *, id_col):
    """Score an already standardized held-out conditional-choice table."""
    rows = []
    beta = result.params.to_numpy(dtype=float)
    grouped = test.groupby([id_col, "stratum_id"], sort=False)
    for (animal, stratum), group in grouped:
        eta = group[list(predictors)].to_numpy(dtype=float) @ beta
        probability = stable_softmax(eta)
        chosen = np.array([int(np.argmax(group["used"].to_numpy()))])
        per, _ = score_choice_probabilities(probability[None, :], chosen)
        row = per.iloc[0].to_dict()
        row[id_col] = animal
        row["stratum_id"] = stratum
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_by_id(per, *, id_col):
    """Summarize held-out choice scoring separately for each animal."""
    rows = []
    for animal, group in per.groupby(id_col, sort=False):
        gain = group["log_score_gain"].to_numpy(dtype=float)
        rank = group["choice_rank"].to_numpy(dtype=float)
        rows.append(
            {
                id_col: animal,
                "n_strata": len(group),
                "mean_chosen_probability": float(
                    group["chosen_probability"].mean()
                ),
                "median_chosen_probability": float(
                    group["chosen_probability"].median()
                ),
                "top_1": float(np.mean(rank == 1)),
                "top_5": float(np.mean(rank <= 5)),
                "mean_log_score_gain": float(gain.mean()),
                "median_log_score_gain": float(np.median(gain)),
                "fraction_gain_positive": float(np.mean(gain > 0)),
                "predictive_advantage": float(np.exp(gain.mean())),
            }
        )
    return pd.DataFrame(rows)


def _bayesian_sampling(defaults, sample_kwargs, *, seed):
    sampling = dict(defaults)
    if sample_kwargs:
        sampling.update(dict(sample_kwargs))
    sampling.setdefault("random_seed", seed)
    sampling.setdefault("return_inferencedata", True)
    return sampling


def _divergence_count(idata) -> int:
    if "diverging" not in idata.sample_stats:
        return 0
    return int(np.asarray(idata.sample_stats["diverging"]).sum())


def validate_loio(
    analysis,
    scheme,
    *,
    sample_kwargs=None,
    modes=("population_mean", "new_individual"),
):
    """Exact leave-one-individual-out validation with training-only scaling."""
    from hsa.ssf.bayesian import BayesianSSF
    from hsa.ssf.frequentist import FrequentistSSF

    choices = _complete_choices(analysis)
    ids = analysis.individuals if scheme.heldout == "all" else [scheme.heldout]
    unknown = [animal for animal in ids if animal not in analysis.individuals]
    if unknown:
        raise KeyError(f"Unknown held-out individual(s): {unknown}")

    summaries = []
    diagnostics = {}

    for heldout in ids:
        train = choices.loc[choices[analysis.id_col] != heldout].copy()
        test = choices.loc[choices[analysis.id_col] == heldout].copy()
        train = _sample_strata(
            train,
            id_col=analysis.id_col,
            n_per_id=scheme.n_train_per_id,
            seed=scheme.seed,
        )
        test = _sample_strata(
            test,
            id_col=analysis.id_col,
            n_per_id=scheme.n_test_per_id,
            seed=scheme.seed,
        )

        scaling = fit_ssf_scaling(train, analysis.predictors)
        train_z = apply_ssf_scaling(train, analysis.predictors, scaling)
        test_z = apply_ssf_scaling(test, analysis.predictors, scaling)
        predictors = tuple(f"{predictor}_z" for predictor in analysis.predictors)

        if isinstance(analysis, FrequentistSSF):
            model, result = fit_conditional_ssf(
                train_z,
                predictors=predictors,
                id_col=analysis.id_col,
            )
            per = _score_frequentist_test(
                test_z,
                result,
                predictors,
                id_col=analysis.id_col,
            )
            summary = _summary_by_id(per, id_col=analysis.id_col).iloc[0].to_dict()
            summary["heldout_id"] = heldout
            summaries.append(summary)
            diagnostics[heldout] = {
                "model": model,
                "result": result,
                "scaling": scaling,
                "per_stratum": per,
            }
            continue

        if not isinstance(analysis, BayesianSSF):
            raise TypeError("LOIO currently supports FrequentistSSF and BayesianSSF.")

        train_arrays = build_ssf_choice_arrays(
            train_z,
            id_col=analysis.id_col,
            predictors=predictors,
        )
        test_arrays = build_ssf_choice_arrays(
            test_z,
            id_col=analysis.id_col,
            predictors=predictors,
        )
        model = build_hierarchical_ssf_model(
            train_arrays,
            **analysis.model_kwargs,
        )
        sampling = _bayesian_sampling(
            {
                "draws": 600,
                "tune": 800,
                "chains": 4,
                "target_accept": 0.95,
            },
            sample_kwargs,
            seed=scheme.seed,
        )
        import pymc as pm

        with model:
            idata = pm.sample(**sampling)

        fold_diag = {
            "model": model,
            "idata": idata,
            "scaling": scaling,
            "divergences": _divergence_count(idata),
            "scores": {},
        }
        for mode in modes:
            probability = posterior_choice_probabilities_new_individual(
                idata,
                test_arrays.X,
                mode=mode,
                random_seed=scheme.seed,
            )
            per, score = score_choice_probabilities(
                probability,
                test_arrays.chosen,
            )
            per[[analysis.id_col, "stratum_id"]] = test_arrays.strata[
                [analysis.id_col, "stratum_id"]
            ].to_numpy()
            row = score.to_dict()
            row.update(
                {
                    "heldout_id": heldout,
                    "mode": mode,
                    "divergences": fold_diag["divergences"],
                }
            )
            summaries.append(row)
            fold_diag["scores"][mode] = {
                "summary": score,
                "per_stratum": per,
            }
        diagnostics[heldout] = fold_diag

    return SSFValidationResult(
        summary=pd.DataFrame(summaries),
        diagnostics=diagnostics,
        scheme=scheme,
        analysis=analysis,
    )


def validate_temporal_block(analysis, scheme, *, sample_kwargs=None):
    """Exact temporally blocked validation with an embargo around the holdout."""
    from hsa.ssf.bayesian import BayesianSSF
    from hsa.ssf.frequentist import FrequentistSSF

    choices = _complete_choices(analysis)
    split = make_temporal_block_split(
        choices,
        id_col=analysis.id_col,
        n_blocks=scheme.n_blocks,
        holdout_block=scheme.holdout_block,
        embargo=scheme.embargo,
        n_train_per_id=scheme.n_train_per_id,
        n_test_per_id=scheme.n_test_per_id,
        seed=scheme.seed,
    )
    scaling = fit_ssf_scaling(split["train"], analysis.predictors)
    train_z = apply_ssf_scaling(split["train"], analysis.predictors, scaling)
    test_z = apply_ssf_scaling(split["test"], analysis.predictors, scaling)
    predictors = tuple(f"{predictor}_z" for predictor in analysis.predictors)

    if isinstance(analysis, FrequentistSSF):
        model, result = fit_conditional_ssf(
            train_z,
            predictors=predictors,
            id_col=analysis.id_col,
        )
        per = _score_frequentist_test(
            test_z,
            result,
            predictors,
            id_col=analysis.id_col,
        )
        return SSFValidationResult(
            summary=_summary_by_id(per, id_col=analysis.id_col),
            diagnostics={
                "split": split,
                "model": model,
                "result": result,
                "scaling": scaling,
                "per_stratum": per,
            },
            scheme=scheme,
            analysis=analysis,
        )

    if not isinstance(analysis, BayesianSSF):
        raise TypeError(
            "TemporalBlockCV currently supports FrequentistSSF and BayesianSSF."
        )

    train_arrays = build_ssf_choice_arrays(
        train_z,
        id_col=analysis.id_col,
        predictors=predictors,
    )
    test_arrays = build_ssf_choice_arrays(
        test_z,
        id_col=analysis.id_col,
        predictors=predictors,
    )
    model = build_hierarchical_ssf_model(
        train_arrays,
        **analysis.model_kwargs,
    )
    sampling = _bayesian_sampling(
        {
            "draws": 800,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.95,
        },
        sample_kwargs,
        seed=scheme.seed,
    )
    import pymc as pm

    with model:
        idata = pm.sample(**sampling)

    lookup = {
        animal: i
        for i, animal in enumerate(train_arrays.individuals)
    }
    test_ids = test_arrays.strata[analysis.id_col]
    if not test_ids.isin(lookup).all():
        missing = test_ids.loc[~test_ids.isin(lookup)].unique().tolist()
        raise ValueError(
            "Temporal validation requires every held-out individual to have "
            f"training data. Missing from training: {missing}"
        )
    test_idx = test_ids.map(lookup).to_numpy(dtype="int32")
    remapped = type(test_arrays)(
        X=test_arrays.X,
        chosen=test_arrays.chosen,
        individual_idx=test_idx,
        individuals=train_arrays.individuals,
        strata=test_arrays.strata,
        predictors=test_arrays.predictors,
        n_choices=test_arrays.n_choices,
    )
    probability = posterior_choice_probabilities_known_individual(
        idata,
        remapped,
    )
    per, _ = score_choice_probabilities(
        probability,
        remapped.chosen,
    )
    per[[analysis.id_col, "stratum_id"]] = remapped.strata[
        [analysis.id_col, "stratum_id"]
    ].to_numpy()
    divergences = _divergence_count(idata)
    summary = _summary_by_id(per, id_col=analysis.id_col)
    summary["divergences"] = divergences

    return SSFValidationResult(
        summary=summary,
        diagnostics={
            "split": split,
            "model": model,
            "idata": idata,
            "divergences": divergences,
            "per_stratum": per,
            "scaling": scaling,
        },
        scheme=scheme,
        analysis=analysis,
    )


__all__ = [
    "make_temporal_block_split",
    "validate_loio",
    "validate_temporal_block",
]
