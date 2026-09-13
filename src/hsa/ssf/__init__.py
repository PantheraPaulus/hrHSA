"""Step-selection functions and stateful SSF/iSSF workflows."""

from hsa.ssf.base import SSFAnalysis, SSFFit
from hsa.ssf.bayesian import BayesianSSF, BayesianSSFFit, build_hierarchical_ssf_model
from hsa.ssf.bayesian_diagnostics import (
    plot_bayesian_ssf_diagnostics,
    plot_bayesian_ssf_forests,
    plot_bayesian_ssf_trace,
)
from hsa.ssf.choice_sets import (
    angle_logpdf,
    build_movement_choice_sets,
    build_observed_ssf_steps,
    draw_truncated_step_lengths,
    movement_speed_caps,
    sample_available_steps,
    truncated_step_logpdf,
    wrap_angle,
)
from hsa.ssf.data import (
    SSFChoiceArrays,
    apply_ssf_scaling,
    build_ssf_choice_arrays,
    complete_ssf_strata,
    fit_ssf_scaling,
    score_choice_probabilities,
)
from hsa.ssf.diagnostics import (
    canonical_ciif,
    conditional_information,
    plot_selection_opportunity,
    summarize_selection_opportunity,
    within_stratum_correlation,
)
from hsa.ssf.dynamic_diagnostics import (
    extract_dynamic_condition_snapshots,
    plot_dynamic_condition_comparison,
    plot_dynamic_conditions,
    summarize_dynamic_conditions,
)
from hsa.ssf.environment import (
    add_movement_terms,
    add_vector_support_covariates,
    annotate_static_covariates,
    check_raster_coverage,
    sample_dynamic_covariates_at_points,
    sample_dynamic_vectors_at_points,
    sample_era5_at_points,
    sample_era5_wind_at_points,
    sample_static_covariates_batched,
)
from hsa.ssf.fast_conditional import (
    FastConditionalLogitModel,
    FastConditionalLogitResult,
    fit_fast_conditional_ssf,
)
from hsa.ssf.frequentist import (
    FrequentistSSF,
    FrequentistSSFFit,
    IndividualSSFFits,
    fit_conditional_ssf,
    fit_ssf_per_id,
)
from hsa.ssf.issf import (
    ISSFAnalysis,
    ISSFDesign,
    build_hierarchical_issf_model,
    issf_interaction_name,
    prepare_issf_design,
)
from hsa.ssf.parallel import fit_issf_per_id
from hsa.ssf.workflow import (
    BayesianISSF,
    BayesianISSFFit,
    FrequentistISSF,
    FrequentistISSFFit,
    ISSFModelSpec,
    IndividualISSFFits,
)
from hsa.ssf.schemes import LeaveOneIndividualOut, SSFValidationResult, TemporalBlockCV
from hsa.ssf.validation import make_temporal_block_split

__all__ = [
    "SSFAnalysis",
    "SSFFit",
    "FrequentistSSF",
    "FrequentistSSFFit",
    "IndividualSSFFits",
    "FastConditionalLogitModel",
    "FastConditionalLogitResult",
    "BayesianSSF",
    "BayesianSSFFit",
    "ISSFAnalysis",
    "ISSFDesign",
    "ISSFModelSpec",
    "FrequentistISSF",
    "FrequentistISSFFit",
    "IndividualISSFFits",
    "BayesianISSF",
    "BayesianISSFFit",
    "LeaveOneIndividualOut",
    "TemporalBlockCV",
    "SSFValidationResult",
    "SSFChoiceArrays",
    "build_observed_ssf_steps",
    "sample_available_steps",
    "build_movement_choice_sets",
    "movement_speed_caps",
    "draw_truncated_step_lengths",
    "truncated_step_logpdf",
    "angle_logpdf",
    "wrap_angle",
    "check_raster_coverage",
    "sample_static_covariates_batched",
    "annotate_static_covariates",
    "sample_dynamic_covariates_at_points",
    "sample_dynamic_vectors_at_points",
    "sample_era5_at_points",
    "sample_era5_wind_at_points",
    "extract_dynamic_condition_snapshots",
    "summarize_dynamic_conditions",
    "plot_dynamic_condition_comparison",
    "plot_dynamic_conditions",
    "add_vector_support_covariates",
    "add_movement_terms",
    "complete_ssf_strata",
    "fit_ssf_scaling",
    "apply_ssf_scaling",
    "build_ssf_choice_arrays",
    "score_choice_probabilities",
    "fit_fast_conditional_ssf",
    "fit_conditional_ssf",
    "fit_ssf_per_id",
    "fit_issf_per_id",
    "build_hierarchical_ssf_model",
    "prepare_issf_design",
    "issf_interaction_name",
    "build_hierarchical_issf_model",
    "plot_bayesian_ssf_trace",
    "plot_bayesian_ssf_forests",
    "plot_bayesian_ssf_diagnostics",
    "within_stratum_correlation",
    "conditional_information",
    "summarize_selection_opportunity",
    "canonical_ciif",
    "plot_selection_opportunity",
    "make_temporal_block_split",
]
