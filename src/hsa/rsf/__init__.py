"""Resource-selection-function tools.

The object-oriented workflow is the recommended public interface. The functional
kernels remain public for backwards compatibility, advanced use and testing.
"""

from hsa.rsf.base import RSFAnalysis, RSFFit
from hsa.rsf.bayesian import (
    build_bayesian_rsf_model,
    evaluate_bayesian_rsf,
    plot_bayesian_rsf_diagnostics,
    predict_bayesian_rsf_surface,
    prepare_bayesian_rsf_data,
)
from hsa.rsf.bayesian_cv import leave_one_individual_out_bayesian_rsf
from hsa.rsf.bayesian_shrinkage import (
    collinearity_diagnostics,
    posterior_beta_correlation,
    predictor_correlation,
    regularized_horseshoe_summary,
)
from hsa.rsf.bayesian_validation import (
    bayesian_boyce_quantile_scores,
    evaluate_bayesian_loio_uncertainty,
    plot_bayesian_boyce,
    plot_bayesian_loio_uncertainty,
    prepare_bayesian_boyce_scores,
)
from hsa.rsf.bayesian_workflow import BayesianRSF, BayesianRSFFit
from hsa.rsf.frequentist import FrequentistRSF, FrequentistRSFFit
from hsa.rsf.frequentist_validation import (
    evaluate_frequentist_loio_uncertainty,
    plot_frequentist_loio_uncertainty,
)
from hsa.rsf.hpc import leave_one_individual_out_rsf_prepared
from hsa.rsf.model import (
    PreparedRSFDesign,
    fit_prepared_rsf,
    fit_rsf,
    predict_rsf_points,
    prepare_rsf_design,
)
from hsa.rsf.schemes import (
    BayesianLOIOResult,
    BayesianValidationUncertaintyResult,
    BlockedBootstrap,
    ContiguousTemporalBlocks,
    CrossValidationResult,
    FrequentistLOIOResult,
    FrequentistValidationUncertaintyResult,
    LeaveOneIndividualOut,
    ValidationUncertaintyResult,
)
from hsa.rsf.selection import (
    add_aic_weights,
    compare_rsf_specs,
    compare_single_predictors,
    evaluate_linear_candidates_up_to_k,
    evaluate_regex_candidate_family,
    has_duplicate_base_variables,
    select_best_scale_per_predictor,
    select_predictor_columns,
    split_multiscale_name,
    summarize_univariate_scale_selection,
    variable_frequency_summary,
)
from hsa.rsf.surface import predict_rsf_surface, predict_rsf_surface_multiscale
from hsa.rsf.surface_fast import predict_rsf_surface_chunked
from hsa.rsf.validation import (
    boyce_quantile_bins,
    boyce_sliding_window,
    plot_boyce_curves,
    plot_boyce_values,
)

__all__ = [
    # Recommended object-oriented workflow.
    "RSFAnalysis",
    "RSFFit",
    "FrequentistRSF",
    "FrequentistRSFFit",
    "BayesianRSF",
    "BayesianRSFFit",
    "LeaveOneIndividualOut",
    "BlockedBootstrap",
    "ContiguousTemporalBlocks",
    "CrossValidationResult",
    "ValidationUncertaintyResult",
    "FrequentistLOIOResult",
    "BayesianLOIOResult",
    "FrequentistValidationUncertaintyResult",
    "BayesianValidationUncertaintyResult",
    # Functional kernels kept public and backwards compatible.
    "PreparedRSFDesign",
    "prepare_rsf_design",
    "fit_prepared_rsf",
    "fit_rsf",
    "predict_rsf_points",
    "predict_rsf_surface",
    "predict_rsf_surface_chunked",
    "predict_rsf_surface_multiscale",
    "leave_one_individual_out_rsf_prepared",
    "boyce_quantile_bins",
    "boyce_sliding_window",
    "plot_boyce_curves",
    "plot_boyce_values",
    "prepare_bayesian_rsf_data",
    "build_bayesian_rsf_model",
    "evaluate_bayesian_rsf",
    "plot_bayesian_rsf_diagnostics",
    "predict_bayesian_rsf_surface",
    "posterior_beta_correlation",
    "predictor_correlation",
    "collinearity_diagnostics",
    "regularized_horseshoe_summary",
    "prepare_bayesian_boyce_scores",
    "bayesian_boyce_quantile_scores",
    "plot_bayesian_boyce",
    "leave_one_individual_out_bayesian_rsf",
    "evaluate_bayesian_loio_uncertainty",
    "plot_bayesian_loio_uncertainty",
    "evaluate_frequentist_loio_uncertainty",
    "plot_frequentist_loio_uncertainty",
    "add_aic_weights",
    "compare_rsf_specs",
    "compare_single_predictors",
    "evaluate_linear_candidates_up_to_k",
    "evaluate_regex_candidate_family",
    "has_duplicate_base_variables",
    "select_best_scale_per_predictor",
    "select_predictor_columns",
    "split_multiscale_name",
    "summarize_univariate_scale_selection",
    "variable_frequency_summary",
]
