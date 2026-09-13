"""Habitat selection and movement modelling tools.

The package is organised around a reusable workflow:

1. prepare telemetry data,
2. define availability,
3. sample environmental predictors,
4. build RSF/SSF design matrices,
5. fit and validate models,
6. predict habitat-selection surfaces,
7. optionally simulate movement through those surfaces.
"""

from hsa.bayesian_sampling import configure_pymc_sampling
from hsa.types import FeatureSpec, FoldSplit

__all__ = ["FeatureSpec", "FoldSplit", "configure_pymc_sampling"]
