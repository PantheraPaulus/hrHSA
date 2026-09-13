from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative specification of an RSF/SSF design matrix.

    ``linear`` contains continuous predictors that are standardised before model
    fitting. ``quadratic`` and ``interactions`` are defined on the standardised
    scale. ``categorical`` predictors are dummy-encoded with reproducible
    metadata so that fitted models can later be projected back to raster stacks.
    """

    linear: list[str] = field(default_factory=list)
    quadratic: list[str] = field(default_factory=list)
    interactions: list[tuple[str, str]] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    add_const: bool = True


@dataclass
class FoldSplit:
    """Container returned by temporal or spatial cross-validation splitters."""

    train: "gpd.GeoDataFrame"
    test: "gpd.GeoDataFrame"
    train_thin: "gpd.GeoDataFrame"
    test_thin: "gpd.GeoDataFrame"
