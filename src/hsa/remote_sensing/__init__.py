"""Remote-sensing predictor builders.

This module hosts reusable Earth Engine/xarray utilities. Project-specific
stacks, such as the current Okonjima pangolin stack, should first live in
examples and only move here once they are generalized.
"""

from hsa.remote_sensing.earthengine import (
    ee_image_to_xarray_stack,
    ee_samples_to_gdf,
    initialize_earth_engine,
    spatial_summary,
    temporal_summary,
)

__all__ = [
    "initialize_earth_engine",
    "ee_image_to_xarray_stack",
    "ee_samples_to_gdf",
    "spatial_summary",
    "temporal_summary",
]
