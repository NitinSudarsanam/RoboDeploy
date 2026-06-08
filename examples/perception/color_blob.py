"""RGB color-blob centroid → rough 3D object estimate in ``obs.objects``."""

from __future__ import annotations

from robodeploy.perception.vision_predicates import (
    ColorBlobTrackerTransform,
    _camera_to_world,
    _quat_rotate_wxyz,
)

# Backward-compatible alias used by examples and tests.
ColorBlobCentroidTransform = ColorBlobTrackerTransform

__all__ = [
    "ColorBlobCentroidTransform",
    "ColorBlobTrackerTransform",
    "_camera_to_world",
    "_quat_rotate_wxyz",
]
