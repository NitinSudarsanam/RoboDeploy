"""Canonical Kuka pick-place layout (table, source cube, target).

Single source of truth for MuJoCo, RViz markers, and Gazebo SDF worlds.
Geom ``size`` values are half-extents (MuJoCo convention); Gazebo builders
scale boxes by 2 when emitting SDF.
"""

from __future__ import annotations

from robodeploy.core.types import SceneSpec
from robodeploy.scene_builder import SceneBuilder

# Table top surface z = TABLE_TOP_Z (cube center = top + CUBE_HALF_Z).
TABLE_TOP_Z = 0.38
TABLE_THICKNESS = 0.03
TABLE_HALF_THICKNESS = TABLE_THICKNESS / 2.0
TABLE_CENTER_Z = TABLE_TOP_Z - TABLE_HALF_THICKNESS
TABLE_SIZE_HALF = (0.45, 0.35, TABLE_HALF_THICKNESS)  # 0.9 m x 0.7 m tabletop
TABLE_CENTER = (0.55, 0.0, TABLE_CENTER_Z)

CUBE_HALF = 0.025  # 50 mm cube (MuJoCo half-extent)
SOURCE_POS = (0.55, 0.0, TABLE_TOP_Z + CUBE_HALF)
TARGET_POS = (0.60, 0.20, TABLE_TOP_Z + 0.04)  # sphere radius 0.04


def build_pick_place_scene() -> SceneSpec:
    """Build the standard pick-place scene for all backends."""
    return (
        SceneBuilder()
        .add_box(
            "table",
            size=TABLE_SIZE_HALF,
            pos=TABLE_CENTER,
            fixed=True,
            rgba=(0.55, 0.45, 0.35, 1.0),
        )
        .add_box(
            "source",
            size=(CUBE_HALF, CUBE_HALF, CUBE_HALF),
            pos=SOURCE_POS,
            mass=0.05,
            rgba=(1.0, 0.0, 0.0, 1.0),
        )
        .add_target("target", pos=TARGET_POS, radius=0.04)
        .set_table_height(TABLE_TOP_Z)
        .build_spec()
    )
