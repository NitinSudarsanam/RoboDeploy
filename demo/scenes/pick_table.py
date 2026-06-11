"""Kuka pick-place layout for the demo (table, source cube, target)."""

from __future__ import annotations

from robodeploy.core.types import SceneSpec
from robodeploy.scene_builder import SceneBuilder

TABLE_TOP_Z = 0.38
TABLE_THICKNESS = 0.03
TABLE_HALF_THICKNESS = TABLE_THICKNESS / 2.0
TABLE_CENTER_Z = TABLE_TOP_Z - TABLE_HALF_THICKNESS
TABLE_SIZE_HALF = (0.45, 0.35, TABLE_HALF_THICKNESS)
TABLE_CENTER = (0.55, 0.0, TABLE_CENTER_Z)

CUBE_HALF = 0.025
SOURCE_POS = (0.55, 0.0, TABLE_TOP_Z + CUBE_HALF)
TARGET_POS = (0.60, 0.20, TABLE_TOP_Z + 0.04)


def build_pick_place_scene() -> SceneSpec:
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
