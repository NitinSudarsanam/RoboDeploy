from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SceneBuilderTests(unittest.TestCase):
    def test_add_box_builds_spec(self):
        from robodeploy.scene_builder import SceneBuilder

        spec = (
            SceneBuilder()
            .add_box("cube", size=(0.04, 0.04, 0.04), pos=(0.5, 0.0, 0.4), mass=0.05)
            .build_spec()
        )
        props = spec.to_world().props
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0].name, "cube")
        self.assertEqual(props[0].geom.kind, "box")

    def test_pick_place_scene_validates(self):
        from robodeploy.scene_builder import SceneBuilder

        builder = (
            SceneBuilder()
            .add_box("source", size=(0.025, 0.025, 0.025), pos=(0.55, 0.0, 0.38), mass=0.05)
            .add_box("target", size=(0.04, 0.04, 0.003), pos=(0.60, 0.20, 0.38), fixed=True)
        )
        report_ok = builder.build_ir()
        from robodeploy.core.scene_validator import SceneValidator

        self.assertTrue(SceneValidator().validate(report_ok, "mujoco").ok)

    def test_add_prop_accepts_prop_config(self):
        from robodeploy.core.types import GeomSpec, MaterialSpec, PropConfig
        from robodeploy.scene_builder import SceneBuilder

        prop = PropConfig(
            name="peg",
            geom=GeomSpec(kind="cylinder", size=(0.01, 0.06)),
            position=(0.5, 0.0, 0.4),
            mass=0.02,
        )
        ir = SceneBuilder().add_prop(prop).build_ir()
        self.assertEqual(len(ir.props), 1)
        self.assertEqual(ir.props[0].name, "peg")
        self.assertEqual(ir.props[0].geometry.kind, "cylinder")
