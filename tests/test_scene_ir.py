from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SceneIRTests(unittest.TestCase):
    def test_scene_spec_round_trip(self):
        from robodeploy.core.types import GeomSpec, MaterialSpec, PropConfig, SceneSpec
        from robodeploy.core.scene_ir import ir_to_scene_spec, scene_spec_to_ir

        spec = SceneSpec(
            props=[
                PropConfig(
                    name="cube",
                    position=(0.5, 0.0, 0.4),
                    mass=0.05,
                    geom=GeomSpec(kind="box", size=(0.04, 0.04, 0.04)),
                    material=MaterialSpec(rgba=(1.0, 0.0, 0.0, 1.0)),
                )
            ]
        )
        ir = scene_spec_to_ir(spec)
        self.assertEqual(len(ir.props), 1)
        self.assertEqual(ir.props[0].geometry.kind, "box")
        roundtrip = ir_to_scene_spec(ir)
        self.assertEqual(len(roundtrip.to_world().props), 1)
        self.assertEqual(roundtrip.to_world().props[0].name, "cube")

    def test_scene_spec_to_ir_method(self):
        from robodeploy.core.types import GeomSpec, PropConfig, SceneSpec

        spec = SceneSpec(props=[PropConfig(name="a", geom=GeomSpec(kind="sphere", size=(0.02,)))])
        ir = spec.to_ir()
        self.assertEqual(ir.props[0].geometry.kind, "sphere")
