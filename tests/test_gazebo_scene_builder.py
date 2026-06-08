from __future__ import annotations

import unittest
from pathlib import Path

from robodeploy.backends.sim.gazebo.scene_builder import GazeboSceneBuilder
from robodeploy.core.types import GeomSpec, PropConfig, WorldSpec


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class GazeboSceneBuilderTests(unittest.TestCase):
    def test_mesh_geom_uses_file_uri(self):
        world = WorldSpec(
            props=[
                PropConfig(
                    name="mesh_prop",
                    geom=GeomSpec(kind="mesh", size=(), mesh_path=str(FIXTURES / "gazebo_minimal_arm.urdf")),
                    position=(0.1, 0.0, 0.2),
                )
            ]
        )
        sdf = GazeboSceneBuilder().build(world)
        self.assertIn("<mesh>", sdf)
        self.assertIn("gazebo_minimal_arm.urdf", sdf)

    def test_capsule_compound_collision_parts(self):
        world = WorldSpec(
            props=[
                PropConfig(
                    name="peg",
                    geom=GeomSpec(kind="capsule", size=(0.015, 0.06)),
                    mass=0.02,
                )
            ]
        )
        sdf = GazeboSceneBuilder().build(world)
        self.assertIn('model name="peg"', sdf)
        self.assertIn("peg_cap_top_collision", sdf)
        self.assertIn("peg_cap_bot_collision", sdf)

    def test_empty_fixture_world_parses(self):
        text = (FIXTURES / "gazebo_empty.sdf").read_text(encoding="utf-8")
        self.assertIn('<world name="empty">', text)


if __name__ == "__main__":
    unittest.main()
