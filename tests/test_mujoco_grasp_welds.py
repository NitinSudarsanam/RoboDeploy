from __future__ import annotations

import unittest

from robodeploy.backends.sim.mujoco.scene_builder import MjcfSceneBuilder
from robodeploy.core.types import GeomSpec, PropConfig, SceneSpec


class MuJoCoGraspWeldXmlTests(unittest.TestCase):
    def test_attach_grasp_welds_emits_inactive_equalities(self):
        world = SceneSpec(
            props=[
                PropConfig(name="source", is_fixed=False, geom=GeomSpec(kind="box", size=(0.02, 0.02, 0.02))),
                PropConfig(name="target", is_fixed=True, geom=GeomSpec(kind="box", size=(0.04, 0.04, 0.003))),
            ]
        ).to_world()
        builder = MjcfSceneBuilder('<mujoco><worldbody/></mujoco>')
        builder.attach_world(world)
        builder.attach_grasp_welds("robot0/ee_link", world.props)
        xml = builder.emit()
        self.assertIn('weld name="grasp_source"', xml)
        self.assertIn('body1="robot0/ee_link"', xml)
        self.assertIn('body2="source"', xml)
        self.assertIn('active="false"', xml)
        self.assertNotIn('grasp_target', xml)


if __name__ == "__main__":
    unittest.main()
