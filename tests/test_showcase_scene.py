from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ShowcaseSceneTests(unittest.TestCase):
    def test_showcase_scene_props_include_all_geom_kinds(self):
        from examples.tasks.showcase_scene import ShowcaseSceneTask

        scene = ShowcaseSceneTask().scene_spec()
        kinds = {prop.geom.kind for prop in scene.props}
        self.assertEqual(kinds, {"box", "cylinder", "sphere", "capsule"})
        names = {prop.name for prop in scene.props}
        self.assertEqual(
            names,
            {"showcase_box", "showcase_cylinder", "showcase_sphere", "showcase_capsule"},
        )

    def test_showcase_scene_mjcf_compiles_when_mujoco_installed(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")

        from robodeploy.backends.sim.mujoco.scene_builder import MjcfSceneBuilder
        from examples.tasks.showcase_scene import ShowcaseSceneTask

        scene = ShowcaseSceneTask().scene_spec()
        builder = MjcfSceneBuilder("<mujoco><worldbody/></mujoco>")
        builder.ensure_world_defaults(add_camera=False)
        builder.attach_world(scene.to_world())
        xml = builder.emit()
        for kind in ("box", "cylinder", "sphere", "capsule"):
            self.assertIn(f'type="{kind}"', xml)
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertGreater(model.nbody, 0)

    def test_showcase_scene_registered(self):
        from robodeploy.core.registry import get_task

        TaskClass = get_task("showcase_scene")
        from examples.tasks.showcase_scene import ShowcaseSceneTask

        self.assertIs(TaskClass, ShowcaseSceneTask)


if __name__ == "__main__":
    unittest.main()
