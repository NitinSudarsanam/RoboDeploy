from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class EnvSensorRigYamlTests(unittest.TestCase):
    def test_from_config_yaml_wrist_imu_and_overhead_rgbd_materialize_four_plus_sensors(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")

        from robodeploy.env import RoboEnv

        cfg = {
            "robot": "kuka",
            "backend": "mujoco",
            "task": "showcase_scene",
            "policy": "example_joint_track",
            "task_kwargs": {"require_objects": True},
            "custom_modules": ["examples.tasks", "examples.sensors", "examples.policies"],
            "sensor_rigs": [
                {
                    "rig_id": "arm_sensors",
                    "ee_link": "robot0/ee_link",
                    "wrist_rgbd": {"width": 32, "height": 24, "depth": False, "allow_camera_fallback": True},
                    "overhead_rgbd": {
                        "mount": {
                            "parent_link": "world",
                            "position": [0.0, -0.8, 0.9],
                            "orientation": [1.0, 0.0, 0.0, 0.0],
                        },
                        "width": 32,
                        "height": 24,
                        "depth": False,
                        "allow_camera_fallback": True,
                    },
                    "wrist_imu": {},
                    "prop_pose": {"prop_names": ["showcase_box"]},
                }
            ],
            "backend_kwargs": {"config": {"allow_actuator_name_fallback": True, "enable_viewer": False}},
        }
        env = RoboEnv.from_config(cfg)
        try:
            names = {s.name for s in env.robots[0].sensors}
            self.assertGreaterEqual(len(names), 4)
            self.assertIn("wrist_camera", names)
            self.assertIn("overhead_camera", names)
            self.assertIn("wrist_imu", names)
            self.assertIn("prop_pose", names)
        finally:
            env.close()

    def test_mujoco_showcase_preset_has_wrist_imu_in_yaml(self):
        from examples.config import load_example_preset

        preset = load_example_preset("mujoco_showcase_kuka")
        rig = preset["sensor_rigs"][0]
        self.assertIn("wrist_imu", rig)
        self.assertIn("overhead_rgbd", rig)


if __name__ == "__main__":
    unittest.main()
