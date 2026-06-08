from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.config import PRESETS_FILE, load_example_preset, list_example_presets
from examples.env_from_preset import env_from_preset


class PresetsTests(unittest.TestCase):
    def test_list_presets_includes_known_names(self):
        names = list_example_presets()
        self.assertIn("kuka_pick_mujoco", names)
        self.assertIn("kuka_sensor_pick_mujoco", names)
        self.assertIn("kuka_sensor_ros2_rviz", names)
        self.assertIn("mujoco_showcase_kuka", names)
        self.assertIn("mujoco_showcase_franka", names)
        self.assertIn("mujoco_pick_kuka", names)

    def test_load_preset_returns_required_keys(self):
        preset = load_example_preset("kuka_pick_mujoco")
        self.assertEqual(preset["robot"], "kuka")
        self.assertEqual(preset["backend"], "mujoco")
        self.assertEqual(preset["policy"], "example_sensor_reach_pick")

    def test_mujoco_showcase_kuka_preset_keys(self):
        preset = load_example_preset("mujoco_showcase_kuka")
        self.assertEqual(preset["robot"], "kuka")
        self.assertEqual(preset["task"], "showcase_scene")
        self.assertEqual(preset["policy"], "example_joint_track")
        rig = preset["sensor_rigs"][0]
        self.assertIn("wrist_rgbd", rig)
        self.assertIn("wrist_imu", rig)
        self.assertIn("overhead_rgbd", rig)
        self.assertIn("wrist_ft", rig)
        self.assertEqual(len(rig["prop_pose"]["prop_names"]), 4)

    def test_load_preset_unknown_raises(self):
        with self.assertRaises(KeyError):
            load_example_preset("does_not_exist_xyz")

    def test_env_from_preset_delegates_to_from_config(self):
        with patch("robodeploy.env.RoboEnv.from_config", return_value=object()) as mock_from_config:
            env_from_preset("kuka_pick_mujoco", robot_id="r0")
        cfg = mock_from_config.call_args.args[0]
        self.assertEqual(cfg["robot"], "kuka")
        self.assertEqual(cfg["policy"], "example_sensor_reach_pick")
        self.assertEqual(cfg["robot_id"], "r0")

    def test_load_preset_requires_keys(self):
        from unittest.mock import patch

        from examples.config import load_preset

        with patch(
            "examples.config._load_all_presets",
            return_value={"bad": {"robot": "kuka"}},
        ):
            with self.assertRaises(ValueError):
                load_preset("bad", presets_file=PRESETS_FILE)


if __name__ == "__main__":
    unittest.main()
