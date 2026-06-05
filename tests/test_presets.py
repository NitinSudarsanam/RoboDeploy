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

    def test_load_preset_returns_required_keys(self):
        preset = load_example_preset("kuka_pick_mujoco")
        self.assertEqual(preset["robot"], "kuka")
        self.assertEqual(preset["backend"], "mujoco")
        self.assertEqual(preset["policy"], "example_reach_pick")

    def test_load_preset_unknown_raises(self):
        with self.assertRaises(KeyError):
            load_example_preset("does_not_exist_xyz")

    def test_env_from_preset_delegates_to_from_config(self):
        with patch("robodeploy.env.RoboEnv.from_config", return_value=object()) as mock_from_config:
            env_from_preset("kuka_pick_mujoco", robot_id="r0")
        cfg = mock_from_config.call_args.args[0]
        self.assertEqual(cfg["robot"], "kuka")
        self.assertEqual(cfg["policy"], "example_reach_pick")
        self.assertEqual(cfg["robot_id"], "r0")

    def test_robodeploy_loader_requires_presets_file(self):
        from robodeploy.config import load_preset

        with self.assertRaises(TypeError):
            load_preset("kuka_pick_mujoco")


if __name__ == "__main__":
    unittest.main()
