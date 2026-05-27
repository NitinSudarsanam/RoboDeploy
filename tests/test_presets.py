from __future__ import annotations

import unittest

from robodeploy.config import list_presets, load_preset


class PresetTests(unittest.TestCase):
    def test_list_presets_includes_known_names(self):
        names = list_presets()
        self.assertIn("kuka_pick_mujoco", names)

    def test_load_known_preset(self):
        preset = load_preset("kuka_pick_mujoco")
        self.assertEqual(preset["backend"], "mujoco")
        self.assertEqual(preset["task"], "pick_place")
        self.assertEqual(preset["policy"], "joint_pd_stub")

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            load_preset("does_not_exist_xyz")


if __name__ == "__main__":
    unittest.main()
