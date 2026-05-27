from __future__ import annotations

import unittest

from robodeploy.config import load_preset


class PresetTests(unittest.TestCase):
    def test_load_known_preset(self):
        preset = load_preset("kuka_pick_mujoco")
        self.assertEqual(preset["backend"], "mujoco")
        self.assertEqual(preset["task"], "pick_place")

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            load_preset("does_not_exist_xyz")


if __name__ == "__main__":
    unittest.main()
