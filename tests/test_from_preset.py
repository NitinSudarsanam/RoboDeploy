from __future__ import annotations

import unittest
from unittest.mock import patch

from robodeploy.env import RoboEnv


class FromPresetTests(unittest.TestCase):
    def test_from_preset_passes_make_kwargs(self):
        with patch.object(RoboEnv, "make", return_value=object()) as mock_make:
            RoboEnv.from_preset("kuka_pick_mujoco", robot_id="r0")
        kwargs = mock_make.call_args.kwargs
        self.assertEqual(kwargs["robot"], "kuka")
        self.assertEqual(kwargs["backend"], "mujoco")
        self.assertEqual(kwargs["policy"], "joint_pd_stub")
        self.assertEqual(kwargs["robot_id"], "r0")


if __name__ == "__main__":
    unittest.main()
