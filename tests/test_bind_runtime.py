from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class BindRuntimeTests(unittest.TestCase):
    def test_kuka_pick_preset_binds_ik_on_reset_without_manual_wire(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")

        from examples.env_from_preset import env_from_preset
        from examples.policies.reach_pick_place import ReachPickPlacePolicy

        env = env_from_preset("kuka_pick_mujoco", max_episode_steps=10)
        try:
            env.reset()
            policy = None
            for robot in env.robots:
                for robot_task in robot.tasks.values():
                    for pol in robot_task.policies.values():
                        if isinstance(pol, ReachPickPlacePolicy):
                            policy = pol
            self.assertIsNotNone(policy)
            assert policy is not None
            self.assertIsNotNone(policy._ik)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
