from __future__ import annotations

import unittest

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class RewardComponentsEnvTests(unittest.TestCase):
    def test_step_exposes_reward_components_in_extra(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        env.reset()
        _, _reward, _done, info = env.step(
            Action(joint_positions=[0.5, 0.5]),
        )
        components = info.extra.get("reward_components")
        self.assertIsInstance(components, dict)
        self.assertIn("action_norm", components)


if __name__ == "__main__":
    unittest.main()
