from __future__ import annotations

import unittest

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace
from robodeploy.env import RoboEnv
from robodeploy.policies.learned.diffusion import DiffusionPolicy
from robodeploy.policies.learned.negotiation import ActionSpaceIncompatibility, can_adapt, negotiate_action_space
from robodeploy.testing import DummyBackend, DummyRobot, DummyTask


class ActionSpaceNegotiationTests(unittest.TestCase):
    def test_can_adapt_delta_ee_to_joint_pos(self):
        self.assertTrue(can_adapt(ActionSpace.DELTA_EE, ActionSpace.JOINT_POS))

    def test_negotiate_inserts_adapter(self):
        policy = DiffusionPolicy(config={"action_space": "delta_ee"})
        backend = DummyBackend()
        robot_desc = DummyRobot()
        _, effective, adapter = negotiate_action_space(policy, backend, robot_desc)
        self.assertEqual(effective, ActionSpace.JOINT_POS)
        self.assertGreater(len(adapter.transforms), 0)

    def test_env_auto_negotiates_for_delta_policy(self):
        policy = DiffusionPolicy(config={"action_space": "delta_ee", "plan_horizon": 1})
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": policy})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        self.assertEqual(robot.effective_action_space, ActionSpace.JOINT_POS)
        self.assertGreater(len(robot.action_adapter.transforms), 0)
        env.close()

    def test_incompatible_space_raises(self):
        from robodeploy.core.types import Action
        from robodeploy.policies.base import PolicyBase
        from robodeploy.testing.dummies import make_obs

        class TorquePolicy(PolicyBase):
            def __init__(self):
                super().__init__(action_space=ActionSpace.JOINT_TORQUE, config={})

            def get_action(self, obs):  # noqa: ANN001
                return Action(joint_torques=obs.joint_torques)

        with self.assertRaises(ActionSpaceIncompatibility):
            negotiate_action_space(TorquePolicy(), DummyBackend(), DummyRobot())


if __name__ == "__main__":
    unittest.main()
