from __future__ import annotations

import unittest

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.selectors import WeightedPolicySelector
from robodeploy.env import RoboEnv
from test_env_refactor import DummyBackend, DummyRobot, DummyTask, RejectAwarePolicy


class MultiRobotArbitrationTests(unittest.TestCase):
    def test_weighted_policy_selector_picks_higher_weight(self):
        backend = DummyBackend()
        winner = RejectAwarePolicy(2.0)
        loser = RejectAwarePolicy(1.0)
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={
                "task0": RobotTask(
                    task=DummyTask(),
                    policies={"winner": winner, "loser": loser},
                    policy_selector=WeightedPolicySelector({"winner": 10.0, "loser": 0.1}),
                )
            },
        )
        env = RoboEnv(backend=backend, robots=[robot])
        env.reset()
        env.step()
        self.assertAlmostEqual(float(backend.last_actions["robot0"].joint_positions[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
