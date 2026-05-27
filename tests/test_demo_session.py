from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from test_env_refactor import DummyBackend, DummyPolicy, DummyRobot, DummyTask
from robodeploy.core.robot import Robot, RobotTask


class DemoSessionHelperTests(unittest.TestCase):
    def test_demo_session_records_via_helper(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        session = env.demo_session()
        session.reset()
        session.step(Action(joint_positions=jnp.asarray([1.0, 1.0], dtype=jnp.float32)))
        self.assertEqual(len(session.recorder.frames), 1)


if __name__ == "__main__":
    unittest.main()
