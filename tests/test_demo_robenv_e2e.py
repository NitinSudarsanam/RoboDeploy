from __future__ import annotations

import unittest

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.demo_recording import DemoSession
from robodeploy.env import RoboEnv
from test_env_refactor import DummyBackend, DummyPolicy, DummyRobot, DummyTask

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]


class DemoRoboEnvE2ETests(unittest.TestCase):
    def test_robenv_record_and_replay_explicit_actions(self):
        backend = DummyBackend()
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=backend, robots=[robot])
        session = DemoSession(env)
        session.reset()
        session.step(Action(joint_positions=jnp.asarray([1.0, 1.0], dtype=jnp.float32)))
        session.step(Action(joint_positions=jnp.asarray([2.0, 2.0], dtype=jnp.float32)))

        replay_backend = DummyBackend()
        replay_robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        replay_env = RoboEnv(backend=replay_backend, robots=[replay_robot])
        replay_env.reset()
        for action in session.iter_replay_actions():
            replay_env.step(action)
        self.assertAlmostEqual(float(replay_backend.last_actions["robot0"].joint_positions[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
