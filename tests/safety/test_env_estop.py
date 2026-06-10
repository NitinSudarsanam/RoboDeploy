from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from robodeploy.safety import EStop, SafetyMonitor
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class EnvEstopTests(unittest.TestCase):
    def test_estop_trip_halts_env_on_next_step(self):
        estop = EStop(signal_handlers=False)
        monitor = SafetyMonitor(estop=estop)
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot], safety=monitor)
        env.reset()
        env.emergency_stop("unit-test")
        obs, reward, done, info = env.step(
            Action(joint_positions=jnp.asarray([0.5, 0.5], dtype=jnp.float32))
        )
        self.assertTrue(done)
        self.assertEqual(reward, 0.0)
        self.assertTrue(info.extra.get("safety", {}).get("tripped", False))
        self.assertIsNotNone(obs.joint_positions)

    def test_emergency_stop_holds_last_position(self):
        backend = DummyBackend()
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=backend, robots=[robot])
        env.reset()
        env.step(Action(joint_positions=jnp.asarray([0.3, 0.4], dtype=jnp.float32)))
        hold_before = backend.last_actions["robot0"]
        env.emergency_stop("hold-test")
        hold_after = backend.last_actions["robot0"]
        self.assertIsNotNone(hold_after.joint_positions)
        sent = jnp.asarray(hold_after.joint_positions, dtype=jnp.float32)
        self.assertAlmostEqual(float(sent[0]), float(hold_before.joint_positions[0]), places=3)
        env.close()

    def test_safety_payload_on_normal_step(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        _, info = env.reset()
        self.assertIn("safety", info.extra)
        _, _, _, info2 = env.step(
            Action(joint_positions=jnp.asarray([0.1, 0.1], dtype=jnp.float32))
        )
        self.assertIn("safety", info2.extra)


if __name__ == "__main__":
    unittest.main()
