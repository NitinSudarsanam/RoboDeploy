from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from robodeploy.policies.learned.diffusion import DiffusionPolicy
from robodeploy.safety import (
    EStopGuard,
    ForceLimitGuard,
    SafetyFilterGuard,
    VelocityGuard,
)
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class DefaultSafetyMonitorTests(unittest.TestCase):
    def test_default_monitor_aggregates_four_guard_types(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        monitor = env.safety_monitor
        self.assertIsNotNone(monitor)
        guard_types = {type(g) for g in monitor._guards}  # noqa: SLF001
        self.assertIn(SafetyFilterGuard, guard_types)
        self.assertIn(ForceLimitGuard, guard_types)
        self.assertIn(VelocityGuard, guard_types)
        self.assertIn(EStopGuard, guard_types)
        self.assertGreaterEqual(len(monitor._guards), 4)  # noqa: SLF001
        env.close()

    def test_env_clamps_out_of_bounds_joint_action(self):
        backend = DummyBackend()
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=backend, robots=[robot])
        env.reset()
        env.step(Action(joint_positions=jnp.asarray([99.0, -99.0], dtype=jnp.float32)))
        sent = backend.last_actions["robot0"]
        clamped = jnp.asarray(sent.joint_positions, dtype=jnp.float32)
        self.assertAlmostEqual(float(clamped[0]), 3.14, places=3)
        self.assertAlmostEqual(float(clamped[1]), -3.14, places=3)
        env.close()

    def test_safety_filter_guard_uses_effective_action_space(self):
        policy = DiffusionPolicy(config={"action_space": "delta_ee", "plan_horizon": 1})
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": policy})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        self.assertEqual(policy.action_space, ActionSpace.DELTA_EE)
        self.assertEqual(robot.effective_action_space, ActionSpace.JOINT_POS)
        filter_guards = [
            g
            for g in env.safety_monitor._guards  # noqa: SLF001
            if isinstance(g, SafetyFilterGuard) and g.robot_id == "robot0"
        ]
        self.assertEqual(len(filter_guards), 1)
        self.assertEqual(filter_guards[0]._action_space, ActionSpace.JOINT_POS)  # noqa: SLF001
        env.close()


if __name__ == "__main__":
    unittest.main()
