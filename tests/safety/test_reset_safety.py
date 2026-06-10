from __future__ import annotations

import unittest
from typing import Iterator

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.action_adapter import DeltaEEToJointPosTransform
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from robodeploy.policies.learned.diffusion import DiffusionPolicy
from robodeploy.safety import EStop, SafetyMonitor
from robodeploy.tasks.base import TaskBase
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class _ResetMotionTask(TaskBase):
    """Yields a large joint delta during reset_routine for safety tests."""

    def obs_spec(self):
        from robodeploy.core.types import ObsSpec

        return ObsSpec()

    def scene_spec(self):
        from robodeploy.core.types import SceneSpec

        return SceneSpec()

    def language_instruction(self) -> str:
        return "reset"

    def reset_fn(self, backend) -> None:
        del backend

    def reset_routine(self, backend) -> Iterator[Action]:
        del backend
        yield Action(joint_positions=jnp.asarray([2.5, 2.5], dtype=jnp.float32))

    def reward_fn(self, obs, action) -> float:
        del obs, action
        return 0.0

    def success_fn(self, obs) -> bool:
        del obs
        return False

    def failure_fn(self, obs) -> bool:
        del obs
        return False


class ResetSafetyTests(unittest.TestCase):
    def test_estop_reset_skips_reset_routine_motion(self):
        estop = EStop(signal_handlers=False)
        monitor = SafetyMonitor(estop=estop)
        backend = DummyBackend()
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=_ResetMotionTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=backend, robots=[robot], safety=monitor)
        env.reset()
        env.emergency_stop("unit-test")
        backend.last_actions.clear()
        env.reset()
        self.assertNotIn("robot0", backend.last_actions)

    def test_normal_reset_runs_reset_routine(self):
        backend = DummyBackend()
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=_ResetMotionTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=backend, robots=[robot])
        env.reset()
        action = backend.last_actions.get("robot0")
        self.assertIsNotNone(action)
        self.assertIsNotNone(action.joint_positions)
        self.assertGreater(float(action.joint_positions[0]), 1.0)

    def test_reset_routine_step_multi_includes_all_robots(self):
        backend = DummyBackend()
        robot0 = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=_ResetMotionTask(), policies={"p": DummyPolicy(0.0)})},
        )
        robot1 = Robot(
            robot_id="robot1",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(1.0)})},
        )
        env = RoboEnv(backend=backend, robots=[robot0, robot1])
        env.reset()
        self.assertIn("robot0", backend.last_actions)
        self.assertIn("robot1", backend.last_actions)
        self.assertIsNotNone(backend.last_actions["robot1"].joint_positions)
        env.close()

    def test_reset_warm_starts_delta_ee_adapter(self):
        policy = DiffusionPolicy(config={"action_space": "delta_ee", "plan_horizon": 1})
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": policy})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        env.reset()
        delta_transforms = [
            t
            for t in robot.action_adapter.transforms
            if isinstance(t, DeltaEEToJointPosTransform)
        ]
        self.assertEqual(len(delta_transforms), 1)
        self.assertIsNotNone(delta_transforms[0]._last_qpos)  # noqa: SLF001
        env.close()


if __name__ == "__main__":
    unittest.main()
