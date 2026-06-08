"""Teleop e-stop wired to SafetyFilter + RoboEnv.emergency_stop."""

from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action, Observation
from robodeploy.demo_recording import InteractiveDemoSession
from robodeploy.env import RoboEnv
from robodeploy.teleop.base import ITeleopDevice, TeleopCommand
from robodeploy.teleop.controller import TeleopPolicy
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class _EstopDevice(ITeleopDevice):
    def __init__(self, *, trip_on_step: int = 1) -> None:
        self._step = 0
        self._trip_on = int(trip_on_step)

    def start(self) -> None:
        return

    def poll(self) -> TeleopCommand | None:
        self._step += 1
        if self._step >= self._trip_on:
            return TeleopCommand(e_stop=True)
        return TeleopCommand(delta_joint_positions=jnp.asarray([0.01, 0.0], dtype=jnp.float32))

    def stop(self) -> None:
        return


class TeleopEstopIntegrationTests(unittest.TestCase):
    def test_teleop_estop_trips_safety_filter_and_monitor(self):
        device = _EstopDevice(trip_on_step=2)
        policy = TeleopPolicy(device=device)
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": policy})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        env.reset()
        policy.bind_runtime(env.backend, robot.description)

        session = InteractiveDemoSession(env, policy, output_dir=".", start_recording=False, max_steps=10)
        session.run()

        self.assertTrue(env._safety.tripped)
        self.assertTrue(robot.description.get_safety_filter().estop_active)

    def test_emergency_stop_freezes_safety_filter_actions(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        obs, _ = env.reset()
        env.emergency_stop("test")
        filt = robot.description.get_safety_filter()
        frozen = filt.filter(
            Action(joint_positions=jnp.asarray([0.9, 0.9], dtype=jnp.float32)),
            __import__("robodeploy.core.spaces", fromlist=["ActionSpace"]).ActionSpace.JOINT_POS,
        )
        self.assertTrue(filt.estop_active)
        self.assertIsNotNone(frozen.joint_positions)


if __name__ == "__main__":
    unittest.main()
