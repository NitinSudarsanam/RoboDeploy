from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, ObsSpec, Observation, SceneSpec, SensorData
from robodeploy.description.base import RobotDescription
from robodeploy.env import RoboEnv
from robodeploy.obs_pipeline import ObsPipeline
from robodeploy.policies.base import PolicyBase
from robodeploy.tasks.base import TaskBase


class _Desc(RobotDescription):
    dof = 2
    display_name = "d"
    ee_link_name = "ee"
    joint_names = ["j1", "j2"]
    joint_position_limits = jnp.asarray([[-1, 1], [-1, 1]], dtype=jnp.float32)
    joint_velocity_limits = jnp.asarray([1, 1], dtype=jnp.float32)
    joint_torque_limits = jnp.asarray([1, 1], dtype=jnp.float32)
    home_qpos = jnp.asarray([0.0, 0.0], dtype=jnp.float32)

    def asset_path(self, fmt, variant: str = "default"):
        del fmt, variant
        return ""


class _Task(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec()

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "hold"

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs: Observation, action: Action) -> float:
        del obs, action
        return 0.0

    def success_fn(self, obs: Observation) -> bool:
        del obs
        return False


class _Policy(PolicyBase):
    def __init__(self) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)

    def _reset_impl(self) -> None:
        return

    def get_action(self, obs: Observation) -> Action:
        del obs
        return Action(joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32))


def _obs(hw: float = 0.0) -> Observation:
    return Observation(
        joint_positions=jnp.zeros((2,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
        joint_torques=jnp.zeros((2,), dtype=jnp.float32),
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        timestamp=hw,
        timestamp_hw=hw,
    )


def _robot(*, pipeline: ObsPipeline | None = None) -> Robot:
    return Robot(
        robot_id="robot0",
        description=_Desc(),
        sensors=[],
        obs_pipeline=pipeline or ObsPipeline(),
        tasks={"t": RobotTask(task=_Task(), policies={"p": _Policy()}, mode="sequential")},
    )


class EnvSensorPipelineWiringTests(unittest.TestCase):
    def test_process_robot_obs_buffers_pending_sensor_reads(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        pending = [("wrist_camera", SensorData(rgb=rgb, timestamp_hw=0.0, timestamp=0.0))]
        pipeline = ObsPipeline()
        robot = _robot(pipeline=pipeline)
        backend = mock.Mock()
        env = RoboEnv(backend=backend, robots=[robot], max_episode_steps=5)
        try:
            processed = env._process_robot_obs(robot, _obs(0.0), pending_reads=pending)
            self.assertIn("wrist_camera", processed.images)
            pipeline.reset_sync()
            processed2 = env._process_robot_obs(robot, _obs(0.1), pending_reads=[])
            self.assertNotIn("wrist_camera", processed2.images)
        finally:
            env.close()

    def test_drain_backend_sensor_reads_delegates_to_backend(self):
        pending = [("wrist_ft", SensorData(ft_force=np.ones(3, dtype=np.float32), timestamp_hw=0.0))]
        backend = mock.Mock()
        backend.drain_sensor_reads.return_value = list(pending)
        robot = _robot()
        env = RoboEnv(backend=backend, robots=[robot], max_episode_steps=5)
        try:
            drained = env._drain_backend_sensor_reads()
            self.assertEqual(len(drained), 1)
            processed = env._process_robot_obs(robot, _obs(0.0), pending_reads=drained)
            self.assertIn("wrist_ft", processed.ft_forces)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
