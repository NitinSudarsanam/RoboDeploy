from __future__ import annotations

import unittest
from unittest import mock

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action, EpisodeInfo, Observation
from robodeploy.env import RoboEnv
from robodeploy.observability.health import HealthMonitor


def _obs(sensor_status: dict[str, str]) -> Observation:
    return Observation(
        joint_positions=jnp.zeros((2,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
        joint_torques=jnp.zeros((2,), dtype=jnp.float32),
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        sensor_status=sensor_status,
    )


class SensorHealthInfoTests(unittest.TestCase):
    def test_attach_step_observability_surfaces_sensor_health(self):
        info = EpisodeInfo()
        robot = mock.Mock()
        robot.robot_id = "robot0"
        robot.active_task_id = "pick_place"
        env = mock.Mock()
        env._health_monitor = HealthMonitor()
        env._logger = None
        RoboEnv._attach_step_observability(
            env,
            info,
            obs=_obs({"wrist_ft": "ok", "wrist_camera": "stale"}),
            reward=0.0,
            done=False,
            robot=robot,
            robot_task=None,
            action=Action(),
        )
        self.assertEqual(info.extra["sensor_status"]["wrist_camera"], "stale")
        self.assertEqual(info.extra["sensor_health"]["overall"], "degraded")

    def test_attach_step_observability_failed_on_error(self):
        info = EpisodeInfo()
        robot = mock.Mock()
        robot.robot_id = "robot0"
        robot.active_task_id = "pick_place"
        env = mock.Mock()
        env._health_monitor = HealthMonitor()
        env._logger = None
        RoboEnv._attach_step_observability(
            env,
            info,
            obs=_obs({"wrist_ft": "error"}),
            reward=0.0,
            done=False,
            robot=robot,
            robot_task=None,
            action=Action(),
        )
        self.assertEqual(info.extra["sensor_health"]["overall"], "failed")


if __name__ == "__main__":
    unittest.main()
