from __future__ import annotations

import unittest
from unittest import mock

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import EpisodeInfo, Observation
from robodeploy.env import RoboEnv


class EnvSensorStatusStepTests(unittest.TestCase):
    def test_attach_step_observability_surfaces_sensor_status(self):
        env = object.__new__(RoboEnv)
        env._health_monitor = mock.Mock()
        env._health_monitor.observe = mock.Mock(return_value="ok")
        env._policy_diagnostics = {}
        env._backend_diagnostics = mock.Mock(return_value={})
        env._logger = None

        obs = Observation(
            joint_positions=jnp.zeros((2,), dtype=jnp.float32),
            joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
            joint_torques=jnp.zeros((2,), dtype=jnp.float32),
            ee_position=jnp.zeros((3,), dtype=jnp.float32),
            ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
            ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
            sensor_status={"wrist_ft": "ok", "wrist_imu": "stale"},
        )
        info = EpisodeInfo(episode_id=1, step=3)
        RoboEnv._attach_step_observability(
            env,
            info,
            obs=obs,
            reward=0.1,
            done=False,
            robot=mock.Mock(robot_id="robot0", active_task_id="pick"),
            robot_task=None,
            action=mock.Mock(),
        )
        self.assertEqual(info.extra["sensor_status"]["wrist_ft"], "ok")
        self.assertEqual(info.extra["sensor_status"]["wrist_imu"], "stale")
        self.assertIn("overall", info.extra["sensor_health"])
        self.assertEqual(info.extra["sensor_health"]["overall"], "degraded")


if __name__ == "__main__":
    unittest.main()
