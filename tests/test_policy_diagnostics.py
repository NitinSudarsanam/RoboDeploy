from __future__ import annotations

import unittest
from unittest import mock

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action, EpisodeInfo, Observation
from robodeploy.env import RoboEnv
from robodeploy.policies.learned.diagnostics import PolicyDiagnostics


class PolicyDiagnosticsTests(unittest.TestCase):
    def test_summary_accumulates_per_step(self):
        diag = PolicyDiagnostics(expected_dim=2)
        for value in (0.1, 0.2, 0.3):
            diag.record(Action(joint_positions=jnp.asarray([value, value], dtype=jnp.float32)))
        summary = diag.summary()
        self.assertEqual(summary["count"], 3)
        self.assertEqual(len(summary["action_mean"]), 2)

    def test_env_step_surfaces_policy_diagnostics(self):
        env = object.__new__(RoboEnv)
        env._health_monitor = mock.Mock()
        env._health_monitor.observe = mock.Mock(return_value="ok")
        diag = PolicyDiagnostics(expected_dim=2)
        env._policy_diagnostics = {"robot0": diag}
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
        )
        info = EpisodeInfo(episode_id=1, step=1)
        action = Action(joint_positions=jnp.asarray([0.5, -0.2], dtype=jnp.float32))
        RoboEnv._attach_step_observability(
            env,
            info,
            obs=obs,
            reward=0.0,
            done=False,
            robot=mock.Mock(robot_id="robot0", active_task_id="task"),
            robot_task=None,
            action=action,
        )
        self.assertIn("policy_diagnostics", info.extra)
        self.assertEqual(info.extra["policy_diagnostics"]["count"], 1)

        info2 = EpisodeInfo(episode_id=1, step=2)
        RoboEnv._attach_step_observability(
            env,
            info2,
            obs=obs,
            reward=0.0,
            done=False,
            robot=mock.Mock(robot_id="robot0", active_task_id="task"),
            robot_task=None,
            action=action,
        )
        self.assertEqual(info2.extra["policy_diagnostics"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
