from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation, SceneSpec
from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy


def _obs(*, ft_force=None, imu_omega=None) -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.zeros((7,), dtype=jnp.float32),
        joint_torques=jnp.zeros((7,), dtype=jnp.float32),
        ee_position=jnp.asarray([0.55, 0.0, 0.425], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ft_force=ft_force,
        imu_angular_velocity=imu_omega,
    )


class SensorPolicyIntegrationTests(unittest.TestCase):
    def test_contact_grasp_engage_reads_contact_state(self):
        policy = ReachTrajectoryPolicy(
            ReachTrajectoryPolicy.default_pick_place_spec(),
            config={"grasp_detection": "contact", "contact_sensor": "wrist_contact"},
        )
        obs = _obs()
        obs = Observation(
            joint_positions=obs.joint_positions,
            joint_velocities=obs.joint_velocities,
            joint_torques=obs.joint_torques,
            ee_position=obs.ee_position,
            ee_orientation=obs.ee_orientation,
            ee_velocity=obs.ee_velocity,
            ee_angular_velocity=obs.ee_angular_velocity,
            contact_state={"wrist_contact": False},
        )
        self.assertFalse(policy._grasp_engage(obs))
        obs = Observation(
            joint_positions=obs.joint_positions,
            joint_velocities=obs.joint_velocities,
            joint_torques=obs.joint_torques,
            ee_position=obs.ee_position,
            ee_orientation=obs.ee_orientation,
            ee_velocity=obs.ee_velocity,
            ee_angular_velocity=obs.ee_angular_velocity,
            contact_state={"wrist_contact": True},
        )
        self.assertTrue(policy._grasp_engage(obs))

    def test_backend_contact_alias_maps_to_contact_sensor(self):
        policy = ReachTrajectoryPolicy(
            ReachTrajectoryPolicy.default_pick_place_spec(),
            config={"grasp_detection": "backend_contact"},
        )
        self.assertEqual(policy._grasp_detection, "contact")

    def test_ft_grasp_engage_requires_force_window(self):
        policy = ReachTrajectoryPolicy(
            ReachTrajectoryPolicy.default_pick_place_spec(),
            config={"grasp_detection": "ft", "force_threshold": 2.0, "grasp_force_window": 3},
        )
        weak = jnp.asarray([0.5, 0.0, 0.0], dtype=jnp.float32)
        strong = jnp.asarray([3.0, 0.0, 0.0], dtype=jnp.float32)
        self.assertFalse(policy._grasp_engage(_obs(ft_force=weak)))
        self.assertFalse(policy._grasp_engage(_obs(ft_force=strong)))
        self.assertTrue(policy._grasp_engage(_obs(ft_force=strong)))

    def test_imu_settle_requires_hold_steps(self):
        policy = ReachTrajectoryPolicy(
            ReachTrajectoryPolicy.default_pick_place_spec(),
            config={"imu_omega_max": 0.1, "imu_settle_steps": 2},
        )
        calm = jnp.asarray([0.01, 0.0, 0.0], dtype=jnp.float32)
        self.assertFalse(policy._imu_settled(_obs(imu_omega=calm)))
        self.assertTrue(policy._imu_settled(_obs(imu_omega=calm)))

    def test_policy_holds_on_critical_sensor_failure(self):
        policy = ReachTrajectoryPolicy(
            ReachTrajectoryPolicy.default_pick_place_spec(),
            config={"halt_on_sensor_failure": True, "critical_sensors": ["wrist_ft"]},
        )
        obs = _obs()
        obs = Observation(
            joint_positions=obs.joint_positions,
            joint_velocities=obs.joint_velocities,
            joint_torques=obs.joint_torques,
            ee_position=obs.ee_position,
            ee_orientation=obs.ee_orientation,
            ee_velocity=obs.ee_velocity,
            ee_angular_velocity=obs.ee_angular_velocity,
            sensor_status={"wrist_ft": "error"},
        )
        action = policy.get_action(obs)
        np.testing.assert_array_equal(action.joint_positions, obs.joint_positions)
        self.assertEqual(policy._last_sensor_health.get("overall"), "failed")

    def test_policy_reads_sensor_health_from_obs(self):
        policy = ReachTrajectoryPolicy(
            ReachTrajectoryPolicy.default_pick_place_spec(),
            config={"halt_on_sensor_failure": False},
        )
        obs = _obs()
        obs = Observation(
            joint_positions=obs.joint_positions,
            joint_velocities=obs.joint_velocities,
            joint_torques=obs.joint_torques,
            ee_position=obs.ee_position,
            ee_orientation=obs.ee_orientation,
            ee_velocity=obs.ee_velocity,
            ee_angular_velocity=obs.ee_angular_velocity,
            sensor_status={"wrist_imu": "stale"},
        )
        policy.get_action(obs)
        self.assertEqual(policy._last_sensor_health.get("overall"), "degraded")


if __name__ == "__main__":
    unittest.main()
