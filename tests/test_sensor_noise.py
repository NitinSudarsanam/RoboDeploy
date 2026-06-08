from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.transforms import GaussianNoiseTransform
from robodeploy.core.types import Observation
from robodeploy.tasks.randomization import DomainRandomizer, DomainRandomizerConfig, RandomLevel


def _obs() -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.5, 0.0, 0.3], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        images={"wrist_camera": jnp.asarray(np.full((4, 4, 3), 128, dtype=np.uint8))},
        depths={"wrist_camera": jnp.asarray(np.full((4, 4), 0.5, dtype=np.float32))},
        ft_forces={"wrist_ft": jnp.asarray([1.0, 0.0, -0.5], dtype=jnp.float32)},
        ft_torques={"wrist_ft": jnp.asarray([0.1, 0.0, 0.0], dtype=jnp.float32)},
        imu_acceleration=jnp.asarray([0.0, 0.0, 9.8], dtype=jnp.float32),
        imu_angular_velocity=jnp.asarray([0.01, 0.0, 0.0], dtype=jnp.float32),
    )


class SensorNoiseTests(unittest.TestCase):
    def test_zero_std_is_identity(self):
        out = GaussianNoiseTransform(
            joint_pos_std=0.0,
            joint_vel_std=0.0,
            ee_pos_std=0.0,
            rgb_std=0.0,
            depth_std=0.0,
            ft_force_std=0.0,
            ft_torque_std=0.0,
            imu_accel_std=0.0,
            imu_gyro_std=0.0,
            seed=0,
        ).forward(_obs())
        np.testing.assert_array_equal(out.joint_positions, _obs().joint_positions)
        np.testing.assert_array_equal(out.images["wrist_camera"], _obs().images["wrist_camera"])

    def test_sensor_noise_changes_values_with_seed(self):
        transform = GaussianNoiseTransform(
            rgb_std=5.0,
            depth_std=0.01,
            ft_force_std=0.1,
            imu_accel_std=0.05,
            seed=42,
        )
        out = transform.forward(_obs())
        self.assertFalse(np.array_equal(out.images["wrist_camera"], _obs().images["wrist_camera"]))
        self.assertNotAlmostEqual(float(out.depths["wrist_camera"][0, 0]), 0.5, places=3)
        self.assertNotAlmostEqual(float(out.ft_forces["wrist_ft"][0]), 1.0, places=2)

    def test_domain_randomizer_full_returns_noise_transform(self):
        dr = DomainRandomizer(DomainRandomizerConfig(level=RandomLevel.FULL, seed=7))
        transform = dr.obs_noise_transform()
        self.assertIsNotNone(transform)
        assert transform is not None
        out = transform.forward(_obs())
        self.assertIsNotNone(out.imu_acceleration)

    def test_domain_randomizer_light_returns_none(self):
        dr = DomainRandomizer(DomainRandomizerConfig(level=RandomLevel.LIGHT))
        self.assertIsNone(dr.obs_noise_transform())


if __name__ == "__main__":
    unittest.main()
