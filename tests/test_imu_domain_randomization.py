from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.transforms import GaussianNoiseTransform
from robodeploy.core.types import Observation
from robodeploy.tasks.randomization import (
    DomainRandomizer,
    DomainRandomizerConfig,
    RandomLevel,
    SensorNoiseConfig,
)


def _obs_with_imu() -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.5, 0.0, 0.3], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        imu_acceleration=jnp.asarray([0.0, 0.0, 9.81], dtype=jnp.float32),
        imu_angular_velocity=jnp.asarray([0.05, 0.0, 0.0], dtype=jnp.float32),
    )


class ImuDomainRandomizationTests(unittest.TestCase):
    def test_full_domain_randomizer_applies_imu_noise(self):
        dr = DomainRandomizer(
            DomainRandomizerConfig(
                level=RandomLevel.FULL,
                seed=11,
                sensor_noise=SensorNoiseConfig(imu_accel_std=0.1, imu_gyro_std=0.02),
            )
        )
        transform = dr.obs_noise_transform()
        self.assertIsNotNone(transform)
        assert transform is not None
        base = _obs_with_imu()
        out = transform.forward(base)
        self.assertIsNotNone(out.imu_acceleration)
        self.assertIsNotNone(out.imu_angular_velocity)
        np.testing.assert_allclose(base.imu_acceleration, [0.0, 0.0, 9.81], rtol=1e-5)
        self.assertFalse(
            np.allclose(out.imu_acceleration, base.imu_acceleration, rtol=1e-3, atol=1e-3)
        )
        self.assertFalse(
            np.allclose(out.imu_angular_velocity, base.imu_angular_velocity, rtol=1e-3, atol=1e-3)
        )

    def test_imu_noise_zero_std_is_identity(self):
        out = GaussianNoiseTransform(imu_accel_std=0.0, imu_gyro_std=0.0).forward(_obs_with_imu())
        np.testing.assert_array_equal(out.imu_acceleration, _obs_with_imu().imu_acceleration)
        np.testing.assert_array_equal(out.imu_angular_velocity, _obs_with_imu().imu_angular_velocity)

    def test_imu_noise_deterministic_with_seed(self):
        t1 = GaussianNoiseTransform(imu_accel_std=0.05, imu_gyro_std=0.01, seed=99)
        t2 = GaussianNoiseTransform(imu_accel_std=0.05, imu_gyro_std=0.01, seed=99)
        base = _obs_with_imu()
        np.testing.assert_array_equal(
            t1.forward(base).imu_acceleration,
            t2.forward(base).imu_acceleration,
        )


if __name__ == "__main__":
    unittest.main()
