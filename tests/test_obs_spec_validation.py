from __future__ import annotations

import unittest
import warnings

from robodeploy.core.types import ObsSpec, Observation, validate_observation

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]


def _base_obs(**kwargs) -> Observation:
    defaults = dict(
        joint_positions=jnp.zeros((2,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
        joint_torques=jnp.zeros((2,), dtype=jnp.float32),
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1, 0, 0, 0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
    )
    defaults.update(kwargs)
    return Observation(**defaults)


class ObsSpecValidationTests(unittest.TestCase):
    def test_warns_when_objects_required_but_missing(self):
        obs = _base_obs()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_observation(obs, ObsSpec(objects=True), policy="warn")
        self.assertTrue(any("objects" in str(w.message) for w in caught))

    def test_raises_when_objects_required_but_missing(self):
        obs = _base_obs()
        with self.assertRaises(ValueError):
            validate_observation(obs, ObsSpec(objects=True), policy="raise")

    def test_passes_when_objects_present(self):
        obs = _base_obs(objects={"source": ((0.0, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))})
        validate_observation(obs, ObsSpec(objects=True), policy="raise")


if __name__ == "__main__":
    unittest.main()
