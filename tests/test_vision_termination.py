from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Observation
from robodeploy.tasks.success_predicates import get_success_predicate, vision_target_in_view


def _obs_with_red_blob(*, min_pixels: int = 200) -> Observation:
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[20:28, 30:38] = (255, 0, 0)
    return Observation(
        joint_positions=jnp.zeros((7,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((7,), dtype=jnp.float32),
        joint_torques=jnp.zeros((7,), dtype=jnp.float32),
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        rgb=jnp.asarray(rgb),
    )


class VisionTerminationTests(unittest.TestCase):
    def test_vision_target_in_view_detects_red_blob(self):
        obs = _obs_with_red_blob()
        self.assertTrue(vision_target_in_view(obs, min_pixels=50))

    def test_vision_target_in_view_fails_without_target(self):
        obs = _obs_with_red_blob()
        obs = Observation(
            joint_positions=obs.joint_positions,
            joint_velocities=obs.joint_velocities,
            joint_torques=obs.joint_torques,
            ee_position=obs.ee_position,
            ee_orientation=obs.ee_orientation,
            ee_velocity=obs.ee_velocity,
            ee_angular_velocity=obs.ee_angular_velocity,
            rgb=jnp.zeros((48, 64, 3), dtype=jnp.uint8),
        )
        self.assertFalse(vision_target_in_view(obs, min_pixels=50))

    def test_vision_only_task_success_without_oracle_objects(self):
        """Termination via vision predicate — no obs.objects oracle."""
        predicate = get_success_predicate("vision_target_in_view")

        class _VisionTask:
            def success_fn(self, obs: Observation) -> bool:
                return predicate(obs, min_pixels=50)

        task = _VisionTask()
        self.assertTrue(task.success_fn(_obs_with_red_blob()))
        blank = _obs_with_red_blob()
        blank = Observation(
            joint_positions=blank.joint_positions,
            joint_velocities=blank.joint_velocities,
            joint_torques=blank.joint_torques,
            ee_position=blank.ee_position,
            ee_orientation=blank.ee_orientation,
            ee_velocity=blank.ee_velocity,
            ee_angular_velocity=blank.ee_angular_velocity,
            rgb=jnp.zeros((48, 64, 3), dtype=jnp.uint8),
        )
        self.assertFalse(task.success_fn(blank))


if __name__ == "__main__":
    unittest.main()
