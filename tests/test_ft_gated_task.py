from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from examples.tasks.pick_place import PickPlaceTask
from robodeploy.core.types import Observation
from robodeploy.tasks.base import TaskBase


def _obs(*, ft_force=None, contact_state=None) -> Observation:
    return Observation(
        joint_positions=jnp.zeros((7,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((7,), dtype=jnp.float32),
        joint_torques=jnp.zeros((7,), dtype=jnp.float32),
        ee_position=jnp.asarray([0.6, 0.2, 0.41], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ft_force=ft_force,
        contact_state=contact_state or {},
        objects={
            "source": ((0.6, 0.2, 0.41), (1.0, 0.0, 0.0, 0.0)),
            "target": ((0.6, 0.2, 0.38), (1.0, 0.0, 0.0, 0.0)),
        },
    )


class _GraspTask(TaskBase):
    def obs_spec(self):
        from robodeploy.core.types import ObsSpec

        return ObsSpec()

    def scene_spec(self):
        from robodeploy.core.types import SceneSpec

        return SceneSpec()

    def language_instruction(self) -> str:
        return "test"

    def reset_fn(self, backend) -> None:
        return

    def reward_fn(self, obs, action) -> float:
        return 0.0

    def success_fn(self, obs) -> bool:
        return self.grasp_confirmed(obs)


class FTGatedTaskTests(unittest.TestCase):
    def test_grasp_confirmed_ft_threshold(self):
        task = _GraspTask(config={"grasp_success_force_min": 2.0})
        weak = _obs(ft_force=jnp.asarray([0.5, 0.0, 0.0], dtype=jnp.float32))
        strong = _obs(ft_force=jnp.asarray([3.0, 0.0, 0.0], dtype=jnp.float32))
        self.assertFalse(task.grasp_confirmed(weak))
        self.assertTrue(task.grasp_confirmed(strong))

    def test_grasp_confirmed_contact_fallback(self):
        task = _GraspTask(config={"grasp_success_force_min": 2.0, "contact_sensor": "wrist_contact"})
        obs = _obs(contact_state={"wrist_contact": True})
        self.assertTrue(task.grasp_confirmed(obs))

    def test_pick_place_success_requires_grasp_force(self):
        task = PickPlaceTask(config={"grasp_success_force_min": 2.0, "require_objects": True})
        at_goal_weak_ft = _obs(ft_force=jnp.asarray([0.2, 0.0, 0.0], dtype=jnp.float32))
        at_goal_strong_ft = _obs(ft_force=jnp.asarray([3.0, 0.0, 0.0], dtype=jnp.float32))
        self.assertFalse(task.success_fn(at_goal_weak_ft))
        self.assertTrue(task.success_fn(at_goal_strong_ft))

    def test_grasp_confirmed_disabled_when_threshold_zero(self):
        task = _GraspTask(config={"grasp_success_force_min": 0.0})
        self.assertTrue(task.grasp_confirmed(_obs()))


if __name__ == "__main__":
    unittest.main()
