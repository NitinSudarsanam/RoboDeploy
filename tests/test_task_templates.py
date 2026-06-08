from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TaskTemplateTests(unittest.TestCase):
    def test_pick_place_template_reward(self):
        from robodeploy.core.types import Action, Observation, SceneSpec
        from robodeploy.scene_builder import SceneBuilder
        from robodeploy.tasks.templates import PickPlaceTemplate

        class _Task(PickPlaceTemplate):
            def scene_spec(self) -> SceneSpec:
                return (
                    SceneBuilder()
                    .add_box("source", size=(0.025, 0.025, 0.025), pos=(0.55, 0.0, 0.38), mass=0.05)
                    .add_box("target", size=(0.04, 0.04, 0.003), pos=(0.60, 0.20, 0.38), fixed=True)
                    .build_spec()
                )

        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        task = _Task()
        obs = Observation(
            joint_positions=jnp.zeros(7),
            joint_velocities=jnp.zeros(7),
            joint_torques=jnp.zeros(7),
            ee_position=jnp.array([0.55, 0.0, 0.45]),
            ee_orientation=jnp.array([1.0, 0.0, 0.0, 0.0]),
            ee_velocity=jnp.zeros(3),
            ee_angular_velocity=jnp.zeros(3),
            objects={"source": ((0.55, 0.0, 0.39), (1.0, 0.0, 0.0, 0.0))},
        )
        reward = task.reward_fn(obs, Action())
        self.assertIsInstance(reward, float)

    def test_example_pick_place_import(self):
        from examples.tasks.pick_place import PickPlaceTask

        task = PickPlaceTask()
        spec = task.scene_spec()
        self.assertEqual(len(spec.to_world().props), 2)
