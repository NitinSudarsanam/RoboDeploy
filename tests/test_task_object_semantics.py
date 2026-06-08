from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action, Observation
from examples.tasks.peg_insertion import PegTask
from examples.tasks.pick_place import PickPlaceTask
from examples.tasks.pour import PourTask


def make_obs(position=(0.0, 0.0, 0.0)) -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray(position, dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
    )


class _FakeSceneBackend:
    def __init__(self, poses: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]]) -> None:
        self._poses = dict(poses)

    def get_prop_pose(self, name: str):
        return self._poses[name]


class TaskObjectSemanticsTests(unittest.TestCase):
    def test_pick_place_uses_source_prop_pose_for_reward_and_success(self):
        task = PickPlaceTask()
        goal = task._placement_goal()
        action = Action()

        far_backend = _FakeSceneBackend({"source": ((0.5, 0.0, 0.025), (1.0, 0.0, 0.0, 0.0))})
        task.reset_fn(far_backend)
        far_reward = task.reward_fn(make_obs((0.0, 0.0, 0.0)), action)
        self.assertFalse(task.success_fn(make_obs((0.0, 0.0, 0.0))))

        placed_backend = _FakeSceneBackend({"source": (goal, (1.0, 0.0, 0.0, 0.0))})
        task.reset_fn(placed_backend)
        placed_reward = task.reward_fn(make_obs((0.0, 0.0, 0.0)), action)
        self.assertTrue(task.success_fn(make_obs((0.0, 0.0, 0.0))))
        self.assertGreater(placed_reward, far_reward)

    def test_pick_place_prefers_obs_objects_when_required(self):
        task = PickPlaceTask(config={"require_objects": True})
        goal = task._placement_goal()
        action = Action()
        task.reset_fn(_FakeSceneBackend({"source": ((0.5, 0.0, 0.025), (1.0, 0.0, 0.0, 0.0))}))

        far_obs = make_obs((0.0, 0.0, 0.0))
        far_obs.objects = {"source": ((0.5, 0.0, 0.025), (1.0, 0.0, 0.0, 0.0))}
        self.assertFalse(task.success_fn(far_obs))

        placed_obs = make_obs((0.0, 0.0, 0.0))
        placed_obs.objects = {"source": (goal, (1.0, 0.0, 0.0, 0.0))}
        self.assertTrue(task.success_fn(placed_obs))

    def test_pour_uses_source_cup_pose_and_tilt(self):
        task = PourTask()
        goal = task._pour_goal()
        action = Action()

        upright_backend = _FakeSceneBackend({"cup_source": (goal, (1.0, 0.0, 0.0, 0.0))})
        task.reset_fn(upright_backend)
        self.assertFalse(task.success_fn(make_obs((0.0, 0.0, 0.0))))
        upright_reward = task.reward_fn(make_obs((0.0, 0.0, 0.0)), action)

        tilted_backend = _FakeSceneBackend({"cup_source": (goal, (0.70710678, 0.70710678, 0.0, 0.0))})
        task.reset_fn(tilted_backend)
        self.assertTrue(task.success_fn(make_obs((0.0, 0.0, 0.0))))
        self.assertGreater(task.reward_fn(make_obs((0.0, 0.0, 0.0)), action), upright_reward)

    def test_pour_prefers_obs_objects_when_required(self):
        task = PourTask(config={"require_objects": True})
        goal = task._pour_goal()
        action = Action()
        task.reset_fn(_FakeSceneBackend({"cup_source": ((0.5, 0.0, 0.04), (1.0, 0.0, 0.0, 0.0))}))

        tilted_obs = make_obs((0.0, 0.0, 0.0))
        tilted_obs.objects = {
            "cup_source": (goal, (0.70710678, 0.70710678, 0.0, 0.0)),
        }
        self.assertTrue(task.success_fn(tilted_obs))

    def test_peg_insertion_prefers_obs_objects_when_required(self):
        task = PegTask(config={"require_objects": True})
        goal = task._insert_goal()
        action = Action()
        task.reset_fn(_FakeSceneBackend({"peg": ((0.5, 0.0, 0.06), (1.0, 0.0, 0.0, 0.0))}))

        inserted_obs = make_obs((0.0, 0.0, 0.0))
        inserted_obs.objects = {"peg": (goal, (1.0, 0.0, 0.0, 0.0))}
        self.assertTrue(task.success_fn(inserted_obs))
        self.assertGreater(task.reward_fn(inserted_obs, action), task.reward_fn(make_obs(), action))

    def test_peg_insertion_uses_peg_pose_for_reward_and_success(self):
        task = PegTask()
        goal = task._insert_goal()
        action = Action()

        far_backend = _FakeSceneBackend({"peg": ((0.5, 0.0, 0.06), (1.0, 0.0, 0.0, 0.0))})
        task.reset_fn(far_backend)
        far_reward = task.reward_fn(make_obs((0.0, 0.0, 0.0)), action)
        self.assertFalse(task.success_fn(make_obs((0.0, 0.0, 0.0))))

        inserted_backend = _FakeSceneBackend({"peg": (goal, (1.0, 0.0, 0.0, 0.0))})
        task.reset_fn(inserted_backend)
        inserted_reward = task.reward_fn(make_obs((0.0, 0.0, 0.0)), action)
        self.assertTrue(task.success_fn(make_obs((0.0, 0.0, 0.0))))
        self.assertGreater(inserted_reward, far_reward)


if __name__ == "__main__":
    unittest.main()
