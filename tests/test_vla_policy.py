from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.robot import RobotTask
from robodeploy.core.types import Observation, ObsSpec, SceneSpec
from robodeploy.policies.learned.vla import VLAPolicy
from robodeploy.tasks.base import TaskBase


def make_obs(*, rgb=None, language_instruction=None) -> Observation:  # noqa: ANN001
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        rgb=rgb,
        language_instruction=language_instruction,
    )


class _LanguageTask(TaskBase):
    def obs_spec(self) -> ObsSpec:
        return ObsSpec(rgb=True)

    def scene_spec(self) -> SceneSpec:
        return SceneSpec()

    def language_instruction(self) -> str:
        return "pick the red cube"

    def reset_fn(self, backend) -> None:  # noqa: ANN001
        self._bind_backend(backend)

    def reward_fn(self, obs, action) -> float:  # noqa: ANN001
        del obs, action
        return 0.0

    def success_fn(self, obs) -> bool:  # noqa: ANN001
        del obs
        return False

    def failure_fn(self, obs) -> bool:  # noqa: ANN001
        del obs
        return False


class VLAPolicyTests(unittest.TestCase):
    def test_robot_task_policy_observation_injects_language(self):
        task = _LanguageTask()
        robot_task = RobotTask(task=task, policies={"p": VLAPolicy(config={"action_space": "joint_pos"})})
        annotated = robot_task.policy_observation(make_obs())
        self.assertEqual(annotated.language_instruction, "pick the red cube")

    def test_vla_predictor_receives_instruction_and_rgb(self):
        captured = {}

        def predict_fn(packet):
            captured.update(packet)
            return {"ee_position": [0.1, 0.0, 0.0], "gripper": 1.0}

        policy = VLAPolicy(config={"predict_fn": predict_fn, "action_space": "delta_ee"})
        obs = make_obs(
            rgb=np.ones((4, 4, 3), dtype=np.uint8) * 127,
            language_instruction="pick the object",
        )
        action = policy.get_action(obs)

        self.assertEqual(captured["instruction"], "pick the object")
        self.assertEqual(captured["rgb"].shape, (4, 4, 3))
        self.assertTrue(action.is_delta_ee)
        self.assertEqual(action.gripper, 1.0)

    def test_vla_batch_predictor_uses_packets(self):
        seen_packets = []

        def predict_batch_fn(packets):
            seen_packets.extend(packets)
            return [
                {"joint_positions": [1.0, 2.0]},
                {"joint_positions": [3.0, 4.0]},
            ]

        policy = VLAPolicy(config={"predict_batch_fn": predict_batch_fn, "action_space": "joint_pos"})
        actions = policy.get_action_batch(
            [
                make_obs(language_instruction="one"),
                make_obs(language_instruction="two"),
            ]
        )

        self.assertEqual([packet["instruction"] for packet in seen_packets], ["one", "two"])
        self.assertEqual(float(actions[0].joint_positions[0]), 1.0)
        self.assertEqual(float(actions[1].joint_positions[0]), 3.0)

    def test_vla_heuristic_fallback_uses_instruction_and_image(self):
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        rgb[2, 0] = 255
        policy = VLAPolicy(config={"action_space": "delta_ee", "max_delta": 0.1})
        obs = make_obs(rgb=rgb, language_instruction="move left and close")
        action = policy.get_action(obs)

        self.assertTrue(action.is_delta_ee)
        self.assertEqual(action.gripper, 1.0)
        self.assertGreater(float(action.ee_position[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
