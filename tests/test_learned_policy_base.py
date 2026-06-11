from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Observation
from robodeploy.policies.learned.base import LearnedPolicyBase


def _obs() -> Observation:
    return Observation(
        joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        joint_velocities=np.zeros(7, dtype=np.float32),
        joint_torques=np.zeros(7, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


class LearnedPolicyBaseTests(unittest.TestCase):
    def test_predict_and_adapt_pipeline(self):
        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            return obs_dict["state"][:7] + 0.1

        policy = LearnedPolicyBase(
            action_space=ActionSpace.JOINT_POS,
            config={"predict_fn": predict_fn, "arm_dof": 7},
            model_spec={
                "framework": "custom",
                "checkpoint": "unused.pt",
                "expected_action_space": ActionSpace.JOINT_POS,
                "expected_action_dim": 7,
                "expected_obs_keys": ["state"],
            },
        )
        action = policy.get_action(_obs())
        self.assertAlmostEqual(float(action.joint_positions[0]), 0.2, places=5)

    def test_learned_policy_files_at_most_50_lines(self):
        from pathlib import Path

        learned = Path(__file__).resolve().parents[1] / "robodeploy" / "policies" / "learned"
        for name in ("robomimic.py", "diffusion.py", "vla.py"):
            lines = (learned / name).read_text(encoding="utf-8").splitlines()
            self.assertLessEqual(len(lines), 50, msg=name)


if __name__ == "__main__":
    unittest.main()
