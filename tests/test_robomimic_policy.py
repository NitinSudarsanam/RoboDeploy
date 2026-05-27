from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.types import Observation
from robodeploy.policies.learned.robomimic import RobomimicPolicy


def make_obs() -> Observation:
    return Observation(
        joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        joint_velocities=np.zeros(7, dtype=np.float32),
        joint_torques=np.zeros(7, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


class RobomimicPolicyTests(unittest.TestCase):
    def test_injected_predict_fn_produces_joint_action(self):
        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            state = obs_dict["state"]
            return np.concatenate([state[:7] + 0.05, np.array([0.5])])

        policy = RobomimicPolicy(config={"predict_fn": predict_fn, "arm_dof": 7})
        action = policy.get_action(make_obs())
        self.assertAlmostEqual(float(action.joint_positions[0]), 0.15, places=5)
        self.assertEqual(action.gripper, 0.5)


if __name__ == "__main__":
    unittest.main()
