from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Observation
from robodeploy.policies.learned.adapter import LearnedActionAdapter


class _MockIKSolver:
    def fk(self, q):
        del q
        return np.array([0.5, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0])

    def ik(self, target_pos, target_quat, q_init=None):
        del target_quat
        base = np.zeros(3) if q_init is None else np.asarray(q_init, dtype=np.float64)
        base[0] = float(target_pos[0])
        return base


def _obs() -> Observation:
    return Observation(
        joint_positions=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        joint_velocities=np.zeros(3, dtype=np.float32),
        joint_torques=np.zeros(3, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


class LearnedActionAdapterTests(unittest.TestCase):
    def test_unnormalize_and_joint_pos(self):
        adapter = LearnedActionAdapter(
            source_space=ActionSpace.JOINT_POS,
            target_space=ActionSpace.JOINT_POS,
            source_dim=2,
            target_dim=2,
            normalization={"low": -1.0, "high": 1.0, "out_min": 0.0, "out_max": 1.0},
        )
        action = adapter(np.array([0.0, 1.0]), _obs())
        self.assertAlmostEqual(float(action.joint_positions[0]), 0.5, places=5)
        self.assertAlmostEqual(float(action.joint_positions[1]), 1.0, places=5)

    def test_delta_ee_to_joint_pos_via_ik(self):
        adapter = LearnedActionAdapter(
            source_space=ActionSpace.DELTA_EE,
            target_space=ActionSpace.JOINT_POS,
            source_dim=3,
            target_dim=3,
            ik_solver=_MockIKSolver(),
            dt=1.0,
        )
        action = adapter(np.array([0.05, 0.0, 0.0]), _obs())
        self.assertIsNotNone(action.joint_positions)
        self.assertAlmostEqual(float(action.joint_positions[0]), 0.55, places=3)


if __name__ == "__main__":
    unittest.main()
