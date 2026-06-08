from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from robodeploy.backends.sim.gazebo.backend import ROS2GazeboBackend
from robodeploy.core.types import Action, Observation


class _FakeDriver:
    def get_obs(self):
        z = np.zeros(7, dtype=np.float32)
        z3 = np.zeros(3, dtype=np.float32)
        return Observation(
            joint_positions=z,
            joint_velocities=z,
            joint_torques=z,
            ee_position=np.array([0.5, 0.1, 0.6], dtype=np.float32),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ee_velocity=z3,
            ee_angular_velocity=z3,
        )

    def send_action(self, action: Action) -> None:
        del action


class GazeboGraspFollowTests(unittest.TestCase):
    def test_set_grasp_prop_follow_tracks_ee(self):
        backend = ROS2GazeboBackend({})
        backend._initialized = True
        backend._drivers = {"robot0": _FakeDriver()}
        backend._scene_prop_poses = {"source": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))}
        backend._grasp_prop = None
        backend._grasp_mode = "follow"
        backend._grasp_offset = (0.0, 0.0, 0.03)

        backend.set_grasp_prop("source", mode="follow")
        pos, _ = backend.get_prop_pose("source")
        self.assertAlmostEqual(pos[0], 0.5, places=3)
        self.assertAlmostEqual(pos[1], 0.1, places=3)
        self.assertAlmostEqual(pos[2], 0.63, places=3)

    def test_weld_mode_not_supported(self):
        backend = ROS2GazeboBackend({})
        with self.assertRaises(NotImplementedError):
            backend.set_grasp_prop("source", mode="weld")


if __name__ == "__main__":
    unittest.main()
