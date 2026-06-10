"""Offline tests for Gazebo grasp-follow prop pose bookkeeping."""

from __future__ import annotations

import unittest

import numpy as np

from robodeploy.backends.sim.gazebo.backend import ROS2GazeboBackend


class GazeboGraspFollowTests(unittest.TestCase):
    def test_set_prop_pose_updates_bookkeeping_and_queues_gz_sync(self):
        backend = ROS2GazeboBackend(config={"sim": {"kind": "gazebo"}})
        backend._scene_prop_poses = {"source": ((0.5, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))}
        backend._gz_world_name = "robodeploy_world"
        backend._prop_pose_syncer = type(
            "S",
            (),
            {"set_entity_pose": staticmethod(lambda **kwargs: kwargs["entity_name"] == "source")},
        )()
        backend._pending_gz_prop_sync = set()

        backend.set_prop_pose("source", (0.51, 0.0, 0.42), (1.0, 0.0, 0.0, 0.0))
        self.assertIn("source", backend._pending_gz_prop_sync)
        backend._flush_pending_gz_prop_sync()
        self.assertNotIn("source", backend._pending_gz_prop_sync)
        pos, _quat = backend.get_prop_pose("source")
        self.assertAlmostEqual(pos[0], 0.51)
        self.assertAlmostEqual(pos[2], 0.42)

    def test_sync_grasped_prop_follows_ee(self):
        backend = ROS2GazeboBackend(config={"sim": {"kind": "gazebo"}})
        backend._scene_prop_poses = {"source": ((0.5, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))}
        backend._grasp_prop = "source"
        backend._grasp_mode = "follow"
        backend._grasp_offset = (0.0, 0.0, 0.03)
        backend._pending_gz_prop_sync = set()
        backend._get_ee_pose = lambda: (np.array([0.6, 0.1, 0.5]), np.array([1.0, 0.0, 0.0, 0.0]))

        backend._sync_grasped_prop()
        pos, _ = backend.get_prop_pose("source")
        self.assertAlmostEqual(pos[0], 0.6)
        self.assertAlmostEqual(pos[1], 0.1)
        self.assertAlmostEqual(pos[2], 0.53)
        self.assertIn("source", backend._pending_gz_prop_sync)


if __name__ == "__main__":
    unittest.main()
