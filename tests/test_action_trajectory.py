from __future__ import annotations

import unittest

import numpy as np

from robodeploy.action_trajectory import ActionTrajectory, ActionTrajectorySpec
from robodeploy.core.types import Action


class ActionTrajectoryTests(unittest.TestCase):
    def test_write_then_read_returns_latest(self) -> None:
        spec = ActionTrajectorySpec(robot_ids=["robot0"], dof_by_robot={"robot0": 3})
        traj = ActionTrajectory(spec)
        try:
            traj.write("robot0", Action(joint_positions=np.array([1.0, 2.0, 3.0], dtype=np.float32)))
            q, ts = traj.read_latest_joint_positions("robot0")
            self.assertIsNotNone(q)
            self.assertGreater(ts, 0.0)
            np.testing.assert_allclose(q, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        finally:
            traj.close()
            traj.unlink()

    def test_odd_sequence_returns_last_valid(self) -> None:
        spec = ActionTrajectorySpec(robot_ids=["robot0"], dof_by_robot={"robot0": 3})
        traj = ActionTrajectory(spec)
        try:
            expected = np.array([4.0, 5.0, 6.0], dtype=np.float32)
            traj.write("robot0", Action(joint_positions=expected))
            slot = traj._slot_view("robot0")
            traj._HDR.pack_into(slot, 0, 1, 0, 3)
            q, _ = traj.read_latest_joint_positions("robot0", timeout_s=0.00001)
            del slot
            np.testing.assert_allclose(q, expected)
        finally:
            traj.close()
            traj.unlink()


if __name__ == "__main__":
    unittest.main()

