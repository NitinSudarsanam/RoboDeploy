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


if __name__ == "__main__":
    unittest.main()

