from __future__ import annotations

import tempfile
import unittest

import numpy as np

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from robodeploy.observability.snapshot import SnapshotManager
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class SnapshotTests(unittest.TestCase):
    def test_capture_restore_round_trip_dummy_backend(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        mgr = SnapshotManager(env=env)
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        try:
            env.reset(seed=0)
            snap0 = mgr.capture()
            env.step(Action(joint_positions=jnp.asarray([0.5, 0.0], dtype=jnp.float32)))
            snap1 = mgr.capture()
            self.assertGreater(snap1.step, snap0.step)
            mgr.restore(snap0)
            obs_map = env.get_processed_obs_by_robot()
            restored = np.asarray(obs_map["robot0"].joint_positions, dtype=np.float64)
            np.testing.assert_array_almost_equal(restored, np.asarray(snap0.obs.joint_positions, dtype=np.float64))
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                path = tmp.name
            mgr.save(path)
            mgr2 = SnapshotManager(env=env)
            mgr2.load(path)
            self.assertEqual(len(mgr2.snapshots), 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
