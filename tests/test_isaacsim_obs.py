from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from robodeploy.backends.sim.isaacsim.backend import IsaacSimBackend


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeRobot:
    def __init__(self):
        self._efforts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)

    def get_joint_positions(self):
        return np.zeros(7, dtype=np.float32)

    def get_joint_velocities(self):
        return np.zeros(7, dtype=np.float32)

    def get_measured_joint_efforts(self):
        return _FakeTensor(self._efforts)

    def get_link_index(self, name: str):
        del name
        return 3

    def get_link_world_poses(self, *, indices):
        del indices
        return types.SimpleNamespace(
            positions=_FakeTensor([[0.4, 0.1, 0.5]]),
            orientations=_FakeTensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    def get_link_velocities(self, *, indices):
        del indices
        return _FakeTensor([[0.01, 0.0, 0.0]]), _FakeTensor([[0.0, 0.0, 0.1]])


class _FakeDescription:
    ee_link_name = "robot0/ee_link"

    def ros_ee_frame_id(self):
        return "ee_link"


class _FakeArticulationView:
    def get_link_index(self, name: str):
        if name == "ee_link":
            return 5
        raise KeyError(name)

    def get_world_poses(self, *, indices):
        del indices
        return types.SimpleNamespace(
            positions=_FakeTensor([[0.3, 0.2, 0.6]]),
            orientations=_FakeTensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    def get_link_velocities(self, *, indices):
        del indices
        return _FakeTensor([[0.02, 0.0, 0.0]]), _FakeTensor([[0.0, 0.0, 0.2]])


class _FakeRobotViaArticulationView:
    _articulation_view = _FakeArticulationView()

    def get_joint_positions(self):
        return np.zeros(7, dtype=np.float32)

    def get_joint_velocities(self):
        return np.zeros(7, dtype=np.float32)


class IsaacSimObsTests(unittest.TestCase):
    def test_build_obs_reads_nonzero_ee_and_efforts(self):
        backend = IsaacSimBackend({})
        backend._robot = _FakeRobot()
        backend._description = _FakeDescription()
        backend._sim_time = 0.5
        backend._warnings = []

        obs = backend._build_obs()

        ee = np.asarray(obs.ee_position).reshape(-1)
        tau = np.asarray(obs.joint_torques).reshape(-1)
        self.assertGreater(float(np.linalg.norm(ee)), 0.0)
        self.assertGreater(float(np.linalg.norm(tau)), 0.0)

    def test_read_ee_state_uses_articulation_view_and_ros_link_name(self):
        backend = IsaacSimBackend({})
        backend._robot = _FakeRobotViaArticulationView()
        backend._description = _FakeDescription()
        backend._warnings = []

        ee_pos, _, ee_vel, ee_avel = backend._read_ee_state()

        self.assertAlmostEqual(float(ee_pos[0]), 0.3, places=4)
        self.assertAlmostEqual(float(ee_pos[1]), 0.2, places=4)
        self.assertAlmostEqual(float(ee_pos[2]), 0.6, places=4)
        self.assertGreater(float(np.linalg.norm(ee_vel)), 0.0)
        self.assertGreater(float(np.linalg.norm(ee_avel)), 0.0)

    def test_read_joint_efforts_fallback_warning(self):
        backend = IsaacSimBackend({})
        backend._robot = types.SimpleNamespace(get_joint_positions=lambda: np.zeros(3))
        backend._warnings = []
        qfrc = backend._read_joint_efforts(3)
        self.assertEqual(qfrc.shape, (3,))
        self.assertTrue(any("unavailable" in w for w in backend._warnings))

    def test_set_physics_params_damping_calls_apply_joint_damping(self):
        backend = IsaacSimBackend({})
        backend._warnings = []
        with unittest.mock.patch.object(backend, "_apply_joint_damping") as damping_mock:
            backend.set_physics_params(damping=12.5)
        damping_mock.assert_called_once_with(12.5)


if __name__ == "__main__":
    unittest.main()
