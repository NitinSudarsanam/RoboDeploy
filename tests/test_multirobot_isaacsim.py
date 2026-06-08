from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from robodeploy.backends.sim.isaacsim.backend import IsaacSimBackend, _IsaacRobotRuntime
from robodeploy.core.types import Action, Observation


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeRobot:
    def __init__(self, *, offset: float = 0.0):
        self._offset = float(offset)
        self._efforts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)

    def get_joint_positions(self):
        return np.full(7, self._offset, dtype=np.float32)

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
            positions=_FakeTensor([[0.4 + self._offset, 0.1, 0.5]]),
            orientations=_FakeTensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    def get_link_velocities(self, *, indices):
        del indices
        return _FakeTensor([[0.01, 0.0, 0.0]]), _FakeTensor([[0.0, 0.0, 0.1]])

    def apply_action(self, action) -> None:
        del action

    def initialize(self) -> None:
        return


class _FakeDescription:
    ee_link_name = "robot0/ee_link"
    home_qpos = [0.0] * 7

    def ros_base_frame_id(self):
        return "base_link"


class _FakeRobotEntry:
    def __init__(self, robot_id: str, *, offset: float = 0.0):
        self.robot_id = robot_id
        self.description = _FakeDescription()
        self.sensors = []


class IsaacSimMultiRobotTests(unittest.TestCase):
    def _backend_with_two_robots(self) -> IsaacSimBackend:
        backend = IsaacSimBackend({})
        backend._initialized = True
        backend._multi_mode = True
        backend._robot_order = ["robot_a", "robot_b"]
        backend._warnings = []
        backend._sim_time = 0.0
        backend._steps_per_control = 1
        backend._physics_dt = 1.0 / 60.0
        backend._isaac = types.SimpleNamespace(ArticulationAction=lambda **kwargs: kwargs)
        backend._world = mock.Mock()
        backend._simulation_app = mock.Mock()
        backend._robot_runtimes = {
            "robot_a": _IsaacRobotRuntime(
                robot_id="robot_a",
                description=_FakeDescription(),
                articulation=_FakeRobot(offset=0.1),
                prim_path="/World/robot_a",
                sensors=[],
            ),
            "robot_b": _IsaacRobotRuntime(
                robot_id="robot_b",
                description=_FakeDescription(),
                articulation=_FakeRobot(offset=-0.1),
                prim_path="/World/robot_b",
                sensors=[],
            ),
        }
        return backend

    def test_get_obs_multi_returns_one_obs_per_robot(self):
        backend = self._backend_with_two_robots()
        obs_list = backend.get_obs_multi()
        self.assertEqual(len(obs_list), 2)
        self.assertGreater(float(obs_list[0].ee_position[0]), float(obs_list[1].ee_position[0]))

    def test_step_multi_dispatches_per_robot_actions(self):
        backend = self._backend_with_two_robots()
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        actions = [
            Action(joint_positions=jnp.asarray([0.2] * 7, dtype=jnp.float32)),
            Action(joint_positions=jnp.asarray([-0.2] * 7, dtype=jnp.float32)),
        ]
        obs_list = backend.step_multi(actions)
        self.assertEqual(len(obs_list), 2)
        self.assertAlmostEqual(float(obs_list[0].joint_positions[0]), 0.1, places=3)
        self.assertAlmostEqual(float(obs_list[1].joint_positions[0]), -0.1, places=3)

    def test_initialize_multi_single_robot_uses_legacy_path(self):
        backend = IsaacSimBackend({})
        robot = _FakeRobotEntry("robot0")
        with mock.patch(
            "robodeploy.backends.base.BackendBase.initialize",
            return_value=None,
        ) as init_mock:
            backend.initialize_multi([robot], mock.Mock(), [])
        init_mock.assert_called_once()
        self.assertFalse(getattr(backend, "_multi_mode", True))


if __name__ == "__main__":
    unittest.main()
