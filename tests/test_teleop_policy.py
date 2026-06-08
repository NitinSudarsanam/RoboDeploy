from __future__ import annotations

import unittest

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.teleop.base import ITeleopDevice, TeleopCommand, TeleopSafetyError
from robodeploy.teleop.controller import TeleopPolicy


class _ScriptedDevice(ITeleopDevice):
    def __init__(self, commands: list[TeleopCommand | None]) -> None:
        self._commands = list(commands)
        self._index = 0

    def start(self) -> None:
        return

    def poll(self) -> TeleopCommand | None:
        if self._index >= len(self._commands):
            return None
        cmd = self._commands[self._index]
        self._index += 1
        return cmd

    def stop(self) -> None:
        return


def _obs(q=None) -> Observation:
    q = np.asarray(q if q is not None else [0.1, 0.2], dtype=np.float32)
    return Observation(
        joint_positions=jnp.asarray(q, dtype=jnp.float32),
        joint_velocities=jnp.zeros_like(q),
        joint_torques=jnp.zeros_like(q),
        ee_position=jnp.asarray([0.5, 0.0, 0.4], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros(3, dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros(3, dtype=jnp.float32),
    )


class _MockIk:
    def solve(self, q_init: np.ndarray, target_pos: np.ndarray, **kwargs) -> np.ndarray:
        return np.asarray(q_init, dtype=np.float32) + np.array([0.05, 0.0], dtype=np.float32)

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        return np.array([0.5, 0.0, 0.4], dtype=np.float32)


class TeleopPolicyTests(unittest.TestCase):
    def test_joint_delta_command(self) -> None:
        device = _ScriptedDevice(
            [TeleopCommand(delta_joint_positions=np.array([0.01, -0.02], dtype=np.float32))]
        )
        policy = TeleopPolicy(device=device)
        action = policy.get_action(_obs())
        self.assertIsNotNone(action.joint_positions)
        arr = np.asarray(action.joint_positions, dtype=np.float32)
        self.assertAlmostEqual(float(arr[0]), 0.11, places=5)
        self.assertAlmostEqual(float(arr[1]), 0.18, places=5)

    def test_cartesian_delta_uses_ik(self) -> None:
        device = _ScriptedDevice(
            [TeleopCommand(delta_position=np.array([0.01, 0.0, 0.0], dtype=np.float32))]
        )
        policy = TeleopPolicy(device=device, ik_solver=_MockIk())
        action = policy.get_action(_obs())
        arr = np.asarray(action.joint_positions, dtype=np.float32)
        self.assertAlmostEqual(float(arr[0]), 0.15, places=5)

    def test_estop_raises(self) -> None:
        device = _ScriptedDevice([TeleopCommand(e_stop=True)])
        policy = TeleopPolicy(device=device)
        with self.assertRaises(TeleopSafetyError):
            policy.get_action(_obs())

    def test_hold_when_idle(self) -> None:
        device = _ScriptedDevice([None, None])
        policy = TeleopPolicy(device=device, default_action="hold")
        action = policy.get_action(_obs([0.3, 0.4]))
        arr = np.asarray(action.joint_positions, dtype=np.float32)
        self.assertAlmostEqual(float(arr[0]), 0.3, places=5)

    def test_registered_name(self) -> None:
        from robodeploy.core.registry import get_policy, use

        use("robodeploy.teleop.controller")
        cls = get_policy("teleop")
        self.assertIs(cls, TeleopPolicy)


if __name__ == "__main__":
    unittest.main()
