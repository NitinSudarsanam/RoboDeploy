from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase
from robodeploy.policies.remote.server import PolicyServer


class _EchoPolicy(PolicyBase):
    def __init__(self) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)

    def get_action(self, obs: Observation) -> Action:
        return Action(joint_positions=jnp.asarray(obs.joint_positions, dtype=jnp.float32))


class _MockTransport:
    def connect(self) -> None:
        return

    def close(self) -> None:
        return


class PolicyServerTests(unittest.TestCase):
    def test_infer_delegates_to_policy(self):
        obs = Observation(
            joint_positions=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            joint_velocities=jnp.zeros(2, dtype=jnp.float32),
            joint_torques=jnp.zeros(2, dtype=jnp.float32),
            ee_position=jnp.zeros(3, dtype=jnp.float32),
            ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            ee_velocity=jnp.zeros(3, dtype=jnp.float32),
            ee_angular_velocity=jnp.zeros(3, dtype=jnp.float32),
        )
        server = PolicyServer(policy=_EchoPolicy(), transport=_MockTransport(), verbose=False)
        action = server.infer(obs)
        self.assertAlmostEqual(float(action.joint_positions[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
