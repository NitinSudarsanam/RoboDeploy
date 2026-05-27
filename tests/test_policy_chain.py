from __future__ import annotations

import unittest

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase
from robodeploy.policies.composition import PolicyChain


def make_obs() -> Observation:
    return Observation(
        joint_positions=np.zeros(2, dtype=np.float32),
        joint_velocities=np.zeros(2, dtype=np.float32),
        joint_torques=np.zeros(2, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


class _ConstPolicy(PolicyBase):
    def __init__(self, values: list[float]) -> None:
        super().__init__(action_space=ActionSpace.JOINT_POS)
        self._values = values

    def get_action(self, obs: Observation) -> Action:
        del obs
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]
        return Action(joint_positions=jnp.asarray(self._values, dtype=jnp.float32))


class PolicyChainTests(unittest.TestCase):
    def test_refine_mode_keeps_last_non_none_fields(self):
        chain = PolicyChain(
            policies=[
                _ConstPolicy([1.0, 1.0]),
                _ConstPolicy([2.0, 3.0]),
            ],
            config={"mode": "refine"},
        )
        action = chain.get_action(make_obs())
        self.assertAlmostEqual(float(action.joint_positions[0]), 2.0)
        self.assertAlmostEqual(float(action.joint_positions[1]), 3.0)

    def test_last_mode_uses_final_policy_only(self):
        chain = PolicyChain(
            policies=[
                _ConstPolicy([9.0, 9.0]),
                _ConstPolicy([4.0, 5.0]),
            ],
            config={"mode": "last"},
        )
        action = chain.get_action(make_obs())
        self.assertAlmostEqual(float(action.joint_positions[0]), 4.0)


if __name__ == "__main__":
    unittest.main()
