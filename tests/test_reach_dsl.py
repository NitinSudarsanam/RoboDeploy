from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ReachDSLTests(unittest.TestCase):
    def test_from_yaml_loads_phases(self):
        yaml = REPO_ROOT / "examples" / "policies" / "reach_pick_place.yaml"
        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        policy = ReachTrajectoryPolicy.from_yaml(yaml)
        self.assertGreater(len(policy._phases), 5)

    def test_get_action_returns_joint_positions(self):
        from robodeploy.core.types import Observation
        from robodeploy.policies.builder import PolicyBuilder

        policy = (
            PolicyBuilder()
            .add_settle_home(hold_steps=2)
            .add_reach_phase("pregrasp", target="source", offset=(0.0, 0.0, 0.1))
            .build()
        )
        policy.reset()
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        obs = Observation(
            joint_positions=jnp.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]),
            joint_velocities=jnp.zeros(7),
            joint_torques=jnp.zeros(7),
            ee_position=jnp.array([0.55, 0.0, 0.5]),
            ee_orientation=jnp.array([1.0, 0.0, 0.0, 0.0]),
            ee_velocity=jnp.zeros(3),
            ee_angular_velocity=jnp.zeros(3),
        )
        action = policy.get_action(obs)
        self.assertIsNotNone(action.joint_positions)

    def test_learned_phase_delegates_to_stub_policy(self):
        from robodeploy.core.types import Observation
        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        spec = {
            "home": [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0],
            "phases": [
                {
                    "name": "handoff",
                    "kind": "learned",
                    "policy": "vla_stub",
                    "instruction": "reach forward",
                    "max_steps": 3,
                }
            ],
        }
        policy = ReachTrajectoryPolicy(spec)
        policy.reset()
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]
        obs = Observation(
            joint_positions=jnp.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]),
            joint_velocities=jnp.zeros(7),
            joint_torques=jnp.zeros(7),
            ee_position=jnp.array([0.55, 0.0, 0.5]),
            ee_orientation=jnp.array([1.0, 0.0, 0.0, 0.0]),
            ee_velocity=jnp.zeros(3),
            ee_angular_velocity=jnp.zeros(3),
        )
        action = policy.get_action(obs)
        self.assertIsNotNone(action.joint_positions or action.ee_position)

    def test_gripper_phase_sets_gripper_command(self):
        from robodeploy.core.types import Observation
        from robodeploy.policies.builder import PolicyBuilder

        policy = (
            PolicyBuilder()
            .add_settle_home(hold_steps=1)
            .add_close_gripper(hold_steps=2)
            .build()
        )
        policy.reset()
        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        obs = Observation(
            joint_positions=jnp.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]),
            joint_velocities=jnp.zeros(7),
            joint_torques=jnp.zeros(7),
            ee_position=jnp.array([0.55, 0.0, 0.5]),
            ee_orientation=jnp.array([1.0, 0.0, 0.0, 0.0]),
            ee_velocity=jnp.zeros(3),
            ee_angular_velocity=jnp.zeros(3),
        )
        policy.get_action(obs)  # settle
        action = policy.get_action(obs)  # close_gripper
        self.assertEqual(action.gripper, 1.0)
