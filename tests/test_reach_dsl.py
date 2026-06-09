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

    def test_drop_detection_rewinds_to_grasp_phase(self):
        from robodeploy.core.types import Observation
        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        spec = ReachTrajectoryPolicy.default_pick_place_spec()
        policy = ReachTrajectoryPolicy(
            spec,
            config={"grasp_detection": "ft", "grasp_force_loss_threshold": 1.0},
        )
        policy.reset()
        grasp_idx = policy._grasp_phase_idx()
        self.assertIsNotNone(grasp_idx)
        lift_idx = next(i for i, p in enumerate(policy._phases) if p.spec.name == "lift")
        policy._phase_idx = lift_idx
        policy._carrying = True
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
            ft_force=jnp.array([0.1, 0.0, 0.0]),
        )
        policy._maybe_drop_carry(obs)
        self.assertFalse(policy._carrying)
        close_idx = next(i for i, p in enumerate(policy._phases) if p.spec.name == "close_gripper")
        self.assertEqual(policy._phase_idx, close_idx)
        self.assertEqual(policy._phase_step, 0)
        self.assertEqual(policy._gripper_state, 1.0)

    def test_bind_runtime_falls_through_to_pin_when_mujoco_fails(self):
        from unittest.mock import MagicMock, patch

        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        spec = ReachTrajectoryPolicy.default_pick_place_spec()
        policy = ReachTrajectoryPolicy(spec)
        backend = MagicMock()
        backend._model = object()
        desc = MagicMock()
        with patch(
            "robodeploy.kinematics.mujoco_ik.attach_mujoco_ik",
            side_effect=RuntimeError("mujoco ik failed"),
        ):
            with patch("robodeploy.kinematics.pin_ik.attach_pin_ik") as pin_attach:
                pin_attach.return_value = MagicMock()
                policy.bind_runtime(backend, desc)
                pin_attach.assert_called_once()

    def test_gazebo_weld_carry_maps_to_follow(self):
        from unittest.mock import MagicMock

        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        spec = ReachTrajectoryPolicy.default_pick_place_spec()
        spec["carry"] = {"mode": "weld"}
        policy = ReachTrajectoryPolicy(spec)
        backend = MagicMock()
        backend.sensor_backend_name = "gazebo"
        policy.bind_runtime(backend, MagicMock())
        self.assertTrue(policy._backend_follow_carry)
        self.assertFalse(policy._backend_weld_carry)

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
