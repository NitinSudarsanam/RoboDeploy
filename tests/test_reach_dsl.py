from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _obs_at_ee(ee):
    import numpy as np

    from robodeploy.core.types import Observation

    ee = np.asarray(ee, dtype=np.float32).reshape(3)
    return Observation(
        joint_positions=np.zeros(7, dtype=np.float32),
        joint_velocities=np.zeros(7, dtype=np.float32),
        joint_torques=np.zeros(7, dtype=np.float32),
        ee_position=ee,
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


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

    def test_gazebo_place_snap_respects_env_flag(self):
        import os
        from unittest.mock import MagicMock, patch

        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        spec = ReachTrajectoryPolicy.default_pick_place_spec()
        policy = ReachTrajectoryPolicy(spec)
        backend = MagicMock()
        backend.sensor_backend_name = "gazebo"
        backend._scene_prop_poses = {
            "target": ((0.6, 0.2, 0.38), (1.0, 0.0, 0.0, 0.0)),
            "source": ((0.55, 0.0, 0.38), (1.0, 0.0, 0.0, 0.0)),
        }
        policy.bind_runtime(backend, MagicMock())
        with patch.dict(os.environ, {"ROBODEPLOY_GAZEBO_PLACE_SNAP": "0"}):
            policy._maybe_finalize_kinematic_place(0.5)
        backend.set_prop_pose.assert_not_called()
        with patch.dict(os.environ, {"ROBODEPLOY_GAZEBO_PLACE_SNAP": "1"}):
            policy._maybe_finalize_kinematic_place(0.5)
        backend.set_prop_pose.assert_called_once()

    def test_ros2_rviz_place_finalize_does_not_snap_source(self):
        """RViz fake-sim places honestly (no oracle snap) since the pin IK fix."""
        from unittest.mock import MagicMock

        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        spec = ReachTrajectoryPolicy.default_pick_place_spec()
        policy = ReachTrajectoryPolicy(spec)
        backend = MagicMock()
        backend.sensor_backend_name = "ros2_rviz"
        backend._scene_prop_poses = {
            "target": ((0.6, 0.2, 0.38), (1.0, 0.0, 0.0, 0.0)),
            "source": ((0.55, 0.0, 0.38), (1.0, 0.0, 0.0, 0.0)),
        }
        policy.bind_runtime(backend, MagicMock())
        policy._maybe_finalize_kinematic_place(0.2)
        backend.set_prop_pose.assert_not_called()

    def test_gazebo_place_phase_snap_without_carry(self):
        import os

        import numpy as np
        from unittest.mock import MagicMock, patch

        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        spec = ReachTrajectoryPolicy.default_pick_place_spec()
        policy = ReachTrajectoryPolicy(
            spec,
            config={"sensor_only": True, "carry_mode": "kinematic"},
        )
        backend = MagicMock()
        backend.sensor_backend_name = "gazebo"
        backend._scene_prop_poses = {
            "target": ((0.6, 0.2, 0.38), (1.0, 0.0, 0.0, 0.0)),
            "source": ((0.55, 0.0, 0.38), (1.0, 0.0, 0.0, 0.0)),
        }
        policy.bind_runtime(backend, MagicMock())
        policy._waypoints = {
            "source": np.array([0.55, 0.0, 0.405], dtype=np.float32),
            "target": np.array([0.6, 0.2, 0.405], dtype=np.float32),
        }
        policy._recompile_phases()
        place_idx = next(i for i, p in enumerate(policy._phases) if p.spec.name == "place")
        policy._phase_idx = place_idx
        policy._phase_step = policy._phases[place_idx].max_steps
        policy._carrying = False
        from dataclasses import replace

        obs = replace(
            _obs_at_ee([0.6, 0.2, 0.5]),
            objects={
                "source": ([0.55, 0.0, 0.38], [1.0, 0.0, 0.0, 0.0]),
                "target": ([0.6, 0.2, 0.38], [1.0, 0.0, 0.0, 0.0]),
            },
        )
        with patch.dict(os.environ, {"ROBODEPLOY_GAZEBO_PLACE_SNAP": "1"}):
            policy.get_action(obs)
        backend.set_prop_pose.assert_called()

    def test_honest_place_defers_release_until_within_tolerance(self):
        import os

        import numpy as np
        from unittest.mock import MagicMock, patch

        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        spec = ReachTrajectoryPolicy.default_pick_place_spec()
        policy = ReachTrajectoryPolicy(
            spec,
            config={"honest_place_settle_m": 0.03, "steps_per_phase": 40},
        )
        backend = MagicMock()
        backend.sensor_backend_name = "gazebo"
        policy.bind_runtime(backend, MagicMock())
        policy._carrying = True
        place_idx = next(
            i for i, ph in enumerate(policy._phases) if ph.spec.name == "place"
        )
        policy._phase_idx = place_idx
        policy._phase_step = 10
        place = policy._phases[place_idx]
        assert place.ee_target is not None
        far_ee = place.ee_target + np.array([0.2, 0.0, 0.0], dtype=np.float32)
        with patch.dict(os.environ, {"ROBODEPLOY_GAZEBO_PLACE_SNAP": "0"}):
            self.assertFalse(policy._place_phase_release_allowed(place, _obs_at_ee(far_ee)))
            self.assertTrue(
                policy._place_phase_release_allowed(place, _obs_at_ee(place.ee_target))
            )

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

    def test_fallback_delta_tracks_last_valid_q_not_home(self):
        import numpy as np

        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        policy = ReachTrajectoryPolicy(
            {
                "home": [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0],
                "phases": [{"name": "reach", "kind": "reach", "target": "source"}],
            }
        )
        q = np.array([0.4, -0.4, 0.1, -1.5, 0.0, 1.0, 0.0], dtype=np.float32)
        policy._q_goal = q.copy()
        target = np.array([0.6, 0.2, 0.5], dtype=np.float32)

        def _fresh_tracker() -> None:
            # _track_toward integrates internal command state; reset between
            # calls so each call is compared from the same starting state.
            policy._q_cmd = None
            policy._track_bias = np.zeros_like(policy._home)

        _fresh_tracker()
        toward_home = policy._track_toward(q, policy._home)
        _fresh_tracker()
        result = policy._fallback_delta(q, target)
        self.assertFalse(np.allclose(result, toward_home, atol=1e-4))
        _fresh_tracker()
        expected = policy._track_toward(q, q)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-5)
