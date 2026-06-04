from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ExampleEnvTests(unittest.TestCase):
    def test_dummy_pick_place_runs(self):
        from examples.dummy_pick_place.run import main

        main()

    def test_reach_policy_produces_joint_actions(self):
        from robodeploy.core.types import Observation

        try:
            import jax.numpy as jnp
        except Exception:
            import numpy as jnp  # type: ignore[assignment]

        from examples.policies.reach_pick_place import ReachPickPlacePolicy

        policy = ReachPickPlacePolicy()
        policy.reset()
        obs = Observation(
            joint_positions=jnp.asarray([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=jnp.float32),
            joint_velocities=jnp.zeros((7,), dtype=jnp.float32),
            joint_torques=jnp.zeros((7,), dtype=jnp.float32),
            ee_position=jnp.asarray([0.3, 0.0, 0.5], dtype=jnp.float32),
            ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
            ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        )
        action = policy.get_action(obs)
        self.assertIsNotNone(action.joint_positions)
        self.assertEqual(int(action.joint_positions.shape[0]), 7)

    def test_env_from_preset_kuka_pick_when_mujoco_installed(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from examples.env_from_preset import env_from_preset, wire_mujoco_pick_policies

        env = env_from_preset("kuka_pick_mujoco")
        try:
            _, info = env.reset()
            wire_mujoco_pick_policies(env)
            env.step()
            self.assertEqual(info.episode_id, 1)
        finally:
            env.close()

    def test_reach_policy_completes_pick_place_when_mujoco_installed(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from examples.kuka_pick_place_mujoco.run_mujoco import _attach_policy_ik, build_env

        env = build_env(max_steps=1500)
        try:
            env.reset()
            _attach_policy_ik(env)
            for _ in range(1500):
                _, _, done, info = env.step()
                if done:
                    break
            self.assertTrue(bool(info.success))
        finally:
            env.close()

    def test_kuka_pick_mujoco_builds_when_mujoco_installed(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from examples.kuka_pick_place_mujoco.run_mujoco import _attach_policy_ik, build_env

        env = build_env(max_steps=10)
        try:
            obs, info = env.reset()
            _attach_policy_ik(env)
            self.assertIsNotNone(obs)
            self.assertEqual(info.episode_id, 1)
            env.step()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
