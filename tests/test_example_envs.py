from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ExampleEnvTests(unittest.TestCase):
    def test_dummy_pick_place_runs(self):
        from examples.dummy_pick_place.run import main

        main()

    def test_reach_policy_carry_mode_none_skips_prop_teleport(self):
        from examples.policies.reach_pick_place import ReachPickPlacePolicy

        class _Backend:
            calls = 0

            def set_prop_pose(self, name, pos, quat):
                _Backend.calls += 1

        policy = ReachPickPlacePolicy(carry_mode="none")
        policy._backend = _Backend()
        policy._carrying = True
        policy._kinematic_carry = False
        policy._sync_carried_object(np.zeros(3, dtype=np.float32))
        self.assertEqual(_Backend.calls, 0)

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
        from examples.env_from_preset import env_from_preset

        env = env_from_preset("kuka_pick_mujoco")
        try:
            _, info = env.reset()
            env.step()
            self.assertEqual(info.episode_id, 1)
        finally:
            env.close()

    def test_reach_policy_completes_pick_place_when_mujoco_installed(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from examples.env_from_preset import env_from_preset

        env = env_from_preset("kuka_pick_mujoco", max_episode_steps=1500)
        try:
            env.reset()
            for _ in range(1500):
                _, _, done, info = env.step()
                if done:
                    break
            self.assertTrue(bool(info.success))
        finally:
            env.close()

    def test_sensor_preset_pick_succeeds_when_mujoco_installed(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from examples.env_from_preset import env_from_preset

        env = env_from_preset("kuka_sensor_pick_mujoco", max_episode_steps=1500)
        try:
            env.reset()
            self.assertEqual(len(env.robots[0].sensors), 1)
            info = None
            for _ in range(1500):
                _, _, done, info = env.step()
                if done:
                    break
            self.assertTrue(bool(info.success))
        finally:
            env.close()

    def test_kuka_vision_pick_mujoco_objects_from_color_blob_when_mujoco_installed(self):
        import sys

        if sys.platform == "win32":
            self.skipTest("MuJoCo GLFW Renderer unstable on Windows")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from robodeploy.sensors.camera.sim.mujoco_gl import ensure_mujoco_gl_backend
        from examples.env_from_preset import env_from_preset

        ensure_mujoco_gl_backend()
        try:
            env = env_from_preset("kuka_vision_pick_mujoco", max_episode_steps=10)
        except OSError as exc:
            self.skipTest(f"MuJoCo Renderer unavailable headless: {exc}")
        try:
            try:
                obs, _info = env.reset()
            except OSError as exc:
                self.skipTest(f"MuJoCo Renderer unavailable headless: {exc}")
            self.assertIn("source", obs.objects, "color blob transform should populate source pose")
            pos, _quat = obs.objects["source"]
            self.assertEqual(len(pos), 3)
            self.assertGreater(pos[2], 0.0)
            self.assertIn("target", obs.objects, "prop_pose sensor should populate target pose")
        finally:
            env.close()

    def test_vision_preset_pick_succeeds_when_mujoco_installed(self):
        import sys

        if sys.platform == "win32":
            self.skipTest("MuJoCo GLFW Renderer unstable on Windows")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from robodeploy.sensors.camera.sim.mujoco_gl import ensure_mujoco_gl_backend
        from examples.env_from_preset import env_from_preset

        ensure_mujoco_gl_backend()
        try:
            env = env_from_preset("kuka_vision_pick_mujoco", max_episode_steps=1500)
        except OSError as exc:
            self.skipTest(f"MuJoCo Renderer unavailable headless: {exc}")
        try:
            try:
                env.reset()
            except OSError as exc:
                self.skipTest(f"MuJoCo Renderer unavailable headless: {exc}")
            info = None
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
        from examples.env_from_preset import env_from_preset

        env = env_from_preset("kuka_pick_mujoco", max_episode_steps=10)
        try:
            obs, info = env.reset()
            self.assertIsNotNone(obs)
            self.assertEqual(info.episode_id, 1)
            env.step()
        finally:
            env.close()

    def test_two_franka_pick_mujoco_runs_when_mujoco_installed(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from examples.env_from_preset import env_from_preset

        env = env_from_preset("two_franka_pick_mujoco", max_episode_steps=80)
        try:
            env.reset()
            self.assertEqual(len(env.robots), 2)
            obs_map = env.get_processed_obs_by_robot()
            self.assertEqual(set(obs_map), {"franka_left", "franka_right"})
            home_left_q0 = float(obs_map["franka_left"].joint_positions[0])
            home_right_q0 = float(obs_map["franka_right"].joint_positions[0])
            for _ in range(80):
                _, _, done, _info = env.step()
                if done:
                    break
            obs_map = env.get_processed_obs_by_robot()
            left_q0 = float(obs_map["franka_left"].joint_positions[0])
            right_q0 = float(obs_map["franka_right"].joint_positions[0])
            self.assertGreater(left_q0, home_left_q0, "left arm should move toward +q0 target")
            self.assertLess(right_q0, home_right_q0, "right arm should move toward -q0 target")
            self.assertGreater(left_q0, right_q0, "arms should diverge to independent targets")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
