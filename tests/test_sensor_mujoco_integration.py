"""MuJoCo end-to-end sensor integration for FT/IMU/contact presets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


def _skip_on_windows() -> str | None:
    if sys.platform == "win32":
        return "MuJoCo integration tests skipped on Windows"
    return None


class SensorMuJoCoIntegrationTests(unittest.TestCase):
    def test_ft_imu_pick_obs_populates_sensor_fields(self):
        reason = _skip_on_windows()
        if reason:
            self.skipTest(reason)
        if not _mujoco_available():
            self.skipTest("mujoco not installed")

        from examples.env_from_preset import env_from_preset

        env = env_from_preset("kuka_ft_imu_pick_mujoco", max_episode_steps=50)
        try:
            obs, info = env.reset()
            self.assertIsNotNone(obs.ft_force)
            self.assertIsNotNone(obs.imu_angular_velocity)
            self.assertIsInstance(getattr(obs, "contact_state", None), dict)
            self.assertIsInstance(getattr(obs, "sensor_status", None), dict)
            self.assertIn("sensor_status", info.extra)

            obs, _, _, info = env.step()
            self.assertIn("wrist_ft", obs.sensor_status or {})
            self.assertIn("wrist_imu", obs.sensor_status or {})
            self.assertIn("sensor_health", info.extra)
        finally:
            env.close()

    def test_ft_grasp_policy_avoids_backend_contact_query(self):
        reason = _skip_on_windows()
        if reason:
            self.skipTest(reason)
        if not _mujoco_available():
            self.skipTest("mujoco not installed")

        from examples.env_from_preset import env_from_preset

        env = env_from_preset("kuka_ft_imu_pick_mujoco", max_episode_steps=30)
        try:
            env.reset()

            def _forbidden_contact(*_args, **_kwargs):
                raise AssertionError("FT grasp must not call backend.has_prop_contact")

            env.backend.has_prop_contact = mock.Mock(side_effect=_forbidden_contact)

            for _ in range(20):
                env.step()
        finally:
            env.close()

    def test_kuka_ft_imu_pick_success_rate_across_seeds(self):
        reason = _skip_on_windows()
        if reason:
            self.skipTest(reason)
        if not _mujoco_available():
            self.skipTest("mujoco not installed")

        from examples.env_from_preset import env_from_preset

        seeds = list(range(10))
        successes = 0
        for seed in seeds:
            env = env_from_preset("kuka_ft_imu_pick_mujoco", max_episode_steps=1500)
            try:
                env.reset(seed=seed)
                info = None
                for _ in range(1500):
                    _, _, done, info = env.step()
                    if done:
                        break
                if info is not None and bool(info.success):
                    successes += 1
            finally:
                env.close()

        rate = successes / len(seeds)
        self.assertGreaterEqual(
            rate,
            0.80,
            msg=f"success rate {rate:.0%} ({successes}/{len(seeds)}) below 80%",
        )


if __name__ == "__main__":
    unittest.main()
