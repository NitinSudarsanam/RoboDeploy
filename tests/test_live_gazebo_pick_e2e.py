"""Live + offline Gazebo pick-place E2E (WAVE2_01).

PR offline gate: ``pytest tests/test_live_gazebo_pick_e2e.py -k offline -q``
Live CI: ``ROBODEPLOY_LIVE_GAZEBO=1 pytest tests/test_live_gazebo_pick_e2e.py -m live_gazebo -q``

Flake policy: pytest-rerunfailures max 2 on live pick test; quarantine if <50% over 7 days.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = REPO_ROOT / "tests"
for _p in (str(REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_sensor_fixtures import gazebo_binary_available, rclpy_available

from examples.kuka_ft_imu_pick_gazebo.pick_episode import (
    LIVE_PICK_MIN_SUCCESS_RATE,
    LIVE_PICK_SEEDS,
    PICK_MINIMAL_WORLD,
    kuka_ft_imu_pick_gazebo_cfg,
    run_pick_episode,
    source_to_goal_distance,
)
from robodeploy.backends.sim.gazebo.contact import GazeboContactMonitor

LIVE = os.environ.get("ROBODEPLOY_LIVE_GAZEBO", "").strip() in {"1", "true", "yes"}


class GazeboPickOfflineTests(unittest.TestCase):
    def test_pick_minimal_world_fixture_exists(self):
        self.assertTrue(PICK_MINIMAL_WORLD.is_file())

    def test_kuka_ft_imu_pick_gazebo_cfg_offline(self):
        cfg = kuka_ft_imu_pick_gazebo_cfg(max_episode_steps=10)
        self.assertEqual(cfg["backend"], "ros2_gazebo")
        world = cfg["backend_kwargs"]["config"]["sim"]["world"]
        self.assertIn(Path(world).name, ("pick_minimal.sdf", "gazebo_pick_minimal.sdf"))

    def test_injected_contact_matches_grasp_query_offline(self):
        monitor = GazeboContactMonitor()
        monitor.inject_contacts([("robot0::source::collision", "robot0::ee_link::collision")])
        self.assertTrue(monitor.has_contact("source", "ee_link"))
        self.assertTrue(monitor.has_contact("source", "robot0/ee_link"))

    def test_source_to_goal_distance_offline(self):
        import jax.numpy as jnp

        from examples.tasks.pick_place import PickPlaceTask
        from robodeploy.core.types import Observation

        task = PickPlaceTask(config={"require_objects": True})
        z = jnp.zeros
        obs = Observation(
            joint_positions=z(7),
            joint_velocities=z(7),
            joint_torques=z(7),
            ee_position=z(3),
            ee_orientation=jnp.array([1.0, 0.0, 0.0, 0.0]),
            ee_velocity=z(3),
            ee_angular_velocity=z(3),
            objects={
                "source": ((0.60, 0.20, 0.41), (1.0, 0.0, 0.0, 0.0)),
            },
        )
        dist = source_to_goal_distance(obs, task)
        self.assertIsNotNone(dist)
        assert dist is not None
        self.assertLess(dist, 0.05)


@unittest.skipUnless(LIVE, "set ROBODEPLOY_LIVE_GAZEBO=1 to run live Gazebo pick E2E")
@unittest.skipUnless(gazebo_binary_available(), "gz binary not on PATH")
@unittest.skipUnless(rclpy_available(), "rclpy not available")
class LiveGazeboPickE2ETests(unittest.TestCase):
    @pytest.mark.live_gazebo
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_kuka_ft_imu_pick_gazebo_success_rate(self):
        """Live: relaxed tuning; ≥50% success over 10 seeds (WAVE2 target 70%)."""
        results = [
            run_pick_episode(seed=seed, max_steps=1200) for seed in LIVE_PICK_SEEDS
        ]
        successes = [r for r in results if r.success]
        rate = len(successes) / len(LIVE_PICK_SEEDS)
        min_required = int(len(LIVE_PICK_SEEDS) * LIVE_PICK_MIN_SUCCESS_RATE + 0.999)

        for result in successes:
            self.assertTrue(result.sensor_health_ok, msg=f"seed {result.seed} sensor health failed")
            self.assertIsNotNone(
                result.source_to_goal_distance,
                msg=f"seed {result.seed} missing source pose for placement check",
            )
            assert result.source_to_goal_distance is not None
            self.assertLess(
                result.source_to_goal_distance,
                0.04,
                msg=f"seed {result.seed} place distance {result.source_to_goal_distance:.4f}",
            )

        if successes:
            contact_hits = sum(1 for r in successes if r.contact_during_grasp)
            self.assertGreaterEqual(
                contact_hits,
                1,
                msg=(
                    "expected wrist_contact or has_prop_contact during grasp on ≥1 successful "
                    f"episode; got {contact_hits}/{len(successes)}"
                ),
            )

        self.assertGreaterEqual(
            len(successes),
            min_required,
            msg=(
                f"Gazebo pick-place: {len(successes)}/{len(LIVE_PICK_SEEDS)} seeds succeeded "
                f"({rate:.0%}); need ≥{LIVE_PICK_MIN_SUCCESS_RATE:.0%} "
                "(WAVE2_01 target 70% pending JTC/IK tuning)"
            ),
        )


if __name__ == "__main__":
    unittest.main()
