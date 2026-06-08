from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.transforms import GaussianNoiseTransform, LatencyTransform
from robodeploy.core.types import Action, Observation
from robodeploy.env import RoboEnv
from robodeploy.evaluation.transfer_metrics import (
    TransferEvaluator,
    compute_trajectory_distance,
)
from robodeploy.obs_pipeline import ObsPipeline
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask, make_obs


class LatencyTransformTests(unittest.TestCase):
    def test_latency_transform_delays_by_n_steps(self):
        tx = LatencyTransform(latency_steps=2, seed=0)
        o0 = make_obs(0.0)
        o1 = make_obs(1.0)
        o2 = make_obs(2.0)
        self.assertEqual(float(tx.forward(o0).joint_positions[0]), 0.0)
        self.assertEqual(float(tx.forward(o1).joint_positions[0]), 0.0)
        self.assertEqual(float(tx.forward(o2).joint_positions[0]), 0.0)
        o3 = make_obs(3.0)
        self.assertEqual(float(tx.forward(o3).joint_positions[0]), 1.0)


class TransferMetricsTests(unittest.TestCase):
    def test_compute_trajectory_distance(self):
        sim = [{"joint_positions": np.array([0.0, 0.0]), "ee_position": np.array([0.0, 0.0, 0.0]),
                "ee_orientation": np.array([1.0, 0.0, 0.0, 0.0])}]
        real = [{"joint_positions": np.array([0.1, 0.0]), "ee_position": np.array([0.0, 0.1, 0.0]),
                 "ee_orientation": np.array([1.0, 0.0, 0.0, 0.0])}]
        dist = compute_trajectory_distance(sim, real)
        self.assertAlmostEqual(dist["joint_pos_l2"], 0.1, places=5)
        self.assertAlmostEqual(dist["ee_pos_l2"], 0.1, places=5)

    def _dummy_env(self, seed: int):
        del seed
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        return RoboEnv(backend=DummyBackend(), robots=[robot])

    def _noisy_dummy_env(self, seed: int):
        del seed
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
            obs_pipeline=ObsPipeline([GaussianNoiseTransform(joint_pos_std=0.05, seed=1)]),
        )
        return RoboEnv(backend=DummyBackend(), robots=[robot])

    def test_transfer_evaluator_sim_vs_noisy_sim_proxy(self):
        evaluator = TransferEvaluator(
            sim_env_fn=self._dummy_env,
            real_env_fn=self._noisy_dummy_env,
            policy_fn=lambda _env: lambda obs: Action(
                joint_positions=np.asarray(obs.joint_positions, dtype=np.float32)
            ),
            n_episodes=2,
            max_steps_per_episode=5,
        )
        metrics = evaluator.run()
        self.assertEqual(len(metrics.per_episode_breakdown), 2)
        self.assertGreaterEqual(metrics.obs_distribution_kl.get("joint_positions", 0.0), 0.0)
        self.assertIn("joint_positions", metrics.obs_distribution_kl)

    def test_render_report_writes_json(self):
        evaluator = TransferEvaluator(
            sim_env_fn=self._dummy_env,
            real_env_fn=self._dummy_env,
            policy_fn=lambda _env: lambda obs: None,
            n_episodes=1,
            max_steps_per_episode=2,
        )
        evaluator.run()
        with tempfile.TemporaryDirectory() as tmp:
            path = evaluator.render_report(Path(tmp))
            self.assertTrue(path.is_file())
            self.assertIn("sim_success_rate", path.read_text(encoding="utf-8"))
            manifest = Path(tmp) / "manifest.json"
            self.assertTrue(manifest.is_file())
            self.assertTrue((Path(tmp) / "transfer_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
