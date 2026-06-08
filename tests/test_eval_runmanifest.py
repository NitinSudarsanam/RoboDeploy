from __future__ import annotations

import unittest

from robodeploy.evaluation.report import EvalReport
from robodeploy.observability.manifest import RunManifest


class EvalRunManifestIntegrationTests(unittest.TestCase):
    def test_eval_report_manifest_is_run_manifest_not_duplicate(self):
        from robodeploy.evaluation.manifest import BenchmarkRunManifest

        self.assertIs(BenchmarkRunManifest, RunManifest)
        manifest = RunManifest.for_benchmark_eval(
            benchmark="x",
            benchmark_version="1",
            policy="p",
            backend="dummy",
            seed_base=0,
            n_episodes=1,
        )
        self.assertIsInstance(manifest, RunManifest)

    def test_benchmark_manifest_serializes_in_eval_report(self):
        from robodeploy.evaluation.harness import EvalConfig
        from robodeploy.evaluation.metrics import AggregateMetrics, EpisodeMetrics

        manifest = RunManifest.for_benchmark_eval(
            benchmark="manipulation_v1/reach_target",
            benchmark_version="1.0",
            policy="scripted",
            backend="dummy",
            seed_base=0,
            n_episodes=3,
            extra={"tier": 1},
        )
        report = EvalReport(
            benchmark_name="manipulation_v1/reach_target",
            benchmark_version="1.0",
            episodes=[],
            aggregate=AggregateMetrics(
                n_episodes=0,
                success_rate=0.0,
                success_rate_ci95=(0.0, 0.0),
                mean_reward=0.0,
                std_reward=0.0,
                median_time_to_success_steps=None,
                mean_smoothness_jerk=0.0,
                mean_smoothness_action_norm=0.0,
                mean_collision_count=0.0,
            ),
            config=EvalConfig(n_episodes=3),
            manifest=manifest,
            started_at=1.0,
            finished_at=2.0,
        )
        payload = report.to_json()
        self.assertEqual(payload["manifest"]["benchmark"], "manipulation_v1/reach_target")
        self.assertEqual(payload["manifest"]["n_episodes"], 3)
        self.assertEqual(payload["manifest"]["package_version"], manifest.package_version)
        self.assertNotIn("robodeploy_version", payload["manifest"])


if __name__ == "__main__":
    unittest.main()
