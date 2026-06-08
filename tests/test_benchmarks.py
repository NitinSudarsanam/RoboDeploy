from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkRegistryTests(unittest.TestCase):
    def test_registry_discovers_manipulation_v1(self):
        from robodeploy.evaluation.registry import BenchmarkRegistry

        registry = BenchmarkRegistry(REPO_ROOT / "benchmarks")
        suites = registry.list_suites()
        self.assertIn("manipulation_v1", suites)
        suite = registry.load_suite("manipulation_v1")
        names = {task.name for task in suite.tasks}
        self.assertIn("reach_target", names)
        self.assertIn("pick_place_cube", names)
        self.assertIn("pour_into_cup", names)
        self.assertIn("tools_use_screw", names)
        self.assertEqual(len(suite.tasks), 8)

    def test_gazebo_isaacsim_presets_discovered(self):
        from robodeploy.evaluation.registry import BenchmarkRegistry

        registry = BenchmarkRegistry(REPO_ROOT / "benchmarks")
        task = registry.load_suite("manipulation_v1").get_task("reach_target")
        backends = task.available_backends()
        self.assertIn("gazebo", backends)
        self.assertIn("isaacsim", backends)

    def test_resolve_single_task(self):
        from robodeploy.evaluation.registry import BenchmarkRegistry

        registry = BenchmarkRegistry(REPO_ROOT / "benchmarks")
        suite, task = registry.resolve("manipulation_v1/reach_target")
        self.assertEqual(suite.name, "manipulation_v1")
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.qualified_name, "manipulation_v1/reach_target")
        self.assertIn("dummy", task.available_backends())


class MetricsTests(unittest.TestCase):
    def test_ci95_bounds(self):
        from robodeploy.evaluation.metrics import ci95_binomial

        lo, hi = ci95_binomial([True] * 10)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLessEqual(lo, hi)

    def test_aggregate_episodes_success_rate(self):
        from robodeploy.evaluation.metrics import EpisodeMetrics, aggregate_episodes

        metrics = [
            EpisodeMetrics(
                success=True,
                reward_total=1.0,
                reward_per_step=1.0,
                steps=1,
                time_to_success_steps=1,
                time_to_success_seconds=0.1,
                smoothness_jerk=0.0,
                smoothness_action_norm=0.0,
                smoothness_velocity=0.0,
                collision_count=0,
                max_force_N=0.0,
                workspace_violations=0,
                distance_to_goal_final=0.0,
                distance_to_goal_min=0.0,
                constraint_violations={},
            ),
            EpisodeMetrics(
                success=False,
                reward_total=0.0,
                reward_per_step=0.0,
                steps=2,
                time_to_success_steps=None,
                time_to_success_seconds=None,
                smoothness_jerk=0.0,
                smoothness_action_norm=0.0,
                smoothness_velocity=0.0,
                collision_count=0,
                max_force_N=0.0,
                workspace_violations=0,
                distance_to_goal_final=1.0,
                distance_to_goal_min=0.5,
                constraint_violations={},
            ),
        ]
        agg = aggregate_episodes(metrics)
        self.assertEqual(agg.n_episodes, 2)
        self.assertAlmostEqual(agg.success_rate, 0.5)


class EvalHarnessTests(unittest.TestCase):
    def test_reach_target_scripted_success_rate(self):
        from robodeploy.evaluation.runner import run_eval

        report = run_eval(
            benchmark="manipulation_v1/reach_target",
            policy="scripted",
            backend="dummy",
            episodes=20,
            base_seed=0,
            benchmarks_root=str(REPO_ROOT / "benchmarks"),
        )
        self.assertEqual(report.aggregate.n_episodes, 20)
        self.assertGreaterEqual(report.aggregate.success_rate, 0.95)

    def test_parallel_matches_sequential(self):
        from robodeploy.evaluation.runner import run_eval

        kwargs = dict(
            benchmark="manipulation_v1/reach_target",
            policy="scripted",
            backend="dummy",
            episodes=8,
            base_seed=42,
            benchmarks_root=str(REPO_ROOT / "benchmarks"),
        )
        seq = run_eval(**kwargs, parallel=False)
        par = run_eval(**kwargs, parallel=True, n_workers=4)
        self.assertEqual(seq.aggregate.n_episodes, par.aggregate.n_episodes)
        self.assertAlmostEqual(seq.aggregate.success_rate, par.aggregate.success_rate)
        self.assertAlmostEqual(seq.aggregate.mean_reward, par.aggregate.mean_reward, places=5)

    def test_suite_robo_score(self):
        from robodeploy.evaluation.runner import run_eval

        report = run_eval(
            benchmark="manipulation_v1",
            policy="scripted",
            backend="dummy",
            episodes=5,
            base_seed=0,
            benchmarks_root=str(REPO_ROOT / "benchmarks"),
        )
        self.assertGreater(report.aggregate.n_episodes, 5)
        self.assertIsNotNone(report.aggregate.robo_score)
        assert report.aggregate.robo_score is not None
        self.assertGreater(report.aggregate.robo_score, 0.0)

    def test_eval_records_trajectory_checkpoints(self):
        import tempfile
        from dataclasses import replace
        from robodeploy.evaluation.harness import EvalConfig
        from robodeploy.observability.trajectory_checkpoint import TrajectoryCheckpoint

        with tempfile.TemporaryDirectory() as tmp:
            from robodeploy.evaluation.registry import BenchmarkRegistry

            registry = BenchmarkRegistry(REPO_ROOT / "benchmarks")
            suite, task = registry.resolve("manipulation_v1/reach_target")
            assert task is not None
            preset = task.load_preset("dummy")
            task.import_task_module()
            from robodeploy.evaluation.harness import run_benchmark_eval

            report = run_benchmark_eval(
                preset=preset,
                benchmark_name=task.qualified_name,
                benchmark_version=suite.version,
                backend="dummy",
                policy_name="scripted",
                config=replace(
                    EvalConfig(n_episodes=2, base_seed=0, max_steps_per_episode=task.max_steps),
                    record_trajectories=True,
                    trajectory_dir=tmp,
                ),
            )
            checkpoints = sorted(Path(tmp).glob("*.checkpoint.json"))
            self.assertEqual(len(checkpoints), 2)
            loaded = TrajectoryCheckpoint.load(checkpoints[0])
            self.assertEqual(loaded.manifest["benchmark"], task.qualified_name)
            self.assertIsInstance(report.manifest, type(report.manifest))

    def test_report_save_roundtrip(self):
        from robodeploy.evaluation.runner import run_eval

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            report = run_eval(
                benchmark="manipulation_v1/reach_target",
                policy="scripted",
                backend="dummy",
                episodes=3,
                benchmarks_root=str(REPO_ROOT / "benchmarks"),
            )
            report.save(out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("aggregate", payload)
            self.assertIn("success_rate", payload["aggregate"])
            self.assertEqual(len(payload["episodes"]), 3)


class FailureClassifierTests(unittest.TestCase):
    def test_classifies_timeout(self):
        from robodeploy.evaluation.failure_analysis import FailureClassifier
        from robodeploy.evaluation.metrics import EpisodeMetrics

        metrics = EpisodeMetrics(
            success=False,
            reward_total=0.0,
            reward_per_step=0.0,
            steps=100,
            time_to_success_steps=None,
            time_to_success_seconds=None,
            smoothness_jerk=0.0,
            smoothness_action_norm=0.0,
            smoothness_velocity=0.0,
            collision_count=0,
            max_force_N=0.0,
            workspace_violations=0,
            distance_to_goal_final=0.2,
            distance_to_goal_min=0.1,
            constraint_violations={},
            metadata={"max_steps": 100},
        )
        self.assertEqual(FailureClassifier().classify(metrics), "timeout")


class RenderTests(unittest.TestCase):
    def test_html_report_writes_file(self):
        from robodeploy.evaluation.runner import run_eval

        with tempfile.TemporaryDirectory() as tmp:
            html = Path(tmp) / "report.html"
            report = run_eval(
                benchmark="manipulation_v1/reach_target",
                policy="scripted",
                backend="dummy",
                episodes=3,
                benchmarks_root=str(REPO_ROOT / "benchmarks"),
                html_output=str(html),
            )
            self.assertTrue(html.is_file())
            text = html.read_text(encoding="utf-8")
            self.assertIn(report.benchmark_name, text)
            self.assertIn("Success rate", text)


class EvalCliTests(unittest.TestCase):
    def test_list_benchmarks_json(self):
        from robodeploy.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(
                [
                    "list-benchmarks",
                    "--json",
                    "--benchmarks-root",
                    str(REPO_ROOT / "benchmarks"),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertIn("manipulation_v1", payload)

    def test_eval_cli_writes_output(self):
        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eval_report.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "manipulation_v1/reach_target",
                        "--policy",
                        "scripted",
                        "--backend",
                        "dummy",
                        "--episodes",
                        "5",
                        "--output",
                        str(out),
                        "--benchmarks-root",
                        str(REPO_ROOT / "benchmarks"),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark_name"], "manipulation_v1/reach_target")

    def test_eval_compare_cli(self):
        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.json"
            b = Path(tmp) / "b.json"
            out = Path(tmp) / "cmp.html"
            for path in (a, b):
                code = main(
                    [
                        "eval",
                        "--benchmark",
                        "manipulation_v1/reach_target",
                        "--policy",
                        "scripted",
                        "--backend",
                        "dummy",
                        "--episodes",
                        "3",
                        "--output",
                        str(path),
                        "--benchmarks-root",
                        str(REPO_ROOT / "benchmarks"),
                    ]
                )
                self.assertEqual(code, 0)
            code = main(["eval-compare", str(a), str(b), "--output", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())

    def test_leaderboard_submit_cli(self):
        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            main(
                [
                    "eval",
                    "--benchmark",
                    "manipulation_v1/reach_target",
                    "--policy",
                    "scripted",
                    "--backend",
                    "dummy",
                    "--episodes",
                    "3",
                    "--output",
                    str(report),
                    "--benchmarks-root",
                    str(REPO_ROOT / "benchmarks"),
                ]
            )
            code = main(
                [
                    "leaderboard",
                    "submit",
                    str(report),
                    "--benchmark",
                    "manipulation_v1/reach_target",
                    "--author",
                    "cli_test",
                    "--benchmarks-root",
                    str(REPO_ROOT / "benchmarks"),
                ]
            )
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
