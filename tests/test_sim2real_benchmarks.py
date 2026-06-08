from __future__ import annotations

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class Sim2RealBenchmarkTests(unittest.TestCase):
    def test_registry_discovers_sim2real_suite(self):
        from robodeploy.evaluation.registry import BenchmarkRegistry

        registry = BenchmarkRegistry(REPO_ROOT / "benchmarks")
        self.assertIn("sim2real", registry.list_suites())
        suite = registry.load_suite("sim2real")
        names = {t.name for t in suite.tasks}
        self.assertIn("reach_to_target", names)
        self.assertIn("pick_place_cube", names)
        self.assertIn("peg_insert", names)

    def test_reach_to_target_reference_scores(self):
        from robodeploy.evaluation.registry import BenchmarkRegistry

        registry = BenchmarkRegistry(REPO_ROOT / "benchmarks")
        _, task = registry.resolve("sim2real/reach_to_target")
        assert task is not None
        scores = task.load_reference_scores()
        assert scores is not None
        targets = scores["transfer_targets"]
        self.assertAlmostEqual(targets["sim_success_rate"], 0.95, places=2)
        self.assertAlmostEqual(targets["real_success_rate"], 0.80, places=2)

    def test_task_imports_manipulation_v1(self):
        task_py = REPO_ROOT / "benchmarks" / "sim2real" / "reach_to_target" / "task.py"
        text = task_py.read_text(encoding="utf-8")
        self.assertIn("manipulation_v1.reach_target", text)

    def test_reach_dummy_preset_resolves_sim2real_pair(self):
        from robodeploy.evaluation.registry import BenchmarkRegistry
        from robodeploy.sim2real.config import load_pair_for_preset

        registry = BenchmarkRegistry(REPO_ROOT / "benchmarks")
        _, task = registry.resolve("sim2real/reach_to_target")
        assert task is not None
        preset = task.load_preset("dummy")
        pair = load_pair_for_preset(preset)
        assert pair is not None
        self.assertEqual(pair.name, "kuka_reach_dummy")

    def test_reach_benchmark_meets_sim_target_on_dummy(self):
        from robodeploy.evaluation.runner import run_eval

        report = run_eval(
            benchmark="sim2real/reach_to_target",
            policy="benchmark_reach_scripted",
            backend="dummy",
            episodes=20,
            base_seed=0,
            benchmarks_root=str(REPO_ROOT / "benchmarks"),
        )
        scores_path = REPO_ROOT / "benchmarks" / "sim2real" / "reach_to_target" / "reference_scores.json"
        targets = json.loads(scores_path.read_text(encoding="utf-8"))["transfer_targets"]
        self.assertGreaterEqual(report.aggregate.success_rate, targets["sim_success_rate"])


if __name__ == "__main__":
    unittest.main()
