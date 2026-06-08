from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class LeaderboardSchemaTests(unittest.TestCase):
    def test_valid_submission_passes(self):
        from robodeploy.evaluation.leaderboard import report_to_submission
        from robodeploy.evaluation.schema_validate import validate_leaderboard_submission

        submission = report_to_submission(
            {
                "benchmark_name": "manipulation_v1/reach_target",
                "benchmark_version": "1.0",
                "aggregate": {
                    "success_rate": 0.95,
                    "success_rate_ci95": [0.9, 1.0],
                    "n_episodes": 20,
                    "robo_score": 0.95,
                },
                "manifest": {"policy": "scripted", "backend": "dummy"},
            },
            author="tester",
        )
        errors = validate_leaderboard_submission(submission)
        self.assertEqual(errors, [])

    def test_missing_author_rejected(self):
        from robodeploy.evaluation.schema_validate import validate_leaderboard_submission

        bad = {
            "benchmark": "manipulation_v1/reach_target",
            "benchmark_version": "1.0",
            "policy_name": "scripted",
            "success_rate": 0.5,
            "n_episodes": 10,
            "manifest": {},
            "reproduce": {"command": "robodeploy eval", "docker_image": "robodeploy/cpu:latest"},
        }
        errors = validate_leaderboard_submission(bad)
        self.assertTrue(any("author" in err for err in errors))

    def test_submit_roundtrip(self):
        from robodeploy.evaluation.leaderboard import submit_score
        from robodeploy.evaluation.runner import run_eval

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            run_eval(
                benchmark="manipulation_v1/reach_target",
                policy="scripted",
                backend="dummy",
                episodes=3,
                benchmarks_root=str(REPO_ROOT / "benchmarks"),
            ).save(report_path)
            out = submit_score(
                report_path,
                benchmark="manipulation_v1/reach_target",
                author="unit_test",
                benchmarks_root_path=Path(tmp) / "bench_root",
            )
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["author"], "unit_test")

    def test_benchmark_spec_schema(self):
        from robodeploy.evaluation.schema_validate import validate_benchmark_spec

        spec = json.loads((REPO_ROOT / "benchmarks/manipulation_v1/spec.json").read_text(encoding="utf-8"))
        errors = validate_benchmark_spec(spec)
        self.assertEqual(errors, [])
        self.assertEqual(len(spec["tasks"]), 8)


if __name__ == "__main__":
    unittest.main()
