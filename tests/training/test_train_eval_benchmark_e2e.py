"""Train BC on dummy demos, then evaluate checkpoint on manipulation_v1/reach_target."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainEvalBenchmarkE2ETests(unittest.TestCase):
    def test_train_bc_then_eval_reach_target_checkpoint(self):
        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "demos.jsonl"
            log_dir = tmp_path / "runs"
            ckpt = log_dir / "bc_final.pt"
            report = tmp_path / "reach_eval.json"

            code = main(
                [
                    "train",
                    "bc",
                    "--dataset",
                    str(dataset),
                    "--dummy",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "4",
                    "--log-dir",
                    str(log_dir),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(ckpt.is_file(), "BC training should write bc_final.pt")

            code = main(
                [
                    "eval",
                    "--benchmark",
                    "manipulation_v1/reach_target",
                    "--policy",
                    str(ckpt),
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
            self.assertEqual(code, 0)
            self.assertTrue(report.is_file())

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["benchmark_name"], "manipulation_v1/reach_target")
            self.assertEqual(payload["aggregate"]["n_episodes"], 3)
            self.assertIn("success_rate", payload["aggregate"])
            self.assertEqual(len(payload["episodes"]), 3)


if __name__ == "__main__":
    unittest.main()
