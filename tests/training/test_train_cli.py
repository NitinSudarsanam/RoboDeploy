from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainCliTests(unittest.TestCase):
    def test_train_ppo_reach_example_exists(self):
        script = REPO_ROOT / "examples" / "train_ppo_reach.py"
        self.assertTrue(script.is_file(), "GOAL 02 references examples/train_ppo_reach.py")

    def test_train_bc_dummy_smoke(self):
        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "demos.jsonl"
            log_dir = Path(tmp) / "runs"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
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
            self.assertTrue((log_dir / "bc_final.pt").exists())

    def test_train_ppo_dummy_smoke(self):
        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "runs"
            code = main(
                [
                    "train",
                    "ppo",
                    "--dummy",
                    "--n-envs",
                    "2",
                    "--total-steps",
                    "64",
                    "--rollout-steps",
                    "32",
                    "--log-dir",
                    str(log_dir),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((log_dir / "ppo_final.pt").exists())

    def test_convert_dataset_jsonl_to_hdf5(self):
        from robodeploy.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "demos.jsonl"
            dst = Path(tmp) / "demos.hdf5"
            src.write_text(
                '{"observation": {"joint_positions": [0.0, 0.0], "joint_velocities": [0.0, 0.0], '
                '"joint_torques": [0.0, 0.0]}, "action": {"joint_positions": [0.1, 0.1]}, '
                '"reward": 0.0, "done": false}\n',
                encoding="utf-8",
            )
            code = main(["convert-dataset", "--from", str(src), "--to", str(dst)])
            self.assertEqual(code, 0)
            self.assertTrue(dst.exists())

    def test_convert_dataset_lerobot_uri(self):
        try:
            import lerobot  # noqa: F401
        except ImportError:
            self.skipTest("lerobot not installed")

        from robodeploy.cli import main
        from robodeploy.dataset_export import export_to_lerobot
        from robodeploy.demo_recording import DemoFrame, DemoRecorder

        recorder = DemoRecorder()
        recorder.frames.append(
            DemoFrame(
                observation={"joint_positions": [0.1, 0.2], "joint_velocities": [0.0, 0.0]},
                action={"joint_positions": [0.2, 0.3]},
                reward=0.0,
                done=False,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lerobot_root"
            export_to_lerobot(recorder, repo_id="local/cli_convert", fps=10, root=root)
            dst = Path(tmp) / "out.hdf5"
            code = main(
                [
                    "convert-dataset",
                    "--from",
                    "lerobot://local/cli_convert",
                    "--lerobot-root",
                    str(root),
                    "--to",
                    str(dst),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(dst.exists())


if __name__ == "__main__":
    unittest.main()
