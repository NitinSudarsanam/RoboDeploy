"""Headless teleop record → DemoDataset → BC train stub (GOAL 04)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robodeploy.cli import main as cli_main
from robodeploy.teleop.base import ITeleopDevice, TeleopCommand
from robodeploy.teleop.session import record_stub_episode
from robodeploy.training.dataset import DemoDataset


class _ScriptedDevice(ITeleopDevice):
    def __init__(self, n_steps: int = 5) -> None:
        self._remaining = int(n_steps)

    def start(self) -> None:
        return

    def poll(self) -> TeleopCommand | None:
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return TeleopCommand(
            delta_joint_positions=np.array([0.01, -0.01], dtype=np.float32),
        )

    def stop(self) -> None:
        return


class TeleopRecordStubTests(unittest.TestCase):
    def test_record_stub_jsonl_loads_for_bc_train(self) -> None:
        from robodeploy.cli_observability import _make_dummy_env

        env = _make_dummy_env()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / "demos"
                saved = record_stub_episode(
                    env,
                    _ScriptedDevice(n_steps=4),
                    output_dir=out_dir,
                    fmt="jsonl",
                    max_steps=6,
                    metadata={"device": "scripted"},
                )
                self.assertEqual(len(saved), 1)
                dataset = DemoDataset.from_teleop_jsonl(saved[0])
                self.assertGreaterEqual(len(dataset), 1)

                log_dir = Path(tmp) / "runs"
                code = cli_main(
                    [
                        "train",
                        "bc",
                        "--dataset",
                        str(saved[0]),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
                        "--log-dir",
                        str(log_dir),
                    ]
                )
                self.assertEqual(code, 0)
                self.assertTrue((log_dir / "bc_final.pt").is_file())
        finally:
            env.close()

    def test_record_stub_json_includes_metadata(self) -> None:
        from robodeploy.cli_observability import _make_dummy_env

        env = _make_dummy_env()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                saved = record_stub_episode(
                    env,
                    _ScriptedDevice(n_steps=2),
                    output_dir=Path(tmp),
                    fmt="json",
                    max_steps=4,
                    metadata={"device": "scripted", "preset": "dummy"},
                )
                payload = json.loads(saved[0].read_text(encoding="utf-8"))
                self.assertEqual(payload["metadata"].get("source"), "teleop_stub")
                self.assertEqual(payload["metadata"].get("device"), "scripted")
                self.assertEqual(payload["metadata"].get("preset"), "dummy")
                self.assertGreaterEqual(len(payload["frames"]), 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
