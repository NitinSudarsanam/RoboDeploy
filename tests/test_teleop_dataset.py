from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from robodeploy.demo_recording import DemoFrame, DemoRecorder
from robodeploy.dataset_export import export_demo_jsonl
from robodeploy.training.dataset import DemoDataset


class TeleopDatasetTests(unittest.TestCase):
    def test_teleop_jsonl_loads_into_demo_dataset(self):
        recorder = DemoRecorder()
        recorder.frames.append(
            DemoFrame(
                observation={
                    "joint_positions": [0.1, 0.2],
                    "joint_velocities": [0.0, 0.0],
                    "joint_torques": [0.0, 0.0],
                    "ee_position": [0.0, 0.0, 0.5],
                    "ee_orientation": [1.0, 0.0, 0.0, 0.0],
                },
                action={"joint_positions": [0.2, 0.3], "action_space": "JOINT_POS"},
                reward=0.5,
                done=False,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_001.jsonl"
            export_demo_jsonl(recorder, path)
            dataset = DemoDataset.from_teleop_jsonl(path)
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(item["action"].shape[0], 2)

    def test_from_jsonl_bundle_format(self):
        frames = [
            DemoFrame(
                observation={"joint_positions": [0.0, 0.0], "joint_velocities": [0.0, 0.0], "joint_torques": [0.0, 0.0]},
                action={"joint_positions": [0.1, 0.1]},
                reward=0.0,
                done=False,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.jsonl"
            path.write_text(json.dumps({"version": 1, "frames": [asdict(f) for f in frames]}), encoding="utf-8")
            dataset = DemoDataset.from_jsonl(path)
            self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
