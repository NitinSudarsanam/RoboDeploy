from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from robodeploy.demo_recording import DemoFrame
from robodeploy.training.dataset import DemoCollator, DemoDataset, SequenceDataset


def _sample_frame(value: float) -> DemoFrame:
    return DemoFrame(
        observation={
            "joint_positions": [value, value],
            "joint_velocities": [0.0, 0.0],
            "joint_torques": [0.0, 0.0],
        },
        action={"joint_positions": [value + 0.1, value + 0.1]},
        reward=value,
        done=False,
    )


class DatasetTests(unittest.TestCase):
    def test_jsonl_load_and_batch_shapes(self):
        frames = [_sample_frame(float(i)) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demos.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for frame in frames:
                    handle.write(json.dumps(asdict(frame)) + "\n")
            dataset = DemoDataset.from_jsonl(path)
            self.assertEqual(len(dataset), 5)
            item = dataset[0]
            self.assertEqual(item["obs"]["proprio"].shape[0], 6)
            self.assertEqual(item["action"].shape[0], 2)
            batch = DemoCollator()([dataset[i] for i in range(3)])
            self.assertEqual(batch["action"].shape, (3, 2))
            self.assertEqual(batch["obs"]["proprio"].shape, (3, 6))

    def test_sequence_dataset_window(self):
        dataset = DemoDataset([_sample_frame(float(i)) for i in range(4)])
        seq = SequenceDataset(dataset, horizon=2)
        self.assertEqual(len(seq), 3)
        item = seq[0]
        self.assertEqual(item["action"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
