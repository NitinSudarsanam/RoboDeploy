from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robodeploy.dataset_export import export_to_lerobot, export_to_rlds, export_to_robomimic
from robodeploy.demo_recording import DemoFrame, DemoRecorder
from robodeploy.training.dataset import DemoDataset

try:
    import lerobot  # noqa: F401

    _HAS_LEROBOT = True
except ImportError:
    _HAS_LEROBOT = False


def _sample_recorder() -> DemoRecorder:
    recorder = DemoRecorder()
    for index in range(4):
        recorder.frames.append(
            DemoFrame(
                observation={
                    "joint_positions": [0.1 * index, 0.2, 0.3],
                    "joint_velocities": [0.0, 0.0, 0.0],
                },
                action={"joint_positions": [0.1 * index, 0.2, 0.3], "gripper": 0.5},
                reward=float(index),
                done=index == 3,
            )
        )
    return recorder


@unittest.skipUnless(_HAS_LEROBOT, "lerobot not installed")
class LeRobotExportTests(unittest.TestCase):
    def test_export_round_trip_via_demo_dataset(self) -> None:
        recorder = _sample_recorder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lerobot_root"
            dataset = export_to_lerobot(
                recorder,
                repo_id="local/robodeploy_test",
                fps=10,
                root=root,
            )
            loaded = DemoDataset.from_lerobot("local/robodeploy_test", root=root)
            self.assertEqual(len(loaded), len(recorder.frames))
            self.assertEqual(loaded.proprio_dim, 6)
            self.assertGreater(loaded.action_dim, 0)
            self.assertIsNotNone(dataset.root)

    def test_lerobot_dataset_loadable(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        recorder = _sample_recorder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "lerobot_root"
            export_to_lerobot(recorder, repo_id="local/load_test", fps=10, root=root)
            ds = LeRobotDataset(repo_id="local/load_test", root=root)
            self.assertEqual(len(ds), len(recorder.frames))


class OtherExportTests(unittest.TestCase):
    def test_robomimic_round_trip(self) -> None:
        recorder = _sample_recorder()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.hdf5"
            export_to_robomimic(recorder, output_path=path)
            loaded = DemoDataset.from_robomimic(path)
            self.assertEqual(len(loaded), len(recorder.frames))

    def test_rlds_round_trip(self) -> None:
        recorder = _sample_recorder()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rlds_bundle"
            export_to_rlds(recorder, output_dir=out)
            loaded = DemoDataset.from_rlds(out)
            self.assertEqual(len(loaded), len(recorder.frames))


if __name__ == "__main__":
    unittest.main()
