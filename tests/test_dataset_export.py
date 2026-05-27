from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robodeploy.core.types import Action, Observation
from robodeploy.dataset_export import export_demo_jsonl
from robodeploy.demo_recording import DemoRecorder


class DatasetExportTests(unittest.TestCase):
    def test_export_demo_jsonl_writes_lines(self):
        recorder = DemoRecorder()
        obs = Observation(
            joint_positions=np.zeros(2, dtype=np.float32),
            joint_velocities=np.zeros(2, dtype=np.float32),
            joint_torques=np.zeros(2, dtype=np.float32),
            ee_position=np.zeros(3, dtype=np.float32),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ee_velocity=np.zeros(3, dtype=np.float32),
            ee_angular_velocity=np.zeros(3, dtype=np.float32),
        )
        recorder.record_step(obs, Action(joint_positions=np.ones(2, dtype=np.float32)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.jsonl"
            export_demo_jsonl(recorder, path)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)


    def test_export_hdf5_when_h5py_available(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py not installed")
        from robodeploy.dataset_export import export_demo_hdf5

        recorder = DemoRecorder()
        obs = Observation(
            joint_positions=np.zeros(2, dtype=np.float32),
            joint_velocities=np.zeros(2, dtype=np.float32),
            joint_torques=np.zeros(2, dtype=np.float32),
            ee_position=np.zeros(3, dtype=np.float32),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ee_velocity=np.zeros(3, dtype=np.float32),
            ee_angular_velocity=np.zeros(3, dtype=np.float32),
        )
        recorder.record_step(obs, Action(joint_positions=np.ones(2, dtype=np.float32)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.h5"
            export_demo_hdf5(recorder, path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
