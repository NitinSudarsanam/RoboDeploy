from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robodeploy.core.types import Action, Observation
from robodeploy.demo_recording import DemoRecorder, _action_from_json


def make_obs() -> Observation:
    return Observation(
        joint_positions=np.array([0.1, 0.2], dtype=np.float32),
        joint_velocities=np.zeros(2, dtype=np.float32),
        joint_torques=np.zeros(2, dtype=np.float32),
        ee_position=np.zeros(3, dtype=np.float32),
        ee_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ee_velocity=np.zeros(3, dtype=np.float32),
        ee_angular_velocity=np.zeros(3, dtype=np.float32),
    )


class DemoRecordingTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        recorder = DemoRecorder()
        action = Action(joint_positions=np.array([1.0, 2.0], dtype=np.float32), gripper=0.5)
        recorder.record_step(make_obs(), action, reward=1.0, done=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.json"
            recorder.save(path)
            loaded = DemoRecorder.load(path)
        self.assertEqual(len(loaded.frames), 1)
        self.assertAlmostEqual(loaded.frames[0].reward, 1.0)

    def test_action_from_json_restores_joint_positions(self):
        payload = {"joint_positions": [0.5, 0.6], "gripper": 0.25, "timestamp": 0.0}
        action = _action_from_json(payload)
        self.assertAlmostEqual(float(action.joint_positions[0]), 0.5)
        self.assertEqual(action.gripper, 0.25)


if __name__ == "__main__":
    unittest.main()
