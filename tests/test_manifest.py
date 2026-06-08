from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.env import RoboEnv
from robodeploy.observability.manifest import ManifestRecorder, RunManifest
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class ManifestTests(unittest.TestCase):
    def test_round_trip_and_recorder_fields(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(
            backend=DummyBackend(),
            robots=[robot],
            record_manifest=True,
            run_name="unit-test",
        )
        try:
            env.reset(seed=7)
            recorder = ManifestRecorder(env, run_name="unit-test")
            manifest = recorder.manifest
            self.assertEqual(manifest.run_name, "unit-test")
            self.assertEqual(manifest.seed, 7)
            self.assertEqual(manifest.backend, "DummyBackend")
            self.assertEqual(manifest.robot, "dummy")
            self.assertIsInstance(manifest.python_version, str)
            self.assertIsInstance(manifest.package_version, str)
            with tempfile.TemporaryDirectory() as tmp:
                path = recorder.write(tmp)
                loaded = RunManifest.load(path)
                self.assertEqual(loaded.run_name, manifest.run_name)
                self.assertEqual(loaded.seed, 7)
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                self.assertIn("git_hash", payload)
                self.assertIn("git_dirty", payload)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
