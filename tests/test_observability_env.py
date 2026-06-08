from __future__ import annotations

import tempfile
import unittest

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.env import RoboEnv
from robodeploy.observability.logger import JsonlSink, RoboDeployLogger
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class ObservabilityEnvTests(unittest.TestCase):
    def test_step_surfaces_sensor_status_and_backend_diagnostics(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        try:
            _obs, info = env.reset(seed=1)
            self.assertIn("sensor_status", info.extra)
            self.assertIn("backend_diagnostics", info.extra)
            self.assertIn("sensor_health", info.extra)
            self.assertIn("health_status", info.extra)
            self.assertEqual(info.extra["backend_diagnostics"].get("backend"), "dummy")
            _obs, _r, _d, info = env.step(None)
            self.assertIn("sensor_status", info.extra)
        finally:
            env.close()

    def test_logger_writes_jsonl_on_step(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        with tempfile.TemporaryDirectory() as tmp:
            logger = RoboDeployLogger(sinks=[JsonlSink(f"{tmp}/run.jsonl")])
            env = RoboEnv(backend=DummyBackend(), robots=[robot], logger=logger)
            try:
                env.reset(seed=0)
                env.step(None)
                logger.close()
                text = open(f"{tmp}/run.jsonl", encoding="utf-8").read()
                self.assertIn('"kind": "step"', text)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
