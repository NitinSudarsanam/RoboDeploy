from __future__ import annotations

import unittest

from robodeploy.env import RoboEnv
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class EnvDiagnosticsTests(unittest.TestCase):
    def test_reset_includes_failed_builtin_imports(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        _, info = env.reset()
        diag = info.extra.get("diagnostics", {})
        self.assertIn("failed_builtin_imports", diag)
        self.assertIsInstance(diag["failed_builtin_imports"], list)


if __name__ == "__main__":
    unittest.main()
