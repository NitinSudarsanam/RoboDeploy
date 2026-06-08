from __future__ import annotations

import json
import unittest
import unittest.mock

from robodeploy.cli_safety import cmd_safety_check, cmd_safety_status, cmd_safety_test
from robodeploy.core.robot import Robot, RobotTask
from robodeploy.env import RoboEnv
from robodeploy.safety import SafetyMonitor
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class CliSafetyTests(unittest.TestCase):
    def test_safety_check_dummy_robot(self):
        import io
        import sys

        buf = io.StringIO()
        with unittest.mock.patch("sys.stdout", buf):
            code = cmd_safety_check(
                preset=None,
                robot="franka",
                joint_limits=None,
                presets_file=None,
                as_json=True,
                pretty=False,
            )
        payload = json.loads(buf.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])

    def test_safety_test_force_spike(self):
        import io

        buf = io.StringIO()
        with unittest.mock.patch("robodeploy.cli_safety.print_json", lambda p, **k: buf.write(json.dumps(p))):
            code = cmd_safety_test(
                preset=None,
                inject=["force_spike=80N"],
                steps=4,
                presets_file=None,
                as_json=True,
                pretty=False,
            )
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["tripped"] or payload["history_count"] > 0)

    def test_safety_status_with_active_monitor(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot], safety=SafetyMonitor())
        import io

        buf = io.StringIO()
        with unittest.mock.patch("robodeploy.cli_safety.print_json", lambda p, **k: buf.write(json.dumps(p))):
            code = cmd_safety_status(as_json=True, pretty=False)
        payload = json.loads(buf.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["active"])
        env.close()


if __name__ == "__main__":
    unittest.main()
