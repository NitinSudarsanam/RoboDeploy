from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class PluginDiscoveryTests(unittest.TestCase):
    def test_demo_plugin_registers_via_entry_points(self):
        repo = Path(__file__).resolve().parents[1]
        plugin_dir = repo / "examples" / "plugin_robot_demo"
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(plugin_dir), "--no-deps", "-q"],
            cwd=str(repo),
        )
        from robodeploy.core.registry import get_robot, get_task, list_registered

        from robodeploy import discover

        discover()
        names = list_registered()
        self.assertIn("demo_arm", names["robots"])
        self.assertIn("demo_task", names["tasks"])
        get_robot("demo_arm")
        get_task("demo_task")


if __name__ == "__main__":
    unittest.main()
