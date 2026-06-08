from __future__ import annotations

import unittest
import warnings

from robodeploy.core.registry import (
    auto_discover_entry_points,
    register_robot,
    unregister_robot,
)


class RegistryDuplicateTests(unittest.TestCase):
    def tearDown(self) -> None:
        unregister_robot("dup_test_robot")

    def test_decorator_duplicate_raises_key_error(self):
        class RobotA:
            pass

        class RobotB:
            pass

        register_robot("dup_test_robot")(RobotA)
        with self.assertRaises(KeyError) as ctx:
            register_robot("dup_test_robot")(RobotB)
        self.assertIn("already registered", str(ctx.exception))

    def test_entry_point_discovery_warns_and_overrides(self):
        from unittest import mock

        class BuiltinRobot:
            pass

        class PluginRobot:
            pass

        register_robot("dup_test_robot")(BuiltinRobot)

        class _FakeEP:
            def __init__(self, name: str):
                self.name = name

            def load(self):
                register_robot("dup_test_robot")(PluginRobot)
                return PluginRobot

        def _fake_entry_points(*, group: str):
            if group == "robodeploy.robots":
                return [_FakeEP("plugin_robot")]
            return []

        with mock.patch("importlib.metadata.entry_points", side_effect=_fake_entry_points):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                auto_discover_entry_points()
        self.assertTrue(any("overriding" in str(w.message) for w in caught))
        from robodeploy.core.registry import get_robot

        self.assertIs(get_robot("dup_test_robot"), PluginRobot)
        unregister_robot("dup_test_robot")


if __name__ == "__main__":
    unittest.main()
