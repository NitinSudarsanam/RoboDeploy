"""Backend registry alias and resolution tests."""

from __future__ import annotations

import unittest


class BackendRegistryAliasTests(unittest.TestCase):
    def test_gazebo_alias_resolves_same_class(self):
        import robodeploy.backends.sim.gazebo.backend  # noqa: F401
        from robodeploy.core.registry import canonical_backend_name, get_backend, resolve_backend_name

        self.assertEqual(resolve_backend_name("gazebo"), "ros2_gazebo")
        self.assertEqual(canonical_backend_name("ros2_gazebo"), "gazebo")
        self.assertIs(get_backend("gazebo"), get_backend("ros2_gazebo"))

    def test_robo_env_coerce_backend_accepts_gazebo_alias(self):
        from unittest.mock import MagicMock, patch

        from robodeploy.env import RoboEnv

        sentinel = MagicMock()
        backend_cls = MagicMock(return_value=sentinel)
        with patch("robodeploy.env.get_backend", return_value=backend_cls) as get_backend:
            backend = RoboEnv._coerce_backend("gazebo", {})
        get_backend.assert_called_once_with("gazebo")
        backend_cls.assert_called_once_with()
        self.assertIs(backend, sentinel)


if __name__ == "__main__":
    unittest.main()
