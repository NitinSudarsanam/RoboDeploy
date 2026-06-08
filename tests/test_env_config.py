from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class EnvConfigTests(unittest.TestCase):
    def test_config_builder_builds(self):
        from robodeploy.core.config_builder import EnvConfigBuilder
        from robodeploy.core.env_config import EnvConfig

        cfg = (
            EnvConfigBuilder()
            .with_robot("franka")
            .with_backend("mujoco")
            .with_task("pick_place")
            .with_policy("example_reach_pick")
            .add_sensor("wrist_ft")
            .validate()
            .build()
        )
        self.assertIsInstance(cfg, EnvConfig)
        self.assertEqual(cfg.robot, "franka")

    def test_task_config_validation(self):
        from robodeploy.core.task_config import TaskConfig

        with self.assertRaises(ValueError):
            TaskConfig(success_threshold=-1.0)

    def test_robo_env_from_config_accepts_env_config(self):
        from unittest.mock import MagicMock, patch

        from robodeploy.core.config_builder import EnvConfigBuilder
        from robodeploy.core.env_config import EnvConfig
        from robodeploy.env import RoboEnv

        cfg = (
            EnvConfigBuilder()
            .with_robot("franka")
            .with_backend("mujoco")
            .with_task("pick_place")
            .with_policy("teleop")
            .validate()
            .build()
        )
        self.assertIsInstance(cfg, EnvConfig)
        backend = MagicMock(is_real=False)
        with (
            patch.object(cfg, "to_dict", wraps=cfg.to_dict) as mock_to_dict,
            patch.object(RoboEnv, "_coerce_backend", return_value=backend),
            patch.object(RoboEnv, "_coerce_description", return_value=MagicMock()),
            patch.object(RoboEnv, "_coerce_task", return_value=MagicMock()),
            patch.object(RoboEnv, "_coerce_policy", return_value=MagicMock()),
            patch.object(RoboEnv, "_coerce_sensors", return_value=[]),
            patch.object(RoboEnv, "__init__", return_value=None),
        ):
            RoboEnv.from_config(cfg)
        mock_to_dict.assert_called_once()
