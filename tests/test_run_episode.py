from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action
from robodeploy.env import RoboEnv
from test_env_refactor import DummyBackend, DummyPolicy, DummyRobot, DummyTask
from robodeploy.core.robot import Robot, RobotTask


class RunEpisodeTests(unittest.TestCase):
    def test_run_episode_records_steps(self):
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        recorder = env.run_episode(
            2,
            action_fn=lambda obs: Action(
                joint_positions=jnp.asarray([float(obs.joint_positions[0]) + 0.1] * 2, dtype=jnp.float32)
            ),
        )
        self.assertEqual(len(recorder.frames), 2)


class ExportRecordedEpisodeTests(unittest.TestCase):
    def test_export_recorded_episode_writes_jsonl(self):
        import tempfile
        from pathlib import Path

        from robodeploy.dataset_export import export_recorded_episode

        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            recorder = export_recorded_episode(
                env,
                1,
                path,
                action_fn=lambda obs: Action(
                    joint_positions=jnp.asarray([0.5, 0.5], dtype=jnp.float32)
                ),
            )
            self.assertEqual(len(recorder.frames), 1)
            self.assertTrue(path.exists())


class PresetValidationTests(unittest.TestCase):
    def test_load_preset_requires_keys(self):
        from robodeploy.config import load_preset

        with patch(
            "robodeploy.config._load_all_presets",
            return_value={"bad": {"robot": "kuka"}},
        ):
            with self.assertRaises(ValueError):
                load_preset("bad")


if __name__ == "__main__":
    unittest.main()
