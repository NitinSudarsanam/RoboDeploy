from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robodeploy.cli_observability import cmd_replay
from robodeploy.cli_teleop import cmd_demo_replay, cmd_teleop
from robodeploy.dataset_export import export_demo_jsonl
from robodeploy.demo_recording import DemoFrame, DemoRecorder


def _write_demo(path: Path) -> None:
    recorder = DemoRecorder()
    recorder.frames.append(
        DemoFrame(
            observation={"joint_positions": [0.1, 0.2]},
            action={"joint_positions": [0.11, 0.21]},
            reward=0.0,
            done=False,
        )
    )
    export_demo_jsonl(recorder, path)


class ReplayCliTests(unittest.TestCase):
    def test_replay_dummy_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "demo.jsonl"
            _write_demo(demo)
            with mock.patch("robodeploy.cli_observability.close_quietly"):
                code = cmd_replay(
                    recording=str(demo),
                    dummy=True,
                    seed=None,
                    diff=False,
                    output="",
                    on_divergence="warn",
                    as_json=True,
                )
            self.assertEqual(code, 0)

    def test_demo_replay_with_mock_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo = Path(tmp) / "demo.jsonl"
            _write_demo(demo)
            env = mock.MagicMock()
            env.reset.return_value = (mock.MagicMock(), {})
            env.step.return_value = (mock.MagicMock(), 0.0, False, {})
            with mock.patch("robodeploy.cli_teleop._env_from_preset", return_value=env):
                with mock.patch("robodeploy.cli_helpers.close_quietly"):
                    code = cmd_demo_replay(
                        recording=str(demo),
                        preset="kuka_pick_mujoco",
                        presets_file=None,
                        dummy=False,
                        speed=1.0,
                        pause_at_step=1,
                        as_json=True,
                    )
            self.assertEqual(code, 0)

    def test_teleop_cli_mock_session(self) -> None:
        env = mock.MagicMock()
        with mock.patch("robodeploy.cli_teleop._env_from_preset", return_value=env):
            with mock.patch("robodeploy.cli_helpers.close_quietly"):
                with mock.patch(
                    "robodeploy.teleop.session.run_teleop_session",
                    return_value=[Path("demos/episode_001.jsonl")],
                ):
                    code = cmd_teleop(
                        preset="kuka_pick_mujoco",
                        presets_file=None,
                        device="keyboard",
                        record="demos/episode_001.jsonl",
                        fmt="jsonl",
                        max_steps=10,
                        start_recording=False,
                        as_json=True,
                    )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
