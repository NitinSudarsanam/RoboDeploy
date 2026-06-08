from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.robot import Robot, RobotTask
from robodeploy.core.types import Action, Observation
from robodeploy.demo_recording import InteractiveDemoSession, load_demo_jsonl, replay_demo_frames
from robodeploy.env import RoboEnv
from robodeploy.teleop.base import ITeleopDevice, TeleopCommand
from robodeploy.teleop.controller import TeleopPolicy
from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask


class _SequenceDevice(ITeleopDevice):
    def __init__(self, commands: list[TeleopCommand | None]) -> None:
        self._commands = list(commands)

    def start(self) -> None:
        return

    def poll(self) -> TeleopCommand | None:
        if not self._commands:
            return None
        return self._commands.pop(0)

    def stop(self) -> None:
        return


class _StepEnv:
    """Minimal env stub for InteractiveDemoSession tests."""

    def __init__(self, steps_until_done: int = 100) -> None:
        self._step = 0
        self._done_at = steps_until_done

    def reset(self):
        self._step = 0
        return _obs(), {}

    def step(self, action):  # noqa: ANN001
        self._step += 1
        done = self._step >= self._done_at
        return _obs(), 0.0, done, {}


def _obs() -> Observation:
    q = jnp.asarray([0.1, 0.2], dtype=jnp.float32)
    return Observation(
        joint_positions=q,
        joint_velocities=jnp.zeros(2, dtype=jnp.float32),
        joint_torques=jnp.zeros(2, dtype=jnp.float32),
        ee_position=jnp.zeros(3, dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros(3, dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros(3, dtype=jnp.float32),
    )


class InteractiveDemoSessionTests(unittest.TestCase):
    def test_recording_saves_jsonl_on_reset(self) -> None:
        device = _SequenceDevice(
            [
                TeleopCommand(delta_joint_positions=jnp.asarray([0.01, 0.0], dtype=jnp.float32)),
                TeleopCommand(reset_episode=True),
            ]
        )
        policy = TeleopPolicy(device=device)
        env = _StepEnv()
        with tempfile.TemporaryDirectory() as tmp:
            session = InteractiveDemoSession(
                env,
                policy,
                output_dir=tmp,
                start_recording=True,
                max_steps=2,
            )
            saved = session.run()
            self.assertEqual(len(saved), 1)
            lines = Path(saved[0]).read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 1)

    def test_jsonl_round_trip_replay(self) -> None:
        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot], safety_enabled=False)
        frames = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.jsonl"
            obs, _ = env.reset()
            for i in range(3):
                action = Action(joint_positions=jnp.asarray([float(i) * 0.1, 0.2], dtype=jnp.float32))
                obs, reward, done, _ = env.step(action)
                frames.append(
                    {
                        "observation": json.loads(json.dumps({"joint_positions": [float(i) * 0.1, 0.2]})),
                        "action": {"joint_positions": [float(i) * 0.1, 0.2]},
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )
            path.write_text("\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8")
            loaded = load_demo_jsonl(path)
            self.assertEqual(len(loaded), 3)
            with patch("robodeploy.demo_recording.time.sleep"):
                steps = replay_demo_frames(env, loaded, speed=0.5)
            self.assertEqual(steps, 3)
        env.close()


if __name__ == "__main__":
    unittest.main()
