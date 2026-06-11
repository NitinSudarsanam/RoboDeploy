"""Record and replay RoboEnv trajectories through the public step contract."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from robodeploy.core.interop import to_numpy
from robodeploy.core.types import Action, Observation


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "shape") or hasattr(value, "dtype"):
        return to_numpy(value).tolist()
    return value


@dataclass
class DemoFrame:
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: bool


class DemoRecorder:
    """Collect frames from a DemoSession or manual record_step calls."""

    def __init__(self) -> None:
        self.frames: list[DemoFrame] = []
        self.metadata: dict[str, Any] = {}

    def record_step(
        self,
        obs: Observation,
        action: Action,
        *,
        reward: float = 0.0,
        done: bool = False,
    ) -> None:
        self.frames.append(
            DemoFrame(
                observation=_to_jsonable(obs),
                action=_to_jsonable(action),
                reward=float(reward),
                done=bool(done),
            )
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "metadata": dict(self.metadata),
            "frames": [asdict(frame) for frame in self.frames],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DemoRecorder":
        text = Path(path).read_text(encoding="utf-8").strip()
        if not text:
            return cls()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and "frames" in payload:
                recorder = cls()
                recorder.metadata = dict(payload.get("metadata") or {})
                for item in payload.get("frames", []):
                    recorder.frames.append(DemoFrame(**item))
                return recorder
        recorder = cls()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict) and "frames" in item:
                recorder.metadata = dict(item.get("metadata") or {})
                for frame in item.get("frames", []):
                    recorder.frames.append(DemoFrame(**frame))
                return recorder
            recorder.frames.append(DemoFrame(**item))
        return recorder


class DemoSession:
    """Thin wrapper that records every env.step automatically."""

    def __init__(self, env, recorder: DemoRecorder | None = None) -> None:  # noqa: ANN001
        self._env = env
        self.recorder = recorder or DemoRecorder()

    def reset(self, *, seed: int | None = None):
        import inspect

        if seed is not None:
            self.recorder.metadata["seed"] = int(seed)
        params = inspect.signature(self._env.reset).parameters
        if "seed" in params:
            return self._env.reset(seed=seed)
        return self._env.reset()

    def step(self, action=None):  # noqa: ANN001
        obs, reward, done, info = self._env.step(action)
        if action is not None:
            self.recorder.record_step(obs, action, reward=reward, done=done)
        return obs, reward, done, info

    def iter_replay_actions(self) -> Iterator[Action]:
        for frame in self.recorder.frames:
            yield _action_from_json(frame.action)


def load_demo_jsonl(path: str | Path) -> list[DemoFrame]:
    """Load frames from JSONL export (one DemoFrame per line)."""
    frames: list[DemoFrame] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        frames.append(DemoFrame(**item))
    return frames


def replay_demo_frames(
    env,
    frames: list[DemoFrame],
    *,
    speed: float = 1.0,
    pause_at_step: int | None = None,
    sleep_fn=time.sleep,
) -> int:
    """Replay recorded actions through env.step; returns steps executed."""
    steps = 0
    env.reset()
    for index, frame in enumerate(frames):
        if pause_at_step is not None and index >= int(pause_at_step):
            break
        action = _action_from_json(frame.action)
        env.step(action)
        steps += 1
        if speed > 0.0 and speed < 1.0:
            sleep_fn((1.0 / max(speed, 1e-6) - 1.0) * 0.01)
    return steps


class InteractiveDemoSession:
    """DemoSession with teleop hot-keys for record toggle and episode reset."""

    def __init__(
        self,
        env,
        policy,
        *,
        output_dir: str | Path,
        fmt: Literal["jsonl", "hdf5", "json", "lerobot"] = "jsonl",
        start_recording: bool = False,
        max_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:  # noqa: ANN001
        self._env = env
        self._policy = policy
        self._output_dir = Path(output_dir)
        self._fmt = str(fmt)
        self._recording = bool(start_recording)
        self._episode_index = 0
        self._recorder = DemoRecorder()
        if metadata:
            self._recorder.metadata.update(metadata)
        self._max_steps = max_steps
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def recording(self) -> bool:
        return self._recording

    def _teleop_command(self):
        return getattr(self._policy, "last_command", None)

    def _save_episode(self) -> Path | None:
        if not self._recorder.frames:
            return None
        self._episode_index += 1
        stem = f"episode_{self._episode_index:03d}"
        if self._fmt == "hdf5":
            from robodeploy.dataset_export import export_demo_hdf5

            path = self._output_dir / f"{stem}.hdf5"
            export_demo_hdf5(self._recorder, path)
        elif self._fmt == "json":
            path = self._output_dir / f"{stem}.json"
            self._recorder.save(path)
        elif self._fmt == "lerobot":
            from robodeploy.dataset_export import export_to_lerobot

            repo_id = f"robodeploy/{stem}"
            export_to_lerobot(
                self._recorder,
                repo_id=repo_id,
                root=self._output_dir / "lerobot",
            )
            path = self._output_dir / "lerobot" / repo_id.replace("/", os.sep)
        else:
            from robodeploy.dataset_export import export_demo_jsonl

            path = self._output_dir / f"{stem}.jsonl"
            export_demo_jsonl(self._recorder, path)
        self._recorder = DemoRecorder()
        return path

    def run(self) -> list[Path]:
        """Run until max_steps, e-stop, or env close. Returns saved episode paths."""
        from robodeploy.teleop.base import TeleopSafetyError

        saved: list[Path] = []
        obs, _info = self._env.reset()
        self._policy.reset()
        step_count = 0
        while self._max_steps is None or step_count < self._max_steps:
            try:
                action = self._policy.get_action(obs)
            except TeleopSafetyError as exc:
                emergency = getattr(self._env, "emergency_stop", None)
                if callable(emergency):
                    emergency(str(exc))
                break
            obs, reward, done, _info = self._env.step(action)
            step_count += 1

            cmd = self._teleop_command()
            if cmd is not None and cmd.record_toggle:
                self._recording = not self._recording
            if self._recording:
                self._recorder.record_step(obs, action, reward=reward, done=done)

            if cmd is not None and cmd.reset_episode:
                path = self._save_episode()
                if path is not None:
                    saved.append(path)
                obs, _info = self._env.reset()
                self._policy.reset()
                continue

            if done:
                if self._recording:
                    path = self._save_episode()
                    if path is not None:
                        saved.append(path)
                obs, _info = self._env.reset()
                self._policy.reset()
        if self._recording:
            path = self._save_episode()
            if path is not None:
                saved.append(path)
        return saved


def _action_from_json(payload: dict[str, Any]) -> Action:
    try:
        import jax.numpy as jnp
    except Exception:
        import numpy as jnp  # type: ignore[assignment]

    def _arr(key: str):
        raw = payload.get(key)
        if raw is None:
            return None
        return jnp.asarray(raw, dtype=jnp.float32)

    return Action(
        joint_positions=_arr("joint_positions"),
        joint_velocities=_arr("joint_velocities"),
        joint_torques=_arr("joint_torques"),
        ee_position=_arr("ee_position"),
        ee_orientation=_arr("ee_orientation"),
        ee_velocity=_arr("ee_velocity"),
        gripper=payload.get("gripper"),
        timestamp=float(payload.get("timestamp", 0.0)),
    )
