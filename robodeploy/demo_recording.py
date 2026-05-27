"""Record and replay RoboEnv trajectories through the public step contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterator

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
        payload = {"version": 1, "frames": [asdict(frame) for frame in self.frames]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DemoRecorder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        recorder = cls()
        for item in payload.get("frames", []):
            recorder.frames.append(DemoFrame(**item))
        return recorder


class DemoSession:
    """Thin wrapper that records every env.step automatically."""

    def __init__(self, env, recorder: DemoRecorder | None = None) -> None:  # noqa: ANN001
        self._env = env
        self.recorder = recorder or DemoRecorder()

    def reset(self):
        return self._env.reset()

    def step(self, action=None):  # noqa: ANN001
        obs, reward, done, info = self._env.step(action)
        if action is not None:
            self.recorder.record_step(obs, action, reward=reward, done=done)
        return obs, reward, done, info

    def iter_replay_actions(self) -> Iterator[Action]:
        for frame in self.recorder.frames:
            yield _action_from_json(frame.action)


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
