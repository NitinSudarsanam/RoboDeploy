"""Trajectory replay with optional divergence detection."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from robodeploy.core.interop import to_numpy
from robodeploy.demo_recording import DemoRecorder


def _load_recording(recording: DemoRecorder | Path | str) -> DemoRecorder:
    if not isinstance(recording, (str, Path)):
        return recording
    path = Path(recording)
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict) and "schema_version" in payload and "frames" in payload:
            from robodeploy.observability.trajectory_checkpoint import TrajectoryCheckpoint

            return TrajectoryCheckpoint.load(path).to_recorder()
    return DemoRecorder.load(path)

if TYPE_CHECKING:
    from robodeploy.env import RoboEnv


def _vec_delta(a, b) -> float:  # noqa: ANN001
    if a is None or b is None:
        return float("inf")
    av = to_numpy(a).astype("float64").reshape(-1)
    bv = to_numpy(b).astype("float64").reshape(-1)
    if av.shape != bv.shape:
        return float("inf")
    return float(abs(av - bv).max())


def _obs_from_recorded(payload: dict) -> dict:
    return dict(payload or {})


@dataclass
class ReplayReport:
    divergences: list[dict] = field(default_factory=list)
    max_divergence: dict[str, float] = field(default_factory=dict)
    diverged_steps: list[int] = field(default_factory=list)
    steps_played: int = 0
    halted: bool = False

    def add(self, step: int, div: dict[str, float], *, exceeded: bool) -> None:
        self.divergences.append({"step": int(step), **div})
        for key, value in div.items():
            self.max_divergence[key] = max(float(self.max_divergence.get(key, 0.0)), float(value))
        if exceeded:
            self.diverged_steps.append(int(step))

    def to_dict(self) -> dict:
        return {
            "steps_played": self.steps_played,
            "halted": self.halted,
            "max_divergence": self.max_divergence,
            "diverged_steps": self.diverged_steps,
            "divergences": self.divergences,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class TrajectoryReplayer:
    """Replay recorded actions and compare observations."""

    def __init__(
        self,
        *,
        env: "RoboEnv",
        recording: DemoRecorder | Path | str,
        divergence_threshold: dict[str, float] | None = None,
        on_divergence: Literal["warn", "halt", "record"] = "warn",
    ) -> None:
        self._env = env
        self._recorder = _load_recording(recording)
        self._frames = list(self._recorder.frames)
        self._threshold = dict(divergence_threshold or {"joint_pos": 0.01, "ee_pos": 0.005})
        self._on_divergence = on_divergence
        self._metadata = dict(getattr(self._recorder, "metadata", {}) or {})

    def play(self) -> ReplayReport:
        report = ReplayReport()
        seed = self._metadata.get("seed")
        _obs, _info = self._env.reset(seed=seed if seed is not None else None)

        for i, frame in enumerate(self._frames):
            from robodeploy.demo_recording import _action_from_json

            action = _action_from_json(frame.action)
            live_obs, _reward, _done, _info = self._env.step(action)
            recorded = _obs_from_recorded(frame.observation)
            div = self._compute_divergence(live_obs, recorded)
            threshold_val = max(self._threshold.values()) if self._threshold else 0.0
            max_div = max(div.values()) if div else 0.0
            exceeded = max_div > threshold_val
            report.add(i, div, exceeded=exceeded)
            report.steps_played = i + 1
            if exceeded:
                msg = f"Replay divergence at step {i}: {div}"
                if self._on_divergence == "halt":
                    report.halted = True
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    break
                if self._on_divergence == "warn":
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return report

    def _compute_divergence(self, live_obs, recorded: dict) -> dict[str, float]:
        rec_jp = recorded.get("joint_positions")
        rec_ee = recorded.get("ee_position")
        return {
            "joint_pos": _vec_delta(getattr(live_obs, "joint_positions", None), rec_jp),
            "ee_pos": _vec_delta(getattr(live_obs, "ee_position", None), rec_ee),
        }
