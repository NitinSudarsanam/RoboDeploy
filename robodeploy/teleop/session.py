"""High-level teleop session helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from robodeploy.demo_recording import InteractiveDemoSession, load_demo_jsonl, replay_demo_frames
from robodeploy.teleop.controller import TeleopPolicy
from robodeploy.teleop.devices import make_teleop_device


def run_teleop_session(
    env,
    *,
    device: str = "keyboard",
    output_dir: str | Path | None = None,
    record_path: str | Path | None = None,
    fmt: Literal["jsonl", "hdf5", "json", "lerobot"] = "jsonl",
    start_recording: bool = False,
    max_steps: int | None = None,
    device_kwargs: dict | None = None,
) -> list[Path]:
    """Teleoperate env with a device-backed TeleopPolicy; optionally record episodes."""
    teleop_device = make_teleop_device(device, **(device_kwargs or {}))
    policy = TeleopPolicy(device=teleop_device)
    teleop_device.start()
    try:
        env.reset()
        policy.bind_runtime(env.backend, env.primary_robot.description)
        if record_path:
            target = Path(record_path)
            out_dir = target.parent if target.suffix else target
            recording = True
        else:
            out_dir = Path(output_dir or "demos")
            recording = bool(start_recording)
        session = InteractiveDemoSession(
            env,
            policy,
            output_dir=out_dir,
            fmt=fmt,
            start_recording=recording,
            max_steps=max_steps,
        )
        saved = session.run()
        if record_path and Path(record_path).suffix and saved:
            target = Path(record_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            saved[-1].replace(target)
            return [target]
        return saved
    finally:
        teleop_device.stop()


def replay_recording(
    env,
    path: str | Path,
    *,
    speed: float = 1.0,
    pause_at_step: int | None = None,
) -> int:
    """Replay a JSONL or JSON bundle recording through env.step."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        frames = load_demo_jsonl(path)
    else:
        from robodeploy.demo_recording import DemoRecorder

        frames = DemoRecorder.load(path).frames
    return replay_demo_frames(
        env,
        frames,
        speed=float(speed),
        pause_at_step=pause_at_step,
    )
