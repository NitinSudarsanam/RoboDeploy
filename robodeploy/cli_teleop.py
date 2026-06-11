"""CLI handlers for teleop sessions and demo replay."""

from __future__ import annotations

import sys
from pathlib import Path

from robodeploy.cli_helpers import close_quietly, print_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_examples_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _env_from_preset(name: str, *, presets_file: str | None = None):
    _ensure_examples_on_path()
    from robodeploy.presets_loader import resolve_preset
    from robodeploy.env import RoboEnv

    cfg = resolve_preset(name, presets_file=presets_file)
    return RoboEnv.from_config(cfg.to_dict())


def add_teleop_parsers(sub) -> None:  # noqa: ANN001
    p_teleop = sub.add_parser("teleop", help="Teleoperate a preset env with an input device.")
    p_teleop.add_argument("--preset", default="kuka_pick_mujoco", help="YAML preset name.")
    p_teleop.add_argument(
        "--presets-file",
        default=None,
        help="Optional path to presets.yaml (default: examples/config/presets.yaml).",
    )
    p_teleop.add_argument(
        "--device",
        default="keyboard",
        help="Teleop device: keyboard, spacemouse, gamepad, mujoco_mouse, ros2_twist, ros2_joy.",
    )
    p_teleop.add_argument(
        "--record",
        default="",
        help="Output .jsonl path or directory for recorded episodes.",
    )
    p_teleop.add_argument(
        "--format",
        choices=("jsonl", "hdf5", "json", "lerobot"),
        default="jsonl",
        help="Recording export format.",
    )
    p_teleop.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0 = unlimited).")
    p_teleop.add_argument("--start-recording", action="store_true", help="Begin recording immediately.")
    p_teleop.add_argument("--json", action="store_true", help="Print structured JSON result.")


def cmd_teleop(
    *,
    preset: str,
    presets_file: str | None,
    device: str,
    record: str,
    fmt: str,
    max_steps: int,
    start_recording: bool,
    as_json: bool,
) -> int:
    from robodeploy.teleop.session import run_teleop_session

    env = _env_from_preset(preset, presets_file=presets_file)
    try:
        saved = run_teleop_session(
            env,
            device=device,
            record_path=record or None,
            fmt=fmt,  # type: ignore[arg-type]
            start_recording=start_recording,
            max_steps=max_steps if max_steps > 0 else None,
            preset=preset,
        )
        payload = {"saved": [str(p) for p in saved], "preset": preset, "device": device}
        if as_json:
            print_json(payload, pretty=False)
        else:
            for path in saved:
                print(path)
        return 0
    finally:
        close_quietly(env)


def cmd_demo_replay(
    *,
    recording: str,
    preset: str | None,
    presets_file: str | None,
    dummy: bool,
    speed: float,
    pause_at_step: int | None,
    as_json: bool,
) -> int:
    if preset:
        env = _env_from_preset(preset, presets_file=presets_file)
    elif dummy:
        from robodeploy.cli_observability import _make_dummy_env

        env = _make_dummy_env()
    else:
        raise ValueError("replay requires --preset or --dummy.")

    from robodeploy.teleop.session import replay_recording

    try:
        steps = replay_recording(
            env,
            recording,
            speed=float(speed),
            pause_at_step=pause_at_step,
        )
        payload = {"steps": steps, "recording": recording, "preset": preset}
        if as_json:
            print_json(payload, pretty=False)
        else:
            print(f"replayed {steps} steps from {recording}")
        return 0
    finally:
        close_quietly(env)
