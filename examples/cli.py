"""Examples CLI — preset-based list/run/export commands.

Demo presets live under ``examples/config/presets.yaml``, not in the robodeploy package.
Run: ``python -m examples.cli list-presets``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from robodeploy.cli_helpers import (
    action_fn_for_mode,
    close_quietly,
    episode_info_summary,
    print_json,
)


def _resolve_presets_file(explicit: str | None) -> Path:
    if explicit and str(explicit).strip():
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Presets file not found: {path}")
        return path
    env_path = os.environ.get("ROBODEPLOY_PRESETS_FILE", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ROBODEPLOY_PRESETS_FILE not found: {path}")
        return path
    default = Path(__file__).resolve().parent / "config" / "presets.yaml"
    if default.is_file():
        return default.resolve()
    raise FileNotFoundError(
        "No presets file found. Set ROBODEPLOY_PRESETS_FILE or pass --presets-file "
        f"(default: {default})."
    )


def _import_custom_modules(custom_modules: list[str]) -> None:
    if not custom_modules:
        return
    from robodeploy.core.registry import use

    for mod in custom_modules:
        use(str(mod))


def _make_env(
    *,
    preset: str,
    dummy: bool,
    presets_file: str | None,
    custom_modules: list[str],
):
    from robodeploy.env import RoboEnv

    if dummy:
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        return RoboEnv(backend=DummyBackend(), robots=[robot], safety_enabled=False)

    if not str(preset).strip():
        raise ValueError("--preset is required unless --dummy is set.")

    from robodeploy.builtins import import_builtins
    from examples.config import load_preset

    path = _resolve_presets_file(presets_file)
    import_builtins()
    cfg = load_preset(preset, presets_file=path)
    merged_modules = list(custom_modules) + [str(m) for m in (cfg.get("custom_modules") or [])]
    if merged_modules:
        cfg = {**cfg, "custom_modules": merged_modules}
    return RoboEnv.from_config(cfg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="examples.cli", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_presets = sub.add_parser("list-presets", help="List YAML preset names from examples/config/presets.yaml.")
    p_presets.add_argument(
        "--presets-file",
        default="",
        help="Path to presets YAML (default: ROBODEPLOY_PRESETS_FILE or examples/config/presets.yaml).",
    )
    p_presets.add_argument("--json", action="store_true", help="Print as JSON array.")
    p_presets.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_export = sub.add_parser("export-episode", help="Run a preset and export a recorded episode.")
    p_export.add_argument("--preset", default="", help="Preset name from examples/config/presets.yaml.")
    p_export.add_argument(
        "--presets-file",
        default="",
        help="Path to presets YAML (default: ROBODEPLOY_PRESETS_FILE or examples/config/presets.yaml).",
    )
    p_export.add_argument(
        "--custom-module",
        action="append",
        default=[],
        help="Import dotted module path(s) before running (register project components).",
    )
    p_export.add_argument("--steps", type=int, default=50, help="Number of env steps to run.")
    p_export.add_argument("--out", required=True, help="Output file path.")
    p_export.add_argument(
        "--format",
        choices=("jsonl", "hdf5"),
        default="jsonl",
        help="Export format.",
    )
    p_export.add_argument(
        "--dummy",
        action="store_true",
        help="Use built-in dummy backend/robot/task instead of a preset (no simulator required).",
    )
    p_export.add_argument(
        "--action",
        choices=("none", "zero", "hold", "sinusoid"),
        default="none",
        help="Inject explicit actions instead of using policy actions.",
    )
    p_export.add_argument("--json", action="store_true", help="Print a structured JSON result.")
    p_export.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_run = sub.add_parser("run-episode", help="Run an episode and print a small EpisodeInfo summary JSON.")
    p_run.add_argument("--preset", default="", help="Preset name from examples/config/presets.yaml.")
    p_run.add_argument(
        "--presets-file",
        default="",
        help="Path to presets YAML (default: ROBODEPLOY_PRESETS_FILE or examples/config/presets.yaml).",
    )
    p_run.add_argument(
        "--custom-module",
        action="append",
        default=[],
        help="Import dotted module path(s) before running (register project components).",
    )
    p_run.add_argument("--steps", type=int, default=50, help="Number of env steps to run.")
    p_run.add_argument(
        "--dummy",
        action="store_true",
        help="Use built-in dummy backend/robot/task instead of a preset (no simulator required).",
    )
    p_run.add_argument(
        "--action",
        choices=("none", "zero", "hold", "sinusoid"),
        default="none",
        help="Inject explicit actions instead of using policy actions.",
    )
    p_run.add_argument("--json", action="store_true", help="Print a structured JSON result.")
    p_run.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_teleop = sub.add_parser("teleop", help="Teleoperate a preset env (keyboard device).")
    p_teleop.add_argument("--preset", required=True, help="Preset name from examples/config/presets.yaml.")
    p_teleop.add_argument(
        "--presets-file",
        default="",
        help="Path to presets YAML (default: ROBODEPLOY_PRESETS_FILE or examples/config/presets.yaml).",
    )
    p_teleop.add_argument(
        "--custom-module",
        action="append",
        default=[],
        help="Import dotted module path(s) before running.",
    )
    p_teleop.add_argument(
        "--device",
        default="keyboard",
        help="Teleop device name (default: keyboard).",
    )
    p_teleop.add_argument(
        "--record",
        default="",
        help="Output path: directory or .jsonl/.hdf5/.json file for recorded episode(s).",
    )
    p_teleop.add_argument(
        "--format",
        choices=("jsonl", "hdf5", "json", "lerobot"),
        default="jsonl",
        help="Recording export format when --record is a directory.",
    )
    p_teleop.add_argument(
        "--start-recording",
        action="store_true",
        help="Begin recording immediately (when --record is a directory).",
    )
    p_teleop.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after N steps (0 = unlimited).",
    )
    p_teleop.add_argument("--json", action="store_true", help="Print saved episode paths as JSON.")
    p_teleop.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    p_replay = sub.add_parser("replay", help="Replay a recorded demo through a preset env.")
    p_replay.add_argument("demo", help="Path to .jsonl or .json demo file.")
    p_replay.add_argument("--preset", default="", help="Preset name from examples/config/presets.yaml.")
    p_replay.add_argument(
        "--dummy",
        action="store_true",
        help="Use built-in dummy backend/robot/task instead of a preset (no simulator required).",
    )
    p_replay.add_argument(
        "--presets-file",
        default="",
        help="Path to presets YAML (default: ROBODEPLOY_PRESETS_FILE or examples/config/presets.yaml).",
    )
    p_replay.add_argument(
        "--custom-module",
        action="append",
        default=[],
        help="Import dotted module path(s) before running.",
    )
    p_replay.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    p_replay.add_argument(
        "--pause-at-step",
        type=int,
        default=-1,
        help="Stop replay after this step index (-1 = play all).",
    )
    p_replay.add_argument("--json", action="store_true", help="Print replay summary as JSON.")
    p_replay.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    return parser


def _cmd_list_presets(*, presets_file: str, as_json: bool, pretty: bool) -> int:
    from examples.config import list_presets

    names = list_presets(presets_file=_resolve_presets_file(presets_file or None))
    if as_json:
        print_json(names, pretty=pretty)
    else:
        for name in names:
            print(name)
    return 0


def _cmd_export_episode(
    *,
    preset: str,
    presets_file: str,
    steps: int,
    out: str,
    fmt: str,
    dummy: bool,
    custom_modules: list[str],
    action_mode: str,
    as_json: bool,
    pretty: bool,
) -> int:
    env = _make_env(
        preset=preset,
        dummy=dummy,
        presets_file=presets_file or None,
        custom_modules=custom_modules,
    )
    out_path = Path(out)
    try:
        action_fn = action_fn_for_mode(action_mode, env)
        recorder = env.run_episode(int(steps), record=True, action_fn=action_fn)
        if fmt == "hdf5":
            from robodeploy.dataset_export import export_demo_hdf5

            export_demo_hdf5(recorder, out_path)
        else:
            from robodeploy.dataset_export import export_demo_jsonl

            export_demo_jsonl(recorder, out_path)
    finally:
        close_quietly(env)
    if as_json:
        payload = {
            "out": str(out_path),
            "format": str(fmt),
            "steps": int(steps),
            "dummy": bool(dummy),
            "preset": str(preset),
            "action": str(action_mode),
        }
        print_json(payload, pretty=pretty)
    else:
        print(str(out_path))
    return 0


def _cmd_teleop(
    *,
    preset: str,
    presets_file: str,
    custom_modules: list[str],
    device: str,
    record: str,
    fmt: str,
    max_steps: int,
    start_recording: bool,
    as_json: bool,
    pretty: bool,
) -> int:
    from robodeploy.teleop.session import run_teleop_session

    env = _make_env(
        preset=preset,
        dummy=False,
        presets_file=presets_file or None,
        custom_modules=custom_modules,
    )
    try:
        saved = run_teleop_session(
            env,
            device=device,
            record_path=record or None,
            fmt=fmt,  # type: ignore[arg-type]
            start_recording=bool(start_recording),
            max_steps=max_steps if max_steps > 0 else None,
        )
    finally:
        close_quietly(env)
    paths = [str(p) for p in saved]
    if as_json:
        print_json({"saved": paths, "preset": preset, "device": device}, pretty=pretty)
    else:
        for path in paths:
            print(path)
    return 0


def _cmd_replay(
    *,
    demo: str,
    preset: str,
    presets_file: str,
    custom_modules: list[str],
    dummy: bool,
    speed: float,
    pause_at_step: int,
    as_json: bool,
    pretty: bool,
) -> int:
    from robodeploy.teleop.session import replay_recording

    env = _make_env(
        preset=preset,
        dummy=dummy,
        presets_file=presets_file or None,
        custom_modules=custom_modules,
    )
    try:
        steps = replay_recording(
            env,
            demo,
            speed=speed,
            pause_at_step=None if pause_at_step < 0 else pause_at_step,
        )
    finally:
        close_quietly(env)
    if as_json:
        print_json(
            {"demo": demo, "preset": preset, "steps": steps, "speed": speed},
            pretty=pretty,
        )
    else:
        print(f"replayed {steps} steps from {demo}")
    return 0


def _cmd_run_episode(
    *,
    preset: str,
    presets_file: str,
    steps: int,
    dummy: bool,
    custom_modules: list[str],
    action_mode: str,
    pretty: bool,
    as_json: bool,
) -> int:
    from robodeploy.policies.remote.http_client import to_jsonable

    env = _make_env(
        preset=preset,
        dummy=dummy,
        presets_file=presets_file or None,
        custom_modules=custom_modules,
    )

    try:
        action_fn = action_fn_for_mode(action_mode, env)
        _, info = env.run_episode(int(steps), record=False, action_fn=action_fn)
        info_payload = to_jsonable(episode_info_summary(info))
        if as_json:
            payload = {
                "preset": str(preset),
                "dummy": bool(dummy),
                "steps": int(steps),
                "action": str(action_mode),
                "info": info_payload,
            }
            print_json(payload, pretty=pretty)
        else:
            print_json(info_payload, pretty=pretty)
        return 0
    finally:
        close_quietly(env)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = str(args.cmd)

    if cmd == "list-presets":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_list_presets(
            presets_file=str(args.presets_file),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )
    if cmd == "export-episode":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_export_episode(
            preset=str(args.preset),
            presets_file=str(args.presets_file),
            steps=int(args.steps),
            out=str(args.out),
            fmt=str(args.format),
            dummy=bool(args.dummy),
            custom_modules=list(args.custom_module or []),
            action_mode=str(args.action),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )
    if cmd == "run-episode":
        return _cmd_run_episode(
            preset=str(args.preset),
            presets_file=str(args.presets_file),
            steps=int(args.steps),
            dummy=bool(args.dummy),
            custom_modules=list(args.custom_module or []),
            action_mode=str(args.action),
            pretty=bool(args.pretty),
            as_json=bool(args.json),
        )
    if cmd == "teleop":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_teleop(
            preset=str(args.preset),
            presets_file=str(args.presets_file),
            custom_modules=list(args.custom_module or []),
            device=str(args.device),
            record=str(args.record),
            fmt=str(args.format),
            max_steps=int(args.max_steps),
            start_recording=bool(args.start_recording),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )
    if cmd == "replay":
        if bool(args.pretty) and not bool(args.json):
            raise ValueError("--pretty requires --json.")
        return _cmd_replay(
            demo=str(args.demo),
            preset=str(args.preset),
            presets_file=str(args.presets_file),
            custom_modules=list(args.custom_module or []),
            dummy=bool(args.dummy),
            speed=float(args.speed),
            pause_at_step=int(args.pause_at_step),
            as_json=bool(args.json),
            pretty=bool(args.pretty),
        )

    raise RuntimeError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    sys.exit(main())
