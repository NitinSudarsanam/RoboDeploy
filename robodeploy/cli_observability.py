"""CLI handlers for observability, replay, and run manifests."""

from __future__ import annotations

import json
import time
from pathlib import Path

from robodeploy.cli_helpers import action_fn_for_mode, close_quietly, print_json


def add_observability_parsers(sub) -> None:  # noqa: ANN001
    p_logs = sub.add_parser("logs", help="Inspect JSONL run logs.")
    logs_sub = p_logs.add_subparsers(dest="logs_cmd", required=True)
    p_tail = logs_sub.add_parser("tail", help="Follow a JSONL log file.")
    p_tail.add_argument("path", help="Path to JSONL log or run directory.")
    p_tail.add_argument("--interval", type=float, default=0.5, help="Poll interval seconds.")
    p_tail.add_argument("--max-lines", type=int, default=0, help="Stop after N new lines (0 = until interrupted).")
    p_summary = logs_sub.add_parser("summary", help="Summarize a JSONL log file.")
    p_summary.add_argument("path", help="Path to JSONL log or run directory.")
    p_summary.add_argument("--json", action="store_true", help="Print summary as JSON.")

    p_replay = sub.add_parser("replay", help="Replay a recorded demo through an env.")
    p_replay.add_argument("recording", help="Path to demo JSON/JSONL.")
    p_replay.add_argument("--preset", default=None, help="YAML preset name (examples/config/presets.yaml).")
    p_replay.add_argument("--presets-file", default=None, help="Optional path to presets.yaml.")
    p_replay.add_argument("--dummy", action="store_true", help="Use built-in dummy env.")
    p_replay.add_argument("--seed", type=int, default=None, help="Override replay seed.")
    p_replay.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    p_replay.add_argument(
        "--pause-at-step",
        type=int,
        default=-1,
        help="Stop after this step index (-1 = play all).",
    )
    p_replay.add_argument("--diff", action="store_true", help="Compute observation divergence report.")
    p_replay.add_argument("--output", default="", help="Write replay/diff report JSON here.")
    p_replay.add_argument("--on-divergence", choices=("warn", "halt", "record"), default="warn")
    p_replay.add_argument("--json", action="store_true", help="Print report JSON to stdout.")

    p_manifest = sub.add_parser("manifest", help="Show a run manifest.")
    manifest_sub = p_manifest.add_subparsers(dest="manifest_cmd", required=True)
    p_manifest_show = manifest_sub.add_parser("show", help="Display manifest.json.")
    p_manifest_show.add_argument("path", help="manifest.json or run directory.")
    p_manifest_show.add_argument("--json", action="store_true", help="Print raw manifest JSON.")

    p_snapshot = sub.add_parser("snapshot", help="Save or restore env snapshots.")
    snap_sub = p_snapshot.add_subparsers(dest="snapshot_cmd", required=True)
    p_snap_save = snap_sub.add_parser("save", help="Capture snapshots to a pickle file.")
    p_snap_save.add_argument("path", help="Output .pkl path.")
    p_snap_save.add_argument("--dummy", action="store_true", help="Use built-in dummy env.")
    p_snap_save.add_argument("--steps", type=int, default=5, help="Steps to capture after reset.")
    p_snap_restore = snap_sub.add_parser("restore", help="Load snapshots from pickle (inspect only).")
    p_snap_restore.add_argument("path", help="Input .pkl path.")
    p_snap_restore.add_argument("--json", action="store_true", help="Print snapshot metadata as JSON.")


def _resolve_jsonl(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        candidates = sorted(p.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No .jsonl files under {p}")
        return candidates[0]
    return p


def cmd_logs_tail(*, path: str, interval: float, max_lines: int) -> int:
    log_path = _resolve_jsonl(path)
    seen = 0
    emitted = 0
    with log_path.open("r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if not line:
                time.sleep(max(0.05, float(interval)))
                continue
            seen += 1
            print(line.rstrip())
            emitted += 1
            if max_lines > 0 and emitted >= max_lines:
                break
    return 0


def cmd_logs_summary(*, path: str, as_json: bool) -> int:
    log_path = _resolve_jsonl(path)
    rewards: list[float] = []
    kinds: dict[str, int] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kinds[rec.get("kind", "step")] = kinds.get(rec.get("kind", "step"), 0) + 1
        payload = rec.get("payload") or {}
        if "reward" in payload:
            rewards.append(float(payload["reward"]))
    summary = {
        "path": str(log_path),
        "lines": sum(kinds.values()),
        "kinds": kinds,
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
    }
    if as_json:
        print_json(summary, pretty=False)
    else:
        print(f"log: {summary['path']}")
        print(f"lines: {summary['lines']}")
        print(f"kinds: {summary['kinds']}")
        if rewards:
            print(
                f"reward: mean={summary['reward_mean']:.4f} "
                f"min={summary['reward_min']:.4f} max={summary['reward_max']:.4f}"
            )
    return 0


def _make_dummy_env():
    from robodeploy.core.robot import Robot, RobotTask
    from robodeploy.env import RoboEnv
    from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

    robot = Robot(
        robot_id="robot0",
        description=DummyRobot(),
        tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
    )
    return RoboEnv(backend=DummyBackend(), robots=[robot])


def cmd_replay(
    *,
    recording: str,
    preset: str | None = None,
    presets_file: str | None = None,
    dummy: bool,
    seed: int | None,
    speed: float = 1.0,
    pause_at_step: int | None = None,
    diff: bool,
    output: str,
    on_divergence: str,
    as_json: bool,
) -> int:
    if diff:
        if not dummy:
            raise ValueError("replay --diff requires --dummy.")
        from robodeploy.observability.replay import TrajectoryReplayer

        env = _make_dummy_env()
        try:
            if seed is not None:
                env.reset(seed=int(seed))
            else:
                env.reset()
            replayer = TrajectoryReplayer(
                env=env,
                recording=recording,
                on_divergence=on_divergence,  # type: ignore[arg-type]
            )
            report = replayer.play()
            payload = report.to_dict()
            if output:
                report.save(output)
            if as_json or not output:
                print_json(payload, pretty=False)
            return 0
        finally:
            close_quietly(env)

    if preset or dummy:
        from robodeploy.cli_teleop import cmd_demo_replay

        return cmd_demo_replay(
            recording=recording,
            preset=preset,
            presets_file=presets_file,
            dummy=bool(dummy),
            speed=float(speed),
            pause_at_step=pause_at_step,
            as_json=as_json,
        )
    raise ValueError("replay requires --preset or --dummy.")


def cmd_manifest_show(*, path: str, as_json: bool) -> int:
    from robodeploy.observability.manifest import RunManifest

    p = Path(path)
    if p.is_dir():
        p = p / "manifest.json"
    manifest = RunManifest.load(p)
    if as_json:
        print_json(manifest.__dict__, pretty=False)
        return 0
    print(f"run_name: {manifest.run_name}")
    print(f"seed: {manifest.seed}")
    print(f"backend: {manifest.backend}")
    print(f"robot: {manifest.robot}")
    print(f"task: {manifest.task}")
    print(f"policy: {manifest.policy}")
    print(f"git: {manifest.git_hash} dirty={manifest.git_dirty}")
    print(f"python: {manifest.python_version}")
    print(f"package: {manifest.package_version}")
    print(f"sensors: {', '.join(manifest.sensor_rig) or '(none)'}")
    return 0


def cmd_snapshot_save(*, path: str, dummy: bool, steps: int) -> int:
    if not dummy:
        raise ValueError("snapshot save requires --dummy.")
    from robodeploy.observability.snapshot import SnapshotManager

    env = _make_dummy_env()
    mgr = SnapshotManager(env=env)
    try:
        env.reset(seed=0)
        mgr.capture()
        action_fn = action_fn_for_mode("hold", env)
        for _ in range(int(steps)):
            action = action_fn(None) if action_fn else None
            env.step(action)
            mgr.capture(last_action=action)
        mgr.save(path)
        print(path)
        return 0
    finally:
        close_quietly(env)


def cmd_snapshot_restore(*, path: str, as_json: bool) -> int:
    from robodeploy.observability.snapshot import SnapshotManager

    env = _make_dummy_env()
    mgr = SnapshotManager(env=env)
    try:
        mgr.load(path)
        meta = [
            {
                "step": s.step,
                "episode_id": s.episode_id,
                "timestamp": s.timestamp,
                "has_sim_state": s.sim_state is not None,
            }
            for s in mgr.snapshots
        ]
        if as_json:
            print_json({"count": len(meta), "snapshots": meta}, pretty=False)
        else:
            print(f"loaded {len(meta)} snapshots from {path}")
        return 0
    finally:
        close_quietly(env)


def dispatch_observability(cmd: str, args) -> int:  # noqa: ANN001
    if cmd == "logs":
        if str(args.logs_cmd) == "tail":
            return cmd_logs_tail(
                path=str(args.path),
                interval=float(args.interval),
                max_lines=int(args.max_lines),
            )
        if str(args.logs_cmd) == "summary":
            return cmd_logs_summary(path=str(args.path), as_json=bool(args.json))
        raise RuntimeError(f"Unknown logs subcommand: {args.logs_cmd}")
    if cmd == "replay":
        pause = int(getattr(args, "pause_at_step", -1))
        return cmd_replay(
            recording=str(args.recording),
            preset=getattr(args, "preset", None),
            presets_file=getattr(args, "presets_file", None),
            dummy=bool(args.dummy),
            seed=args.seed,
            speed=float(getattr(args, "speed", 1.0)),
            pause_at_step=None if pause < 0 else pause,
            diff=bool(args.diff),
            output=str(args.output),
            on_divergence=str(args.on_divergence),
            as_json=bool(args.json),
        )
    if cmd == "manifest":
        if str(args.manifest_cmd) == "show":
            return cmd_manifest_show(path=str(args.path), as_json=bool(args.json))
        raise RuntimeError(f"Unknown manifest subcommand: {args.manifest_cmd}")
    if cmd == "snapshot":
        if str(args.snapshot_cmd) == "save":
            return cmd_snapshot_save(path=str(args.path), dummy=bool(args.dummy), steps=int(args.steps))
        if str(args.snapshot_cmd) == "restore":
            return cmd_snapshot_restore(path=str(args.path), as_json=bool(args.json))
        raise RuntimeError(f"Unknown snapshot subcommand: {args.snapshot_cmd}")
    raise RuntimeError(f"Unknown observability command: {cmd}")
