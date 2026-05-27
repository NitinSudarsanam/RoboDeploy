"""RoboDeploy CLI.

Intentionally thin wrapper over the public Python APIs. This keeps the CLI
useful for quick smoke checks and dataset export without creating a second
configuration system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robodeploy", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_presets = sub.add_parser("list-presets", help="List YAML preset names.")
    p_presets.add_argument("--json", action="store_true", help="Print as JSON array.")

    p_reg = sub.add_parser("list-registry", help="List registered component names.")
    p_reg.add_argument(
        "--discover",
        action="store_true",
        help="Load Python entry points before listing (pip-installed extensions).",
    )
    p_reg.add_argument(
        "--builtins",
        action="store_true",
        help="Import builtin modules before listing (populates robots/tasks/policies).",
    )
    p_reg.add_argument("--json", action="store_true", help="Print as JSON object.")

    p_export = sub.add_parser("export-episode", help="Run a preset and export a recorded episode.")
    p_export.add_argument("--preset", required=True, help="Preset name from robodeploy.config.")
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

    p_serve = sub.add_parser("serve-policy", help="Serve a registered policy via ZMQ or gRPC.")
    p_serve.add_argument(
        "--custom-module",
        action="append",
        default=[],
        help="Import dotted module path(s) before looking up policy.",
    )
    p_serve.add_argument("--policy", required=True, help="Registered policy name (e.g. vla_stub).")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host/interface.")
    p_serve.add_argument("--port", type=int, default=5555, help="Bind port.")
    p_serve.add_argument("--transport", choices=("zmq", "grpc"), default="zmq", help="Transport.")
    p_serve.add_argument("--quiet", action="store_true", help="Disable verbose request logging.")

    return parser


def _cmd_list_presets(*, as_json: bool) -> int:
    from robodeploy.config import list_presets

    names = list_presets()
    if as_json:
        print(json.dumps(names))
    else:
        for name in names:
            print(name)
    return 0


def _cmd_list_registry(*, discover: bool, builtins: bool, as_json: bool) -> int:
    if discover:
        from robodeploy import discover as _discover

        _discover()
    if builtins:
        from robodeploy.builtins import import_builtins

        import_builtins()
    from robodeploy.core.registry import list_registered

    payload = list_registered()
    if as_json:
        print(json.dumps(payload))
        return 0

    for group in ("backends", "robots", "tasks", "policies", "sensors", "sensor_pairs"):
        items = payload.get(group, [])
        print(f"{group}:")
        for name in items:
            print(f"  - {name}")
    return 0


def _cmd_export_episode(
    *,
    preset: str,
    steps: int,
    out: str,
    fmt: str,
    dummy: bool,
    custom_modules: list[str],
) -> int:
    from robodeploy.env import RoboEnv

    if custom_modules:
        from robodeploy.core.registry import use

        for mod in custom_modules:
            use(str(mod))

    if dummy:
        from robodeploy.core.robot import Robot, RobotTask
        from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

        robot = Robot(
            robot_id="robot0",
            description=DummyRobot(),
            tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
        )
        env = RoboEnv(backend=DummyBackend(), robots=[robot])
    else:
        env = RoboEnv.from_preset(preset)
    out_path = Path(out)
    try:
        recorder = env.run_episode(int(steps), record=True)
        if fmt == "hdf5":
            from robodeploy.dataset_export import export_demo_hdf5

            export_demo_hdf5(recorder, out_path)
        else:
            from robodeploy.dataset_export import export_demo_jsonl

            export_demo_jsonl(recorder, out_path)
    finally:
        try:
            env.close()
        except Exception:
            pass
    print(str(out_path))
    return 0


def _cmd_serve_policy(*, policy: str, host: str, port: int, transport: str, quiet: bool) -> int:
    from robodeploy.builtins import import_builtins
    from robodeploy.core.registry import get_policy
    from robodeploy.policies.remote.server import serve

    import_builtins()
    PolicyClass = get_policy(policy)
    serve(PolicyClass(), host=host, port=int(port), transport=str(transport), verbose=not quiet)
    return 0


def _cmd_serve_policy_with_modules(
    *,
    policy: str,
    host: str,
    port: int,
    transport: str,
    quiet: bool,
    custom_modules: list[str],
) -> int:
    if custom_modules:
        from robodeploy.core.registry import use

        for mod in custom_modules:
            use(str(mod))
    return _cmd_serve_policy(policy=policy, host=host, port=port, transport=transport, quiet=quiet)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = str(args.cmd)

    if cmd == "list-presets":
        return _cmd_list_presets(as_json=bool(args.json))
    if cmd == "list-registry":
        return _cmd_list_registry(
            discover=bool(args.discover),
            builtins=bool(args.builtins),
            as_json=bool(args.json),
        )
    if cmd == "export-episode":
        return _cmd_export_episode(
            preset=str(args.preset),
            steps=int(args.steps),
            out=str(args.out),
            fmt=str(args.format),
            dummy=bool(args.dummy),
            custom_modules=list(args.custom_module or []),
        )
    if cmd == "serve-policy":
        return _cmd_serve_policy_with_modules(
            policy=str(args.policy),
            host=str(args.host),
            port=int(args.port),
            transport=str(args.transport),
            quiet=bool(args.quiet),
            custom_modules=list(args.custom_module or []),
        )

    raise RuntimeError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    sys.exit(main())

