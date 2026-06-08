"""Safety check / test / status CLI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from robodeploy.cli_helpers import print_json


def _default_presets_file() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "config" / "presets.yaml"


def _load_joint_limits_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for --joint-limits.") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Joint limits file must be a mapping: {path}")
    return raw


def _resolve_robot_description(robot_name: str):
    from robodeploy.builtins import import_builtins
    from robodeploy.core.registry import get_robot

    import_builtins()
    return get_robot(str(robot_name))()


def _build_dummy_env():
    from robodeploy.core.robot import Robot, RobotTask
    from robodeploy.env import RoboEnv
    from robodeploy.testing import DummyBackend, DummyPolicy, DummyRobot, DummyTask

    robot = Robot(
        robot_id="robot0",
        description=DummyRobot(),
        tasks={"task0": RobotTask(task=DummyTask(), policies={"p": DummyPolicy(0.0)})},
    )
    return RoboEnv(backend=DummyBackend(), robots=[robot])


def _build_env_from_preset(preset: str, *, presets_file: Path | None) -> Any:
    from robodeploy.presets_loader import resolve_preset
    from robodeploy.env import RoboEnv
    from robodeploy.evaluation.env_builder import build_env_from_preset, is_dummy_preset

    path = presets_file or _default_presets_file()
    cfg = resolve_preset(preset, presets_file=path).to_dict()
    if is_dummy_preset(cfg):
        from robodeploy.evaluation.env_builder import _make_dummy_env

        return _make_dummy_env(cfg)
    return RoboEnv.from_config(cfg)


def _parse_injection(spec: str) -> tuple[str, dict[str, Any]]:
    text = str(spec).strip()
    if "=" not in text:
        raise ValueError(f"Invalid --inject value {spec!r}; expected name=value")
    name, value = text.split("=", 1)
    name = name.strip().lower()
    value = value.strip()

    if name == "force_spike":
        mag = float(re.sub(r"[^0-9.+-]", "", value) or "0")
        return name, {"magnitude_N": mag}
    if name == "collision":
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError("collision inject expects body_a,body_b")
        return name, {"body_a": parts[0], "body_b": parts[1]}
    if name == "joint_excursion":
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError("joint_excursion inject expects joint_idx,magnitude_rad")
        return name, {"joint_idx": int(parts[0]), "magnitude_rad": float(parts[1])}
    if name == "human_proximity":
        dist = float(re.sub(r"[^0-9.+-]", "", value) or "0")
        return name, {"distance_m": dist}
    if name == "singularity":
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 3:
            raise ValueError("singularity inject expects joint_idx,position_rad,velocity_rad_s")
        return name, {
            "joint_idx": int(parts[0]),
            "position_rad": float(parts[1]),
            "velocity_rad_s": float(parts[2]),
        }
    if name == "state_timeout":
        seconds = float(re.sub(r"[^0-9.+-]", "", value) or "0")
        return name, {"duration_s": seconds}
    raise ValueError(f"Unknown injection {name!r}")


def _apply_injection(injector, name: str, payload: dict[str, Any]) -> None:
    if name == "force_spike":
        injector.force_spike(float(payload["magnitude_N"]))
    elif name == "collision":
        injector.collision(str(payload["body_a"]), str(payload["body_b"]))
    elif name == "joint_excursion":
        injector.joint_limit_excursion(int(payload["joint_idx"]), float(payload["magnitude_rad"]))
    elif name == "human_proximity":
        injector.human_proximity(float(payload["distance_m"]))
    elif name == "singularity":
        injector.singularity(
            int(payload["joint_idx"]),
            position_rad=float(payload["position_rad"]),
            velocity_rad_s=float(payload["velocity_rad_s"]),
        )
    elif name == "state_timeout":
        injector.state_timeout(float(payload["duration_s"]))


def cmd_safety_check(
    *,
    preset: str | None,
    robot: str | None,
    joint_limits: Path | None,
    presets_file: Path | None,
    as_json: bool,
    pretty: bool,
) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    guards: list[str] = []

    robot_name = robot
    backend_name = None
    if preset:
        from robodeploy.presets_loader import resolve_preset

        cfg = resolve_preset(preset, presets_file=presets_file or _default_presets_file())
        robot_name = robot_name or str(cfg.robot)
        backend_name = str(cfg.backend)

    if not robot_name:
        robot_name = "dummy"

    try:
        desc = _resolve_robot_description(robot_name)
    except Exception as exc:
        errors.append(f"robot resolution failed: {exc}")
        payload = {"ok": False, "errors": errors, "warnings": warnings}
        if as_json:
            print_json(payload, pretty=pretty)
        else:
            for err in errors:
                print(f"ERROR: {err}")
        return 1

    limits = np.asarray(desc.joint_position_limits, dtype=np.float64)
    vel = np.asarray(desc.joint_velocity_limits, dtype=np.float64)
    if joint_limits is not None:
        override = _load_joint_limits_yaml(joint_limits)
        if "joint_position_limits" in override:
            limits = np.asarray(override["joint_position_limits"], dtype=np.float64)
        if "joint_velocity_limits" in override:
            vel = np.asarray(override["joint_velocity_limits"], dtype=np.float64)
        if limits.shape[0] != desc.dof:
            errors.append(
                f"joint limits dof mismatch: yaml {limits.shape[0]} vs robot {desc.dof}"
            )
        for idx, (lo, hi) in enumerate(limits):
            if lo >= hi:
                errors.append(f"joint {idx}: min {lo} >= max {hi}")
    else:
        for idx, (lo, hi) in enumerate(limits):
            if lo >= hi:
                errors.append(f"joint {idx}: min {lo} >= max {hi}")

    guards.extend(
        [
            "SafetyFilterGuard",
            "ForceLimitGuard",
            "VelocityGuard",
            "EStopGuard",
            "HumanProximityGuard (optional)",
            "SingularityGuard (optional)",
        ]
    )
    if backend_name and "real" in backend_name.lower():
        guards.extend(["JointLimitGuard", "Watchdog", "ROS2RecoveryManager"])
        warnings.append("Real backend: verify hardware e-stop circuit independently.")
    elif backend_name and "mujoco" in backend_name.lower():
        guards.append("CollisionGuard (sim)")

    payload = {
        "ok": not errors,
        "preset": preset,
        "robot": robot_name,
        "backend": backend_name,
        "dof": int(desc.dof),
        "joint_position_limits": limits.tolist(),
        "joint_velocity_limits": vel.tolist(),
        "guards": guards,
        "errors": errors,
        "warnings": warnings,
    }
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        status = "OK" if not errors else "FAIL"
        print(f"{status}: safety check for robot={robot_name}")
        if preset:
            print(f"  preset: {preset}")
        print(f"  dof: {desc.dof}")
        for warn in warnings:
            print(f"  WARN: {warn}")
        for err in errors:
            print(f"  ERROR: {err}")
        if not errors:
            print(f"  guards: {', '.join(guards)}")
    return 0 if not errors else 1


def cmd_safety_test(
    *,
    preset: str | None,
    inject: list[str],
    steps: int,
    presets_file: Path | None,
    as_json: bool,
    pretty: bool,
) -> int:
    from robodeploy.safety import (
        ForceLimitGuard,
        HumanProximityGuard,
        SafetyMonitor,
        SafetyViolationInjector,
        SingularityGuard,
    )

    if preset:
        env = _build_env_from_preset(preset, presets_file=presets_file)
    else:
        env = _build_dummy_env()

    desc = env.primary_robot.description
    injector = SafetyViolationInjector()
    collision_violations: list = []
    for spec in inject:
        name, payload = _parse_injection(spec)
        if name == "collision":
            injector.collision(str(payload["body_a"]), str(payload["body_b"]))
            collision_violations = injector.synthetic_violations()
        else:
            _apply_injection(injector, name, payload)

    monitor = SafetyMonitor(
        guards=[
            ForceLimitGuard(max_force_N=50.0, over_limit_strikes=3),
            HumanProximityGuard(min_distance_m=0.25),
            SingularityGuard(joint_position_limits=desc.joint_position_limits),
        ],
        on_violation="clamp",
        on_critical="raise",
    )

    obs, _info = env.reset()
    tripped = False
    last_hazard = None
    history = 0
    for _ in range(max(int(steps), 1)):
        obs = injector.apply(obs)
        try:
            if collision_violations:
                for violation in collision_violations:
                    monitor._handle([violation])  # noqa: SLF001 — CLI test harness
                collision_violations = []
            monitor.check_observation(obs)
        except Exception as exc:
            tripped = True
            violation = getattr(exc, "violation", None)
            if violation is not None:
                last_hazard = violation.hazard.name
            break
        history = len(monitor.violations())

    payload = {
        "ok": True,
        "preset": preset or "dummy",
        "inject": list(inject),
        "steps_run": int(steps),
        "tripped": tripped,
        "history_count": history,
        "last_hazard": last_hazard,
        "violations": [
            {
                "hazard": v.hazard.name,
                "severity": v.severity.name,
                "message": v.message,
            }
            for v in monitor.violations()[-5:]
        ],
    }
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        print(f"tripped={tripped} history={history} last_hazard={last_hazard}")
        for row in payload["violations"]:
            print(f"  {row['severity']}: {row['hazard']} — {row['message']}")
    env.close()
    return 0


def cmd_safety_status(*, as_json: bool, pretty: bool) -> int:
    from robodeploy.safety.registry import get_active_safety_label, get_active_safety_monitor

    monitor = get_active_safety_monitor()
    if monitor is None:
        payload = {"active": False, "message": "No running RoboEnv safety monitor registered."}
        if as_json:
            print_json(payload, pretty=pretty)
        else:
            print(payload["message"])
        return 0

    status = monitor.status()
    payload = {
        "active": True,
        "label": get_active_safety_label(),
        "tripped": status.tripped,
        "history_count": status.history_count,
        "estop_tripped": monitor.estop.tripped,
        "last_violation": None,
        "active_violations": [
            {
                "hazard": v.hazard.name,
                "severity": v.severity.name,
                "message": v.message,
            }
            for v in status.active_violations
        ],
    }
    if status.last_violation is not None:
        v = status.last_violation
        payload["last_violation"] = {
            "hazard": v.hazard.name,
            "severity": v.severity.name,
            "message": v.message,
        }
    if as_json:
        print_json(payload, pretty=pretty)
    else:
        print(f"active monitor: {payload['label']}")
        print(f"  tripped={payload['tripped']} estop={payload['estop_tripped']} history={payload['history_count']}")
        if payload["last_violation"]:
            lv = payload["last_violation"]
            print(f"  last: {lv['hazard']} ({lv['severity']}) — {lv['message']}")
    return 0
