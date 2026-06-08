"""MuJoCo Universe — one entry point for robots, tasks, policies, and sensor rigs."""

from __future__ import annotations

import argparse
import sys

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from examples.catalog.load import (  # noqa: E402
    build_config,
    get_combo,
    get_policy,
    get_robot,
    get_task,
    list_combos,
    list_geom_kinds,
    list_policies,
    list_robots,
    list_sensor_rigs,
    list_tasks,
    load_catalog,
)
from examples.env_from_preset import env_from_preset  # noqa: E402
from examples.mujoco_universe.artifacts import write_showcase_artifacts  # noqa: E402
from robodeploy.env import RoboEnv  # noqa: E402


OUTPUT_DIR = __import__("pathlib").Path(__file__).resolve().parent / "output"


def _print_list() -> None:
    catalog = load_catalog()
    print("=== MuJoCo Universe catalog ===")
    print("\nGeom kinds:")
    for kind in list_geom_kinds():
        print(f"  - {kind}")
    print("\nRobots:")
    for name in list_robots():
        entry = catalog["robots"][name]
        print(f"  - {name}: ee_link={entry.get('ee_link', '')} — {entry.get('notes', '')}")
    print("\nTasks:")
    for name in list_tasks():
        entry = catalog["tasks"][name]
        kinds = ", ".join(entry.get("geom_kinds", []))
        print(f"  - {name}: geoms=[{kinds}] — {entry.get('notes', '')}")
    print("\nPolicies:")
    for name in list_policies():
        entry = catalog["policies"][name]
        needs = ", ".join(entry.get("needs", [])) or "(none)"
        ik = "yes" if entry.get("needs_ik") else "no"
        print(f"  - {name}: needs=[{needs}] ik={ik} — {entry.get('notes', '')}")
    print("\nSensor rigs:")
    for name in list_sensor_rigs():
        keys = [k for k in catalog["sensor_rigs"][name] if k != "prop_pose"]
        print(f"  - {name}: {', '.join(keys) or 'prop_pose only'}")
    print("\nNamed combos (preset shortcuts):")
    for name in list_combos():
        combo = catalog["combos"][name]
        print(
            f"  - {name}: robot={combo['robot']} task={combo['task']} "
            f"policy={combo['policy']} rig={combo['rig']} preset={combo.get('preset', '')}"
        )


def _scene_summary(task_name: str) -> str:
    task_entry = get_task(task_name)
    props = task_entry.get("prop_names", [])
    geoms = task_entry.get("geom_kinds", [])
    return f"props={props} geom_kinds={geoms}"


def _build_env(
    *,
    preset: str | None,
    robot: str,
    task: str,
    policy: str,
    rig: str,
    viewer: bool,
    steps: int,
) -> tuple[RoboEnv, dict[str, str]]:
    meta = {"robot": robot, "task": task, "policy": policy, "rig": rig}
    if preset:
        env = env_from_preset(
            preset,
            backend_kwargs={"config": {"allow_actuator_name_fallback": True, "enable_viewer": viewer}},
            max_episode_steps=steps,
        )
        combo = get_combo(preset) if preset in list_combos() else {}
        meta.update({k: str(combo.get(k, meta[k])) for k in meta})
        return env, meta

    cfg = build_config(robot=robot, task=task, policy=policy, rig=rig, viewer=viewer)
    cfg["max_episode_steps"] = steps
    env = RoboEnv.from_config(cfg)
    return env, meta


def _run_episode(
    env: RoboEnv,
    *,
    meta: dict[str, str],
    steps: int,
    log_every: int,
    rotate_policies: bool,
    skip_render: bool,
) -> tuple[object, str]:
    lines: list[str] = [
        "MuJoCo Universe run",
        f"robot={meta['robot']} task={meta['task']} policy={meta['policy']} rig={meta['rig']}",
        f"scene: {_scene_summary(meta['task'])}",
    ]
    print("\n".join(lines))
    print()

    obs, _ = env.reset()

    policies = [meta["policy"]]
    if rotate_policies:
        policies = ["example_joint_track", "example_sensor_reach_pick"]
        print("Policy rotation:", " -> ".join(policies))

    per_policy = max(1, steps // len(policies)) if rotate_policies else steps
    step_idx = 0
    for pol in policies:
        if rotate_policies and pol != meta["policy"]:
            from robodeploy.core.registry import get_policy

            policy_obj = get_policy(pol)()
            for robot in env.robots:
                for robot_task in robot.tasks.values():
                    robot_task.policies.clear()
                    robot_task.policies["policy0"] = policy_obj
            env._bind_policy_runtime()

        for _ in range(per_policy):
            obs, _, _, _ = env.step()
            if step_idx % log_every == 0:
                ft = (getattr(obs, "ft_forces", {}) or {}).get("wrist_ft")
                imu = getattr(obs, "imu_acceleration", None)
                intrinsics = list((getattr(obs, "camera_intrinsics", {}) or {}).keys())
                print(
                    f"step {step_idx:4d} status={getattr(obs, 'sensor_status', {})} "
                    f"objects={list(getattr(obs, 'objects', {}).keys())} "
                    f"intrinsics={intrinsics} "
                    f"ft={None if ft is None else [float(x) for x in ft]} "
                    f"imu={None if imu is None else [float(x) for x in imu]}"
                )
            step_idx += 1

    if skip_render:
        print("Skipping RGB montage (Windows / no GLFW).")
    return obs, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MuJoCo Universe flagship demo")
    parser.add_argument("--list", action="store_true", help="Print catalog and exit")
    parser.add_argument("--preset", default=None, help="YAML preset name (e.g. mujoco_showcase_kuka)")
    parser.add_argument("--robot", default="kuka")
    parser.add_argument("--task", default="showcase_scene")
    parser.add_argument("--policy", default="example_joint_track")
    parser.add_argument("--rig", default="full")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--viewer", action="store_true", help="Enable MuJoCo viewer")
    parser.add_argument("--rotate-policies", action="store_true", help="Rotate joint_track and sensor_reach_pick")
    args = parser.parse_args(argv)

    if args.list:
        _print_list()
        return 0

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print('Install MuJoCo: pip install -e ".[sim]"')
        return 1

    skip_render = sys.platform == "win32"
    if not skip_render:
        try:
            from robodeploy.sensors.camera.sim.mujoco_gl import ensure_mujoco_gl_backend

            ensure_mujoco_gl_backend()
        except Exception as exc:
            print(f"GL backend unavailable ({exc}); continuing without render.")
            skip_render = True

    robot = args.robot
    task = args.task
    policy = args.policy
    rig = args.rig
    if args.preset:
        combo = get_combo(args.preset)
        robot = combo.get("robot", robot)
        task = combo.get("task", task)
        policy = combo.get("policy", policy)
        rig = combo.get("rig", rig)

    if args.rotate_policies and task != "pick_place":
        print("Note: --rotate-policies works best on pick_place; continuing with current task.")

    try:
        env, meta = _build_env(
            preset=args.preset,
            robot=robot,
            task=task,
            policy=policy,
            rig=rig,
            viewer=args.viewer,
            steps=args.steps,
        )
    except Exception as exc:
        print("Env build failed:", exc)
        return 1

    try:
        obs, scene_text = _run_episode(
            env,
            meta=meta,
            steps=args.steps,
            log_every=args.log_every,
            rotate_policies=args.rotate_policies,
            skip_render=skip_render,
        )
        write_showcase_artifacts(obs, scene_text, OUTPUT_DIR, skip_render=skip_render)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
