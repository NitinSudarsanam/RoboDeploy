"""Load the MuJoCo Universe catalog YAML."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

CATALOG_FILE = Path(__file__).with_name("mujoco_catalog.yaml")


def load_catalog(*, catalog_file: Path | None = None) -> dict[str, Any]:
    path = catalog_file or CATALOG_FILE
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Catalog must be a mapping: {path}")
    return data


def list_geom_kinds(*, catalog_file: Path | None = None) -> list[str]:
    return list(load_catalog(catalog_file=catalog_file).get("geom_kinds", []))


def list_robots(*, catalog_file: Path | None = None) -> list[str]:
    return sorted(load_catalog(catalog_file=catalog_file).get("robots", {}).keys())


def list_tasks(*, catalog_file: Path | None = None) -> list[str]:
    return sorted(load_catalog(catalog_file=catalog_file).get("tasks", {}).keys())


def list_policies(*, catalog_file: Path | None = None) -> list[str]:
    return sorted(load_catalog(catalog_file=catalog_file).get("policies", {}).keys())


def list_sensor_rigs(*, catalog_file: Path | None = None) -> list[str]:
    return sorted(load_catalog(catalog_file=catalog_file).get("sensor_rigs", {}).keys())


def list_combos(*, catalog_file: Path | None = None) -> list[str]:
    return sorted(load_catalog(catalog_file=catalog_file).get("combos", {}).keys())


def get_robot(name: str, *, catalog_file: Path | None = None) -> dict[str, Any]:
    robots = load_catalog(catalog_file=catalog_file).get("robots", {})
    if name not in robots:
        raise KeyError(f"Unknown robot '{name}'. Known: {sorted(robots)}")
    return dict(robots[name])


def get_task(name: str, *, catalog_file: Path | None = None) -> dict[str, Any]:
    tasks = load_catalog(catalog_file=catalog_file).get("tasks", {})
    if name not in tasks:
        raise KeyError(f"Unknown task '{name}'. Known: {sorted(tasks)}")
    return dict(tasks[name])


def get_policy(name: str, *, catalog_file: Path | None = None) -> dict[str, Any]:
    policies = load_catalog(catalog_file=catalog_file).get("policies", {})
    if name not in policies:
        raise KeyError(f"Unknown policy '{name}'. Known: {sorted(policies)}")
    return dict(policies[name])


def get_sensor_rig(name: str, *, catalog_file: Path | None = None) -> dict[str, Any]:
    rigs = load_catalog(catalog_file=catalog_file).get("sensor_rigs", {})
    if name not in rigs:
        raise KeyError(f"Unknown sensor rig '{name}'. Known: {sorted(rigs)}")
    return deepcopy(rigs[name])


def get_combo(name: str, *, catalog_file: Path | None = None) -> dict[str, Any]:
    combos = load_catalog(catalog_file=catalog_file).get("combos", {})
    if name not in combos:
        raise KeyError(f"Unknown combo '{name}'. Known: {sorted(combos)}")
    return dict(combos[name])


def find_combo(
    *,
    robot: str,
    task: str,
    policy: str,
    rig: str,
    catalog_file: Path | None = None,
) -> str | None:
    """Return catalog combo name matching robot/task/policy/rig, if any."""
    for name in list_combos(catalog_file=catalog_file):
        entry = get_combo(name, catalog_file=catalog_file)
        if (
            entry.get("robot") == robot
            and entry.get("task") == task
            and entry.get("policy") == policy
            and entry.get("rig") == rig
        ):
            return name
    return None


def build_config(
    *,
    robot: str,
    task: str,
    policy: str,
    rig: str,
    viewer: bool = False,
    catalog_file: Path | None = None,
) -> dict[str, Any]:
    """Build a RoboEnv.from_config dict; presets.yaml is canonical when a combo exists."""
    from examples.config import load_example_preset

    combo_name = find_combo(
        robot=robot,
        task=task,
        policy=policy,
        rig=rig,
        catalog_file=catalog_file,
    )
    if combo_name is not None:
        preset_name = str(get_combo(combo_name, catalog_file=catalog_file)["preset"])
        cfg = load_example_preset(preset_name)
    else:
        robot_entry = get_robot(robot, catalog_file=catalog_file)
        task_entry = get_task(task, catalog_file=catalog_file)
        rig_entry = get_sensor_rig(rig, catalog_file=catalog_file)
        prop_names = list(task_entry.get("prop_names", []))
        if "prop_pose" in rig_entry:
            rig_entry["prop_pose"] = dict(rig_entry.get("prop_pose") or {})
            rig_entry["prop_pose"]["prop_names"] = prop_names
        custom_modules = ["examples.tasks", "examples.sensors", "examples.policies"]
        if robot == "example_franka_mujoco":
            custom_modules.append("examples.franka_pick_place_mujoco.components")
        task_kwargs: dict[str, Any] = {}
        if prop_names:
            task_kwargs["require_objects"] = True
        if task == "showcase_scene":
            task_kwargs["require_rgb"] = True
        cfg = {
            "robot": robot,
            "backend": "mujoco",
            "task": task,
            "policy": policy,
            "task_kwargs": task_kwargs,
            "obs_spec_policy": "raise",
            "backend_kwargs": {
                "config": {
                    "allow_actuator_name_fallback": True,
                    "enable_viewer": viewer,
                }
            },
            "sensor_rigs": [
                {
                    "rig_id": "arm_sensors",
                    "ee_link": robot_entry.get("ee_link") or "robot0/ee_link",
                    **rig_entry,
                }
            ],
            "custom_modules": custom_modules,
        }
        return cfg

    cfg = dict(cfg)
    backend_kwargs = dict(cfg.get("backend_kwargs") or {})
    config = dict(backend_kwargs.get("config") or {})
    config["enable_viewer"] = viewer
    backend_kwargs["config"] = config
    cfg["backend_kwargs"] = backend_kwargs
    return cfg
