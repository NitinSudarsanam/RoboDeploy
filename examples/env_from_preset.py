"""Build a RoboEnv from an example YAML preset."""

from __future__ import annotations

from typing import Any

from robodeploy.env import RoboEnv

from examples.config import PRESETS_FILE, load_example_preset


def wire_mujoco_pick_policies(env: RoboEnv) -> None:
    """Attach MuJoCo IK to ``ReachPickPlacePolicy`` instances after ``env.reset()``."""
    from examples.policies.reach_pick_place import ReachPickPlacePolicy

    backend = env._backend
    if not hasattr(backend, "_model"):
        return
    for robot in env.robots:
        for robot_task in robot.tasks.values():
            for policy in robot_task.policies.values():
                if isinstance(policy, ReachPickPlacePolicy):
                    policy.attach_mujoco(backend, robot.description)


def env_from_preset(name: str, **overrides: Any) -> "RoboEnv":
    """Load ``examples/config/presets.yaml`` and construct via ``RoboEnv.make`` / ``from_config``."""
    from robodeploy.builtins import import_builtins
    from robodeploy.core.registry import use

    import_builtins()
    cfg = {**load_example_preset(name), **overrides}
    for mod in cfg.pop("custom_modules", []) or []:
        use(str(mod))
    if "robots" in cfg:
        return RoboEnv.from_config(cfg)
    return RoboEnv.make(
        robot=str(cfg["robot"]),
        backend=str(cfg["backend"]),
        task=str(cfg["task"]),
        policy=str(cfg["policy"]),
        robot_id=str(cfg.get("robot_id", "robot0")),
        task_id=str(cfg.get("task_id", "task0")),
        policy_id=str(cfg.get("policy_id", "policy0")),
        backend_kwargs=cfg.get("backend_kwargs"),
        task_kwargs=cfg.get("task_kwargs"),
        policy_kwargs=cfg.get("policy_kwargs"),
        sensor_kwargs=cfg.get("sensor_kwargs"),
    )


def example_presets_file() -> str:
    return str(PRESETS_FILE)
