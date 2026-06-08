"""Construct RoboEnv instances from benchmark presets."""

from __future__ import annotations

from typing import Any, Callable

from robodeploy.env import RoboEnv


def apply_seed(preset: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = dict(preset)
    task_kwargs = dict(cfg.get("task_kwargs") or {})
    task_kwargs["random_seed"] = int(seed)
    cfg["task_kwargs"] = task_kwargs
    return cfg


def is_dummy_preset(preset: dict[str, Any]) -> bool:
    backend = str(preset.get("backend", "")).lower()
    return backend in {"dummy", "test"} or bool(preset.get("use_dummy_backend"))


def build_env_from_preset(preset: dict[str, Any], *, seed: int) -> RoboEnv:
    cfg = apply_seed(preset, seed)
    if is_dummy_preset(cfg):
        return _make_dummy_env(cfg)
    return RoboEnv.from_config(cfg)


def _make_dummy_env(cfg: dict[str, Any]) -> RoboEnv:
    from robodeploy.builtins import import_builtins
    from robodeploy.core.registry import get_task, use
    from robodeploy.core.robot import Robot, RobotTask
    from robodeploy.testing import DummyBackend, DummyRobot

    import_builtins()
    for module_path in cfg.get("custom_modules") or []:
        use(str(module_path))

    from robodeploy.evaluation.policy_loader import coerce_eval_policy

    task_cls = get_task(str(cfg["task"]))
    task_obj = task_cls(config=dict(cfg.get("task_kwargs") or {}))
    policy_obj = coerce_eval_policy(str(cfg["policy"]), dict(cfg.get("policy_kwargs") or {}))

    robot = Robot(
        robot_id=str(cfg.get("robot_id", "robot0")),
        description=DummyRobot(),
        tasks={
            str(cfg.get("task_id", "task0")): RobotTask(
                task=task_obj,
                policies={str(cfg.get("policy_id", "policy0")): policy_obj},
                task_id=str(cfg.get("task_id", "task0")),
            ),
        },
    )
    return RoboEnv(
        backend=DummyBackend(cfg.get("backend_kwargs")),
        robots=[robot],
        max_episode_steps=cfg.get("max_episode_steps"),
        obs_spec_policy=str(cfg.get("obs_spec_policy", "warn")),
    )


def make_env_factory(preset: dict[str, Any]) -> Callable[[int], RoboEnv]:
    def factory(seed: int) -> RoboEnv:
        return build_env_from_preset(preset, seed=seed)

    return factory
