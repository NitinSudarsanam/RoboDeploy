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
    """Load ``examples/config/presets.yaml`` and construct via ``RoboEnv.from_config``."""
    cfg = {**load_example_preset(name), **overrides}
    return RoboEnv.from_config(cfg)


def example_presets_file() -> str:
    return str(PRESETS_FILE)
