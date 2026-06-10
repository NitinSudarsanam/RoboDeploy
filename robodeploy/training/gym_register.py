"""Gymnasium environment registration for RoboDeploy."""

from __future__ import annotations

from typing import Any

_REGISTERED = False


def make_robodeploy_tiny(
    *,
    max_episode_steps: int = 50,
    render_mode: str | None = None,
    **kwargs: Any,
):
    """Minimal Dict-obs gym env for fast SB3 smoke tests (no RoboEnv)."""
    del kwargs, render_mode
    from robodeploy.training.parallel_vec_env import TinyDictGymEnv

    return TinyDictGymEnv(max_steps=max_episode_steps)


def make_robodeploy_dummy(
    *,
    max_episode_steps: int = 100,
    render_mode: str | None = None,
    **kwargs: Any,
):
    """Factory for ``gym.make('robodeploy/Dummy-v0')``."""
    del kwargs
    from robodeploy.cli import _make_dummy_env
    from robodeploy.training.gym_adapter import GymRoboEnv

    return GymRoboEnv(
        _make_dummy_env(),
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )


def make_kuka_pick_mujoco(
    *,
    max_episode_steps: int = 1000,
    render_mode: str | None = None,
    **kwargs: Any,
):
    """Factory for ``gym.make('robodeploy/kuka_pick_mujoco-v0')``."""
    del kwargs
    from robodeploy.training.gym_adapter import GymRoboEnv

    try:
        from examples.env_from_preset import env_from_preset
    except ImportError as exc:
        raise ImportError(
            "kuka_pick_mujoco gym env requires the RoboDeploy examples package on PYTHONPATH."
        ) from exc
    robo = env_from_preset("kuka_pick_mujoco", max_episode_steps=max_episode_steps)
    return GymRoboEnv(robo, max_episode_steps=max_episode_steps, render_mode=render_mode)


def robodeploy_dummy_gym_env_factory(max_episode_steps: int = 100):
    """Picklable factory for SubprocVecEnv (Windows spawn-safe)."""
    return make_robodeploy_dummy(max_episode_steps=max_episode_steps)


def _reach_target_preset_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "benchmarks/manipulation_v1/reach_target/preset_dummy.yaml"


def make_reach_target_dummy(
    *,
    max_episode_steps: int = 300,
    seed: int = 0,
    render_mode: str | None = None,
    **kwargs: Any,
):
    """Factory for reach_target tier-1 benchmark on the dummy backend."""
    del kwargs
    import yaml

    from robodeploy.evaluation.env_builder import build_env_from_preset
    from robodeploy.training.gym_adapter import GymRoboEnv

    preset = yaml.safe_load(_reach_target_preset_path().read_text(encoding="utf-8"))
    robo = build_env_from_preset(preset, seed=int(seed))
    return GymRoboEnv(robo, max_episode_steps=max_episode_steps, render_mode=render_mode)


def reach_target_dummy_gym_env_factory(max_episode_steps: int = 300):
    """Picklable factory for PPO reach_target integration tests."""
    return make_reach_target_dummy(max_episode_steps=max_episode_steps)


def register_robodeploy_envs() -> None:
    """Register RoboDeploy envs with gymnasium (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        import gymnasium as gym
    except ImportError:
        return
    try:
        gym.register(
            id="robodeploy/Dummy-v0",
            entry_point="robodeploy.training.gym_register:make_robodeploy_dummy",
            max_episode_steps=100,
            force=True,
        )
        gym.register(
            id="robodeploy/Tiny-v0",
            entry_point="robodeploy.training.gym_register:make_robodeploy_tiny",
            max_episode_steps=50,
            force=True,
        )
        gym.register(
            id="robodeploy/kuka_pick_mujoco-v0",
            entry_point="robodeploy.training.gym_register:make_kuka_pick_mujoco",
            max_episode_steps=1000,
            force=True,
        )
    except TypeError:
        gym.register(
            id="robodeploy/Dummy-v0",
            entry_point="robodeploy.training.gym_register:make_robodeploy_dummy",
            max_episode_steps=100,
        )
        gym.register(
            id="robodeploy/Tiny-v0",
            entry_point="robodeploy.training.gym_register:make_robodeploy_tiny",
            max_episode_steps=50,
        )
        gym.register(
            id="robodeploy/kuka_pick_mujoco-v0",
            entry_point="robodeploy.training.gym_register:make_kuka_pick_mujoco",
            max_episode_steps=1000,
        )
    _REGISTERED = True


register_robodeploy_envs()
