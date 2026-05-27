"""Minimal vectorized env helper (sequential stepping, honest contract)."""

from __future__ import annotations

from typing import Any

from robodeploy.core.types import Action, EpisodeInfo, Observation


class SequentialVecEnv:
    """Step multiple RoboEnv instances and return batched observations."""

    def __init__(self, envs: list[Any]) -> None:
        if not envs:
            raise ValueError("SequentialVecEnv requires at least one environment.")
        self._envs = list(envs)

    @classmethod
    def from_presets(cls, preset_names: list[str], **overrides) -> "SequentialVecEnv":
        """Build one RoboEnv per preset name and wrap in a SequentialVecEnv."""
        from robodeploy.env import RoboEnv

        envs = [
            RoboEnv.from_preset(name, robot_id=f"robot{i}", **overrides)
            for i, name in enumerate(preset_names)
        ]
        return cls(envs)

    @property
    def num_envs(self) -> int:
        return len(self._envs)

    def reset(self) -> tuple[list[Observation], list[EpisodeInfo]]:
        obs_list: list[Observation] = []
        info_list: list[EpisodeInfo] = []
        for env in self._envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def step(
        self,
        actions: list[Action | None],
    ) -> tuple[list[Observation], list[float], list[bool], list[EpisodeInfo]]:
        if len(actions) != len(self._envs):
            raise ValueError(f"Expected {len(self._envs)} actions, got {len(actions)}.")
        obs_list: list[Observation] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos: list[EpisodeInfo] = []
        for env, action in zip(self._envs, actions):
            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
        return obs_list, rewards, dones, infos
