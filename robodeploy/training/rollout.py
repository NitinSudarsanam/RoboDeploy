"""Episode rollout collection for RoboEnv."""

from __future__ import annotations

from typing import Callable, Optional

from robodeploy.core.types import Action, Observation
from robodeploy.demo_recording import DemoFrame, DemoRecorder
from robodeploy.env import RoboEnv


class RolloutCollector:
    """Collects trajectories from RoboEnv with explicit or policy-driven actions."""

    def __init__(self, env: RoboEnv, *, max_steps: int | None = None) -> None:
        self._env = env
        self._max_steps = int(max_steps or env._max_steps)

    def collect_episode(
        self,
        action_fn: Optional[Callable[[Observation], Action | None]] = None,
    ) -> list[DemoFrame]:
        obs, _ = self._env.reset()
        frames: list[DemoFrame] = []
        for _ in range(self._max_steps):
            action = action_fn(obs) if action_fn is not None else None
            next_obs, reward, done, info = self._env.step(action)
            if action is not None:
                from robodeploy.demo_recording import _to_jsonable

                frames.append(
                    DemoFrame(
                        observation=_to_jsonable(obs),
                        action=_to_jsonable(action),
                        reward=float(reward),
                        done=bool(done),
                    )
                )
            obs = next_obs
            if done:
                break
        return frames

    def collect_n(
        self,
        n_episodes: int,
        action_fn: Optional[Callable[[Observation], Action | None]] = None,
    ) -> DemoRecorder:
        recorder = DemoRecorder()
        for _ in range(int(n_episodes)):
            recorder.frames.extend(self.collect_episode(action_fn))
        return recorder
