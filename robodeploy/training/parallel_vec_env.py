"""Subprocess and async vectorized environments for parallel rollouts."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from typing import Any, Callable, Sequence

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]


_CMD_STEP = "step"
_CMD_RESET = "reset"
_CMD_CLOSE = "close"
_CMD_GET_ATTR = "get_attr"


def _worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn: Callable[[], Any]) -> None:
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == _CMD_STEP:
                remote.send(env.step(data))
            elif cmd == _CMD_RESET:
                remote.send(env.reset(seed=data))
            elif cmd == _CMD_CLOSE:
                if hasattr(env, "close"):
                    env.close()
                remote.close()
                break
            elif cmd == _CMD_GET_ATTR:
                remote.send(getattr(env, data))
            else:
                raise RuntimeError(f"Unknown vec env command: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass


class SubprocVecEnv:
    """Each env in a subprocess. Mirrors Stable-Baselines3 SubprocVecEnv."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        *,
        start_method: str = "spawn",
    ) -> None:
        if not env_fns:
            raise ValueError("SubprocVecEnv requires at least one env factory.")
        self._closed = False
        self._num_envs = len(env_fns)
        ctx = mp.get_context(start_method)
        self._remotes, self._work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in env_fns])
        self._processes = [
            ctx.Process(
                target=_worker,
                args=(work_remote, remote, env_fn),
                daemon=True,
            )
            for work_remote, remote, env_fn in zip(self._work_remotes, self._remotes, env_fns)
        ]
        for process in self._processes:
            process.start()
        for work_remote in self._work_remotes:
            work_remote.close()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self, seeds: list[int] | None = None) -> tuple[list[Any], list[dict]]:
        if seeds is None:
            seeds = [None] * self._num_envs
        if len(seeds) != self._num_envs:
            raise ValueError(f"Expected {self._num_envs} seeds, got {len(seeds)}.")
        for remote, seed in zip(self._remotes, seeds):
            remote.send((_CMD_RESET, seed))
        results = [remote.recv() for remote in self._remotes]
        obs = [item[0] for item in results]
        infos = [item[1] for item in results]
        return obs, infos

    def step(self, actions: np.ndarray | list[Any]) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        if isinstance(actions, np.ndarray):
            if len(actions) != self._num_envs:
                raise ValueError(f"Expected {self._num_envs} actions, got {len(actions)}.")
            action_list = [actions[i] for i in range(self._num_envs)]
        else:
            action_list = list(actions)
            if len(action_list) != self._num_envs:
                raise ValueError(f"Expected {self._num_envs} actions, got {len(action_list)}.")
        for remote, action in zip(self._remotes, action_list):
            remote.send((_CMD_STEP, action))
        results = [remote.recv() for remote in self._remotes]
        obs = [item[0] for item in results]
        rewards = np.asarray([float(item[1]) for item in results], dtype=np.float32)
        terminated = np.asarray([bool(item[2]) for item in results], dtype=bool)
        truncated = np.asarray([bool(item[3]) for item in results], dtype=bool)
        infos = [item[4] for item in results]
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self._closed:
            return
        for remote in self._remotes:
            try:
                remote.send((_CMD_CLOSE, None))
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            if process._popen is None:
                continue
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
        self._closed = True

    def __del__(self) -> None:
        if not self._closed:
            self.close()


class AsyncVecEnv:
    """Asyncio-driven vec env for I/O-bound real backends (ROS2, network sim)."""

    def __init__(self, env_fns: Sequence[Callable[[], Any]]) -> None:
        if not env_fns:
            raise ValueError("AsyncVecEnv requires at least one env factory.")
        self._env_fns = list(env_fns)
        self._envs: list[Any] = []
        self._closed = False
        self._num_envs = len(env_fns)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    async def _ensure_envs(self) -> None:
        if not self._envs:
            self._envs = [fn() for fn in self._env_fns]

    async def reset(self, seeds: list[int] | None = None):
        await self._ensure_envs()
        if seeds is None:
            seeds = [None] * self._num_envs

        async def _reset_one(env, seed):
            return await asyncio.to_thread(env.reset, seed=seed)

        results = await asyncio.gather(
            *[_reset_one(env, seed) for env, seed in zip(self._envs, seeds)]
        )
        obs = [item[0] for item in results]
        infos = [item[1] for item in results]
        return obs, infos

    async def step(self, actions: np.ndarray | list[Any]):
        await self._ensure_envs()
        if isinstance(actions, np.ndarray):
            action_list = [actions[i] for i in range(self._num_envs)]
        else:
            action_list = list(actions)

        async def _step_one(env, action):
            return await asyncio.to_thread(env.step, action)

        results = await asyncio.gather(
            *[_step_one(env, action) for env, action in zip(self._envs, action_list)]
        )
        obs = [item[0] for item in results]
        rewards = np.asarray([float(item[1]) for item in results], dtype=np.float32)
        terminated = np.asarray([bool(item[2]) for item in results], dtype=bool)
        truncated = np.asarray([bool(item[3]) for item in results], dtype=bool)
        infos = [item[4] for item in results]
        return obs, rewards, terminated, truncated, infos

    def reset_sync(self, seeds: list[int] | None = None):
        return asyncio.run(self.reset(seeds))

    def step_sync(self, actions: np.ndarray | list[Any]):
        return asyncio.run(self.step(actions))

    def close(self) -> None:
        if self._closed:
            return
        for env in self._envs:
            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
        self._envs.clear()
        self._closed = True


def dummy_gym_env_factory(tag: int = 0, max_steps: int = 5, work_iters: int = 0):
    """Picklable env factory for SubprocVecEnv tests."""
    return DummyGymEnv(tag=tag, max_steps=max_steps, work_iters=work_iters)


class DummyGymEnv:
    """Lightweight gym env for vec-env smoke tests without RoboEnv."""

    def __init__(self, *, tag: int = 0, max_steps: int = 5, work_iters: int = 0) -> None:
        self.tag = int(tag)
        self._max_steps = int(max_steps)
        self._work_iters = int(work_iters)
        self._step_count = 0
        if gym is not None:
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        self._step_count = 0
        obs = np.array([float(self.tag), 0.0], dtype=np.float32)
        return obs, {"tag": self.tag}

    def step(self, action: np.ndarray):
        self._step_count += 1
        if self._work_iters > 0:
            x = np.ones(32, dtype=np.float64)
            for _ in range(self._work_iters):
                x = np.sin(x) + np.cos(x)
            _ = float(x[0])
        obs = np.array([float(self.tag), float(action[0])], dtype=np.float32)
        reward = float(self._step_count)
        terminated = self._step_count >= self._max_steps
        truncated = False
        return obs, reward, terminated, truncated, {"tag": self.tag}

    def close(self) -> None:
        return
