"""Replay and rollout buffers for RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Replay buffers require PyTorch. Install with: pip install 'robodeploy[training]'"
        ) from exc
    return torch


@dataclass
class RolloutBuffer:
    """On-policy rollout storage for PPO-style updates."""

    capacity: int
    obs: dict[str, list] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    _size: int = 0

    def add(
        self,
        *,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        value: float = 0.0,
        log_prob: float = 0.0,
        done: bool,
    ) -> None:
        if self._size >= self.capacity:
            raise RuntimeError("RolloutBuffer is full; call clear() before reusing.")
        for key, arr in obs.items():
            self.obs.setdefault(key, []).append(np.asarray(arr, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))
        self._size += 1

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
        self._size = 0

    def get(self, minibatch_size: int) -> Iterator[dict[str, Any]]:
        torch = _require_torch()
        if self._size == 0:
            return
        indices = np.arange(self._size)
        np.random.shuffle(indices)
        for start in range(0, self._size, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batch_obs = {
                key: torch.from_numpy(np.stack([self.obs[key][i] for i in batch_idx], axis=0))
                for key in self.obs
            }
            yield {
                "obs": batch_obs,
                "actions": torch.from_numpy(np.stack([self.actions[i] for i in batch_idx], axis=0)),
                "rewards": torch.tensor([self.rewards[i] for i in batch_idx], dtype=torch.float32),
                "values": torch.tensor([self.values[i] for i in batch_idx], dtype=torch.float32),
                "log_probs": torch.tensor([self.log_probs[i] for i in batch_idx], dtype=torch.float32),
                "advantages": torch.tensor([self.advantages[i] for i in batch_idx], dtype=torch.float32),
                "returns": torch.tensor([self.returns[i] for i in batch_idx], dtype=torch.float32),
                "dones": torch.tensor([self.dones[i] for i in batch_idx], dtype=torch.bool),
            }


class ReplayBuffer:
    """Uniform replay buffer for off-policy RL (SAC/TD3 extension)."""

    def __init__(
        self,
        capacity: int,
        *,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = str(device)
        self._pos = 0
        self._size = 0
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._pos
        self._obs[idx] = np.asarray(obs, dtype=np.float32).reshape(-1)[: self.obs_dim]
        self._actions[idx] = np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim]
        self._rewards[idx] = float(reward)
        self._next_obs[idx] = np.asarray(next_obs, dtype=np.float32).reshape(-1)[: self.obs_dim]
        self._dones[idx] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        torch = _require_torch()
        if self._size < batch_size:
            raise ValueError(f"ReplayBuffer has {self._size} samples, need {batch_size}.")
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self._obs[idx]).to(self.device),
            torch.from_numpy(self._actions[idx]).to(self.device),
            torch.from_numpy(self._rewards[idx]).to(self.device),
            torch.from_numpy(self._next_obs[idx]).to(self.device),
            torch.from_numpy(self._dones[idx]).to(self.device),
        )
