"""Proximal Policy Optimization with GAE for RoboDeploy gym environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from robodeploy.training.replay_buffer import RolloutBuffer


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.distributions import Normal
    except ImportError as exc:
        raise ImportError(
            "PPO training requires PyTorch. Install with: pip install 'robodeploy[training]'"
        ) from exc
    return torch, nn, F, Normal


def _stack_obs(obs_list: list[dict[str, np.ndarray]], key: str = "proprio"):
    return np.stack([np.asarray(o[key], dtype=np.float32) for o in obs_list], axis=0)


@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    n_envs: int = 8
    n_epochs: int = 10
    minibatch_size: int = 256
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    target_kl: float | None = 0.03
    lr: float = 3e-4
    total_steps: int = 1_000_000
    device: str = "auto"
    log_dir: str = "./runs/ppo"
    seed: int = 42

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        torch, _, _, _ = _require_torch()
        return "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic:
    """Gaussian actor + value critic over proprio observations."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        torch, nn, _, Normal = _require_torch()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._module = nn.Module()
        layers: list[Any] = []
        prev = obs_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.Tanh()])
            prev = width
        self._module.shared = nn.Sequential(*layers)
        self._module.policy_mean = nn.Linear(prev, action_dim)
        self._module.value = nn.Linear(prev, 1)
        self._log_std = nn.Parameter(torch.zeros(action_dim))
        self._Normal = Normal

    def forward(self, obs_tensor):
        torch, _, _, Normal = _require_torch()
        shared = self._module.shared(obs_tensor)
        mean = self._module.policy_mean(shared)
        std = self._log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self._module.value(shared).squeeze(-1)
        return dist, value

    def evaluate_actions(self, obs_tensor, actions):
        dist, value = self.forward(obs_tensor)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def sample_action(self, obs_tensor):
        torch, _, _, _ = _require_torch()
        dist, value = self.forward(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def to(self, device: str):
        self._module.to(device)
        self._log_std.data = self._log_std.data.to(device)
        return self

    def parameters(self):
        return list(self._module.parameters()) + [self._log_std]

    def train(self, mode: bool = True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self

    def state_dict(self):
        torch, _, _, _ = _require_torch()
        return {
            "module": self._module.state_dict(),
            "log_std": self._log_std.detach().cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._module.load_state_dict(state_dict["module"])
        self._log_std.data = state_dict["log_std"]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (Schulman et al.)."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_clip_loss(
    log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
    *,
    clip_range: float,
):
    torch, _, _, _ = _require_torch()
    ratio = (log_probs - old_log_probs).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
    return -torch.min(unclipped, clipped).mean()


class PPOTrainer:
    """On-policy PPO trainer using SubprocVecEnv or any vec env with reset/step."""

    def __init__(
        self,
        *,
        env: Any,
        model: ActorCritic,
        config: PPOConfig | None = None,
        callbacks: Iterable[Any] = (),
        obs_key: str = "proprio",
    ) -> None:
        torch, _, _, _ = _require_torch()
        self.env = env
        self.model = model
        self.config = config or PPOConfig()
        self.device = self.config.resolve_device()
        self.model.to(self.device)
        self.callbacks = list(callbacks)
        self.obs_key = obs_key
        self.global_step = 0
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def collect_rollouts(self) -> RolloutBuffer:
        steps_per_env = max(1, self.config.rollout_steps // self.env.num_envs)
        buffer = RolloutBuffer(capacity=steps_per_env * self.env.num_envs)
        obs_list, _ = self.env.reset()
        for _ in range(steps_per_env):
            obs_batch = _stack_obs(obs_list, self.obs_key)
            torch, _, _, _ = _require_torch()
            obs_t = torch.from_numpy(obs_batch).to(self.device)
            with torch.no_grad():
                actions_t, log_probs_t, values_t = self.model.sample_action(obs_t)
            actions_np = actions_t.cpu().numpy()
            next_obs_list, rewards, terminated, truncated, infos = self.env.step(actions_np)
            for i in range(self.env.num_envs):
                done = bool(terminated[i] or truncated[i])
                buffer.add(
                    obs={self.obs_key: obs_batch[i]},
                    action=actions_np[i],
                    reward=float(rewards[i]),
                    value=float(values_t[i].cpu().item()),
                    log_prob=float(log_probs_t[i].cpu().item()),
                    done=done,
                )
                self.global_step += 1
            obs_list = next_obs_list
            if buffer.size >= buffer.capacity:
                break
        self._fill_gae(buffer)
        return buffer

    def _fill_gae(self, buffer: RolloutBuffer) -> None:
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        values = np.asarray(buffer.values, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        buffer.advantages = advantages.tolist()
        buffer.returns = returns.tolist()

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        torch, _, F, _ = _require_torch()
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for _ in range(self.config.n_epochs):
            for batch in buffer.get(self.config.minibatch_size):
                obs = batch["obs"][self.obs_key].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                log_probs, entropy, values = self.model.evaluate_actions(obs, actions)
                policy_loss = ppo_clip_loss(
                    log_probs,
                    old_log_probs,
                    advantages,
                    clip_range=self.config.clip_range,
                )
                value_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )
                approx_kl = float((old_log_probs - log_probs).mean().detach().cpu().item())
                if self.config.target_kl is not None and approx_kl > 1.5 * self.config.target_kl:
                    break
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                metrics["policy_loss"] += float(policy_loss.detach().cpu().item())
                metrics["value_loss"] += float(value_loss.detach().cpu().item())
                metrics["entropy"] += float(entropy.mean().detach().cpu().item())
                metrics["approx_kl"] += approx_kl
                n_updates += 1
        if n_updates:
            for key in metrics:
                metrics[key] /= n_updates
        return metrics

    def fit(self) -> dict[str, float]:
        from pathlib import Path

        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        last_metrics: dict[str, float] = {}
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin(self)
        while self.global_step < self.config.total_steps:
            buffer = self.collect_rollouts()
            metrics = self.update(buffer)
            last_metrics = metrics
            for callback in self.callbacks:
                if hasattr(callback, "on_step_end"):
                    callback.on_step_end(self, metrics)
        return last_metrics


def train_ppo(
    *,
    env_factory: Callable[[], Any],
    obs_dim: int,
    action_dim: int,
    config: PPOConfig | None = None,
    callbacks: Iterable[Any] = (),
    vec_env_cls: Any | None = None,
) -> ActorCritic:
    """Convenience entry: build SubprocVecEnv, train PPO, return ActorCritic."""
    from robodeploy.training.parallel_vec_env import SubprocVecEnv

    cfg = config or PPOConfig()
    cls = vec_env_cls or SubprocVecEnv
    vec = cls([env_factory for _ in range(cfg.n_envs)])
    try:
        model = ActorCritic(obs_dim, action_dim)
        trainer = PPOTrainer(env=vec, model=model, config=cfg, callbacks=callbacks)
        trainer.fit()
        return model
    finally:
        vec.close()


def evaluate_actor_critic(
    model: ActorCritic,
    env_factory: Callable[[], Any],
    *,
    n_episodes: int = 20,
    obs_key: str = "proprio",
    deterministic: bool = True,
) -> dict[str, float]:
    """Roll out a trained PPO policy and report success rate + mean reward."""
    torch, _, _, _ = _require_torch()
    model.eval()
    successes = 0
    rewards: list[float] = []
    for episode in range(int(n_episodes)):
        env = env_factory()
        try:
            obs, _ = env.reset(seed=episode)
            done = False
            ep_reward = 0.0
            while not done:
                obs_vec = np.asarray(obs[obs_key], dtype=np.float32)
                obs_t = torch.from_numpy(obs_vec).float().unsqueeze(0)
                with torch.no_grad():
                    dist, _ = model.forward(obs_t)
                    action = dist.mean if deterministic else dist.sample()
                    action_np = action.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action_np)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
            rewards.append(ep_reward)
            if info.get("success"):
                successes += 1
        finally:
            if hasattr(env, "close"):
                env.close()
    return {
        "eval/success_rate": float(successes / max(n_episodes, 1)),
        "eval/mean_reward": float(sum(rewards) / max(len(rewards), 1)),
    }
