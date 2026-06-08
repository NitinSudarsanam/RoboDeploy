"""Generic supervised trainer for behavior cloning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from robodeploy.training.dataset import DemoCollator, DemoDataset


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Trainer requires PyTorch. Install with: pip install 'robodeploy[training]'"
        ) from exc
    return torch, nn


@dataclass
class TrainerConfig:
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    grad_clip: float = 1.0
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    device: str = "auto"
    log_dir: str = "./runs"
    seed: int = 42

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        torch, _ = _require_torch()
        return "cuda" if torch.cuda.is_available() else "cpu"


class TrainerCallback:
    def on_train_begin(self, trainer: "Trainer") -> None:
        return

    def on_step_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        return

    def on_epoch_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        return

    def on_eval_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        return

    def on_checkpoint(self, trainer: "Trainer", path: str) -> None:
        return


class Trainer:
    def __init__(
        self,
        *,
        policy_module: Any,
        dataset: DemoDataset,
        loss_fn: Callable,
        optimizer_fn: Optional[Callable] = None,
        config: TrainerConfig | None = None,
        eval_env: Any | None = None,
        callbacks: Iterable[TrainerCallback] = (),
    ) -> None:
        torch, _ = _require_torch()
        from torch.utils.data import DataLoader

        self.config = config or TrainerConfig()
        self.device = self.config.resolve_device()
        self.policy = policy_module.to(self.device)
        self.loss_fn = loss_fn
        self.eval_env = eval_env
        self.callbacks = list(callbacks)
        self.global_step = 0
        self.optim = (
            optimizer_fn(self.policy.parameters())
            if optimizer_fn is not None
            else torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)
        )
        self.loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=DemoCollator(),
        )
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def fit(self) -> dict[str, float]:
        torch, _ = _require_torch()
        torch.manual_seed(self.config.seed)
        last_metrics: dict[str, float] = {}
        for callback in self.callbacks:
            callback.on_train_begin(self)
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in self.loader:
                metrics = self.step(batch)
                epoch_loss += metrics.get("loss", 0.0)
                n_batches += 1
                self.global_step += 1
                for callback in self.callbacks:
                    callback.on_step_end(self, metrics)
                if self.eval_env is not None and self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    for callback in self.callbacks:
                        callback.on_eval_end(self, eval_metrics)
                if self.global_step % self.config.checkpoint_interval == 0:
                    ckpt = str(Path(self.config.log_dir) / f"step_{self.global_step}.pt")
                    self.save_checkpoint(ckpt)
                    for callback in self.callbacks:
                        callback.on_checkpoint(self, ckpt)
            last_metrics = {
                "epoch": float(epoch),
                "loss": epoch_loss / max(n_batches, 1),
            }
            for callback in self.callbacks:
                callback.on_epoch_end(self, last_metrics)
        return last_metrics

    def step(self, batch: dict[str, Any]) -> dict[str, float]:
        torch, _ = _require_torch()
        self.policy.train()
        obs = {k: v.to(self.device) for k, v in batch["obs"].items()}
        target = batch["action"].to(self.device)
        pred = self.policy(obs)
        loss = self.loss_fn(pred, target)
        self.optim.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optim.step()
        return {"loss": float(loss.detach().cpu().item())}

    def evaluate(self, n_episodes: int = 5) -> dict[str, float]:
        if self.eval_env is None:
            return {}
        self.policy.eval()
        successes = 0
        rewards: list[float] = []
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                torch, _ = _require_torch()
                with torch.no_grad():
                    obs_t = {
                        k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)
                        for k, v in obs.items()
                    }
                    action = self.policy(obs_t).squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
            rewards.append(ep_reward)
            if info.get("success"):
                successes += 1
        return {
            "eval/mean_reward": float(sum(rewards) / max(len(rewards), 1)),
            "eval/success_rate": float(successes / max(n_episodes, 1)),
        }

    def save_checkpoint(self, path: str) -> None:
        torch, _ = _require_torch()
        payload = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optim.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        torch, _ = _require_torch()
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(payload["policy"])
        if "optimizer" in payload:
            self.optim.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
