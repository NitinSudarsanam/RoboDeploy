"""Behavior cloning policy module and training helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from robodeploy.training.dataset import DemoDataset
from robodeploy.training.trainer import Trainer, TrainerConfig


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "BC training requires PyTorch. Install with: pip install 'robodeploy[training]'"
        ) from exc
    return torch, nn, F


class BCPolicyModule:
    """MLP policy mapping observation dicts to action tensors."""

    def __init__(
        self,
        obs_keys: list[str],
        action_dim: int,
        *,
        hidden: Sequence[int] = (256, 256),
        encoder: str = "mlp",
        proprio_dim: int = 6,
        rgb_shape: tuple[int, int, int] | None = None,
    ) -> None:
        torch, nn, _ = _require_torch()
        self.obs_keys = list(obs_keys)
        self.action_dim = int(action_dim)
        in_dim = proprio_dim
        self._module = nn.Module()
        if "rgb" in self.obs_keys and encoder == "cnn":
            c, h, w = rgb_shape or (3, 64, 64)
            self._module.rgb_encoder = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                flat = self._module.rgb_encoder(torch.zeros(1, c, h, w)).shape[-1]
            in_dim = proprio_dim + flat
        layers: list[Any] = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        layers.append(nn.Linear(prev, self.action_dim))
        self._module.head = nn.Sequential(*layers)

    def forward(self, obs_dict):
        torch, _, _ = _require_torch()
        x = obs_dict["proprio"]
        if hasattr(self._module, "rgb_encoder") and "rgb" in obs_dict:
            x = torch.cat([x, self._module.rgb_encoder(obs_dict["rgb"])], dim=-1)
        return self._module.head(x)

    def to(self, device: str):
        self._module.to(device)
        return self

    def train(self, mode: bool = True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self

    def parameters(self):
        return self._module.parameters()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state_dict):
        return self._module.load_state_dict(state_dict)

    def __call__(self, obs_dict):
        return self.forward(obs_dict)


def bc_mse_loss(pred, target, mask=None):
    _, _, F = _require_torch()
    if mask is not None:
        return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
    return F.mse_loss(pred, target)


def bc_gaussian_nll_loss(mu, log_std, target):
    _, _, F = _require_torch()
    var = (log_std * 2).exp()
    return F.gaussian_nll_loss(mu, target, var)


def train_bc(
    *,
    dataset: DemoDataset,
    obs_keys: list[str] | None = None,
    action_dim: int | None = None,
    config: TrainerConfig | None = None,
    eval_env: Any | None = None,
    callbacks: Sequence[Any] = (),
) -> BCPolicyModule:
    """Convenience BC training entry point."""
    keys = obs_keys or ["proprio"]
    dim = action_dim or dataset.action_dim
    module = BCPolicyModule(
        obs_keys=keys,
        action_dim=dim,
        proprio_dim=dataset.proprio_dim,
    )
    trainer = Trainer(
        policy_module=module,
        dataset=dataset,
        loss_fn=bc_mse_loss,
        config=config,
        eval_env=eval_env,
        callbacks=callbacks,
    )
    trainer.fit()
    out = Path((config or TrainerConfig()).log_dir) / "bc_final.pt"
    trainer.save_checkpoint(str(out))
    return module
