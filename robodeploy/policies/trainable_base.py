"""Trainable policy wrapper bridging torch modules to IPolicy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from robodeploy.core.interop import to_numpy, to_torch
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase
from robodeploy.training.gym_adapter import observation_to_dict


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "TrainablePolicyBase requires PyTorch. Install with: pip install 'robodeploy[training]'"
        ) from exc
    return torch


class TrainablePolicyBase(PolicyBase):
    """PolicyBase subclass wrapping a torch.nn.Module with train/eval modes."""

    def __init__(
        self,
        *,
        module: Any,
        action_space: ActionSpace,
        config: dict | None = None,
        obs_keys: list[str] | None = None,
        obs_to_dict: Callable[[Observation], dict[str, np.ndarray]] | None = None,
        tensor_to_action: Callable[[Any], Action] | None = None,
    ) -> None:
        super().__init__(action_space=action_space, config=config or {})
        self._module = module
        torch = _require_torch()
        params = list(module.parameters()) if hasattr(module, "parameters") else []
        self._device = params[0].device if params else torch.device("cpu")
        self._obs_keys = obs_keys or ["proprio"]
        self._obs_to_dict = obs_to_dict
        self._tensor_to_action = tensor_to_action

    def _obs_to_dict_impl(self, obs: Observation) -> dict[str, np.ndarray]:
        if self._obs_to_dict is not None:
            return self._obs_to_dict(obs)
        from robodeploy.core.types import ObsSpec

        return observation_to_dict(obs, ObsSpec())

    def _tensor_to_action_impl(self, tensor: Any) -> Action:
        if self._tensor_to_action is not None:
            return self._tensor_to_action(tensor)
        arr = to_numpy(tensor).reshape(-1)
        try:
            import jax.numpy as jnp
        except ImportError:
            import numpy as jnp  # type: ignore[assignment]
        vec = jnp.asarray(arr, dtype=jnp.float32)
        if self.action_space == ActionSpace.JOINT_POS:
            return Action(joint_positions=vec, action_space=self.action_space)
        if self.action_space == ActionSpace.JOINT_VEL:
            return Action(joint_velocities=vec, action_space=self.action_space)
        if self.action_space == ActionSpace.JOINT_TORQUE:
            return Action(joint_torques=vec, action_space=self.action_space)
        return Action(joint_positions=vec, action_space=ActionSpace.JOINT_POS)

    def get_action(self, obs: Observation) -> Action:
        torch = _require_torch()
        self._module.eval()
        obs_dict = self._obs_to_dict_impl(obs)
        obs_t = {
            key: to_torch(obs_dict[key]).unsqueeze(0).to(self._device)
            for key in self._obs_keys
            if key in obs_dict
        }
        with torch.no_grad():
            action_tensor = self._module(obs_t).squeeze(0)
        return self._tensor_to_action_impl(action_tensor)

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        torch = _require_torch()
        self._module.eval()
        if not obs_batch:
            return []
        obs_dicts = [self._obs_to_dict_impl(obs) for obs in obs_batch]
        obs_t = {
            key: torch.stack(
                [to_torch(d[key]) for d in obs_dicts if key in d],
                dim=0,
            ).to(self._device)
            for key in self._obs_keys
        }
        with torch.no_grad():
            action_tensors = self._module(obs_t)
        return [self._tensor_to_action_impl(action_tensors[i]) for i in range(len(obs_batch))]

    def train_mode(self) -> None:
        self._module.train()

    def eval_mode(self) -> None:
        self._module.eval()

    def state_dict(self) -> dict:
        return self._module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._module.load_state_dict(state_dict)

    def save_checkpoint(self, path: str | Path) -> None:
        torch = _require_torch()
        payload = {
            "policy": self.state_dict(),
            "action_space": self.action_space.name,
            "config": self.config,
        }
        torch.save(payload, str(path))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        module: Any,
        action_space: ActionSpace,
        config: dict | None = None,
    ) -> "TrainablePolicyBase":
        torch = _require_torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        policy = cls(module=module, action_space=action_space, config=config or payload.get("config"))
        policy.load_state_dict(payload["policy"])
        policy.eval_mode()
        return policy
