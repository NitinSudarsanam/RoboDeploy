"""Policy resolution for benchmark evaluation (scripted, learned, BC checkpoints)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.registry import get_policy
from robodeploy.core.spaces import ActionSpace


def is_checkpoint_path(ref: str) -> bool:
    return str(ref).lower().endswith((".pt", ".pth", ".ckpt"))


def load_bc_checkpoint_policy(path: str | Path, *, config: dict[str, Any] | None = None) -> IPolicy:
    from robodeploy.policies.trainable_base import TrainablePolicyBase
    from robodeploy.training.bc import BCPolicyModule

    cfg = dict(config or {})
    obs_keys = list(cfg.get("obs_keys", ["proprio"]))
    action_dim = int(cfg.get("action_dim", 2))
    proprio_dim = int(cfg.get("proprio_dim", 6))
    module = BCPolicyModule(
        obs_keys=obs_keys,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
    )
    return TrainablePolicyBase.from_checkpoint(
        path,
        module=module,
        action_space=ActionSpace.JOINT_POS,
        config=cfg,
    )


def coerce_eval_policy(policy_ref: str, policy_kwargs: dict[str, Any] | None = None) -> IPolicy:
    ref = str(policy_ref).strip()
    kwargs = dict(policy_kwargs or {})
    if is_checkpoint_path(ref):
        try:
            import torch

            payload = torch.load(ref, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "policy" in payload:
                return load_bc_checkpoint_policy(ref, config=kwargs)
        except Exception:
            pass
        from robodeploy.policies.learned.factory import load_policy_from_ref

        return load_policy_from_ref(ref, config=kwargs)
    if ref.startswith("hf:") or (":" in ref and not ref.startswith("http")):
        from robodeploy.policies.learned.factory import load_policy_from_ref

        return load_policy_from_ref(ref, config=kwargs)
    PolicyClass = get_policy(ref)
    return PolicyClass(config=kwargs)
