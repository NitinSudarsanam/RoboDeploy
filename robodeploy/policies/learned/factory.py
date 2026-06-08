"""Policy factory — resolve CLI / config refs into IPolicy instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robodeploy.core.interfaces.policy import IPolicy
from robodeploy.core.registry import get_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.policies.learned.hf_hub import HFModelRegistry
from robodeploy.policies.learned.loader import ModelLoader, ModelSpec, _action_space_from_value


def parse_policy_ref(ref: str) -> tuple[str, dict[str, Any]]:
    """Parse refs like ``hf:openvla-7b``, ``robomimic:ckpt.pt``, ``vla_stub``."""
    raw = str(ref).strip()
    if raw.startswith("hf:"):
        name = raw[len("hf:") :]
        return "learned_hf", {"name": name}
    if ":" in raw:
        kind, payload = raw.split(":", 1)
        if kind in {"robomimic", "diffusion", "vla", "openvla", "pi0", "custom"}:
            return kind, {"checkpoint": payload}
    if raw.endswith((".pt", ".pth", ".ckpt")):
        return "robomimic", {"checkpoint": raw}
    return raw, {}


def load_policy_from_ref(
    ref: str,
    *,
    action_space: ActionSpace | str = ActionSpace.JOINT_POS,
    config: dict | None = None,
    policy_name: str | None = None,
) -> IPolicy:
    """Instantiate a policy from a registry name, checkpoint path, or HF alias."""
    cfg = dict(config or {})
    target_space = _action_space_from_value(action_space)
    kind, payload = parse_policy_ref(ref)

    if kind == "learned_hf":
        return HFModelRegistry.from_name(payload["name"], action_space=target_space, config=cfg)

    if kind in {"robomimic", "diffusion", "vla", "openvla", "pi0", "custom"}:
        framework = "vla" if kind == "vla" else kind
        if framework == "vla":
            framework = "custom"
        spec: ModelSpec = {
            "framework": framework,  # type: ignore[typeddict-item]
            "checkpoint": payload.get("checkpoint", ""),
            "expected_action_space": target_space,
            "expected_action_dim": int(cfg.get("arm_dof", cfg.get("expected_action_dim", 7))),
            "expected_obs_keys": list(cfg.get("expected_obs_keys", ["state"])),
            "metadata": dict(cfg),
        }
        registry_name = {
            "robomimic": "robomimic",
            "diffusion": "diffusion",
            "vla": "vla",
            "openvla": "vla",
            "pi0": "vla",
            "custom": "vla",
        }.get(kind, "vla")
        PolicyClass = get_policy(policy_name or registry_name)
        cfg["model_spec"] = spec
        return PolicyClass(config=cfg)

    PolicyClass = get_policy(policy_name or kind)
    return PolicyClass(config=cfg)


def load_model_spec_file(path: str | Path) -> ModelSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "expected_action_space" in payload:
        payload["expected_action_space"] = _action_space_from_value(payload["expected_action_space"])
    return payload  # type: ignore[return-value]


def load_policy_from_spec(
    spec: ModelSpec,
    *,
    policy_name: str,
    action_space: ActionSpace | str,
    config: dict | None = None,
) -> IPolicy:
    cfg = dict(config or {})
    cfg["model_spec"] = spec
    PolicyClass = get_policy(policy_name)
    return PolicyClass(
        config=cfg,
        action_space=_action_space_from_value(action_space),
    ) if "action_space" in PolicyClass.__init__.__code__.co_varnames else PolicyClass(config=cfg)
