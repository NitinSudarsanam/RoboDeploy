"""Hugging Face model registry for known learned policies."""

from __future__ import annotations

from robodeploy.core.spaces import ActionSpace
from robodeploy.policies.learned.base import LearnedPolicyBase
from robodeploy.policies.learned.loader import ModelLoader, ModelSpec


class HFModelRegistry:
    """Pull pre-trained policies from Hugging Face Hub by short name."""

    KNOWN_MODELS: dict[str, ModelSpec] = {
        "openvla-7b": {
            "framework": "openvla",
            "checkpoint": "hf://openvla/openvla-7b/model.pt",
            "expected_action_space": ActionSpace.DELTA_EE,
            "expected_action_dim": 7,
            "expected_obs_keys": ["rgb", "instruction"],
        },
        "octo-base": {
            "framework": "octo",
            "checkpoint": "hf://rail-berkeley/octo-base/model.pt",
            "expected_action_space": ActionSpace.DELTA_EE,
            "expected_action_dim": 7,
            "expected_obs_keys": ["rgb", "instruction"],
        },
        "pi0-base": {
            "framework": "pi0",
            "checkpoint": "hf://physical-intelligence/pi0/model.pt",
            "expected_action_space": ActionSpace.DELTA_EE,
            "expected_action_dim": 7,
            "expected_obs_keys": ["rgb", "instruction"],
        },
    }

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls.KNOWN_MODELS)

    @classmethod
    def get_spec(cls, name: str) -> ModelSpec:
        spec = cls.KNOWN_MODELS.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown model {name!r}. Known models: {', '.join(cls.list_models())}"
            )
        return dict(spec)

    @classmethod
    def download(cls, name: str, *, loader: ModelLoader | None = None) -> str:
        spec = cls.get_spec(name)
        resolved = (loader or ModelLoader()).resolve(spec["checkpoint"])
        return str(resolved)

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        action_space: ActionSpace,
        config: dict | None = None,
        loader: ModelLoader | None = None,
    ) -> LearnedPolicyBase:
        spec = cls.get_spec(name)
        return LearnedPolicyBase(
            action_space=action_space,
            model_spec=spec,
            config=config,
            loader=loader,
        )
