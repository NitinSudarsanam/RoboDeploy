"""Vision-language-action policy built on LearnedPolicyBase."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.types import Action, Observation
from robodeploy.policies.learned.base import LearnedPolicyBase
from robodeploy.policies.learned.helpers import coerce_action, configure_remote, resolve_action_space, vla_heuristic_action, vla_packet
from robodeploy.policies.learned.loader import ModelLoader


@register_policy("vla")
@register_policy("vla_stub")
class VLAPolicy(LearnedPolicyBase):
    def __init__(self, config: dict | None = None, *args, **kwargs) -> None:
        del args, kwargs
        cfg = dict(config or {})
        configure_remote(cfg)
        super().__init__(action_space=resolve_action_space(cfg), config=cfg, model_spec=cfg.get("model_spec"), loader=ModelLoader(predict_fn=cfg.get("predict_fn"), predict_batch_fn=cfg.get("predict_batch_fn")))
        self._camera = str(cfg.get("camera_name", "") or "").strip()
        self._max_delta = float(cfg.get("max_delta", 0.05))

    def get_action(self, obs: Observation) -> Action:
        packet = self.build_packet(obs)
        predict_fn = self.config.get("predict_fn")
        if callable(predict_fn):
            return coerce_action(predict_fn(packet), obs, self.action_space)
        if self._model is not None and self._adapter is not None:
            return self._adapter(np.asarray(self._model.predict_fn(packet)), obs)
        return vla_heuristic_action(obs, packet, self._max_delta, self.action_space)

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        predict_batch_fn = self.config.get("predict_batch_fn")
        if not callable(predict_batch_fn):
            return super().get_action_batch(obs_batch)
        return [coerce_action(o, obs, self.action_space) for o, obs in zip(predict_batch_fn([self.build_packet(x) for x in obs_batch]), obs_batch)]

    def build_packet(self, obs: Observation) -> dict[str, Any]:
        return vla_packet(obs, self._instruction, self._camera)
