"""Vision-language-action policy built on LearnedPolicyBase."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.types import Action, Observation
from robodeploy.policies.learned.base import LearnedPolicyBase
from robodeploy.policies.learned.helpers import action_from_delta, coerce_action, configure_remote, image_centroid_delta, keyword_delta, resolve_action_space
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
        return self._heuristic(obs, packet)

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        predict_batch_fn = self.config.get("predict_batch_fn")
        if not callable(predict_batch_fn):
            return super().get_action_batch(obs_batch)
        return [coerce_action(o, obs, self.action_space) for o, obs in zip(predict_batch_fn([self.build_packet(x) for x in obs_batch]), obs_batch)]

    def build_packet(self, obs: Observation) -> dict[str, Any]:
        return {"instruction": str(obs.language_instruction or self._instruction or "").strip(), "rgb": self._image(obs), "depth": self._depth(obs), "images": dict(obs.images), "depths": dict(obs.depths), "obs": obs}

    def _heuristic(self, obs: Observation, packet: dict[str, Any]) -> Action:
        text = packet["instruction"].lower()
        delta = keyword_delta(text, max_delta=self._max_delta)
        img = image_centroid_delta(packet.get("rgb"), self._max_delta)
        if delta.shape[0] > 1:
            delta[1] += img[0]
        if delta.shape[0] > 2:
            delta[2] += img[1]
        gripper = 1.0 if "close" in text or "grasp" in text else (0.0 if "open" in text or "release" in text else None)
        return action_from_delta(obs, delta, self.action_space, gripper=gripper)

    def _image(self, obs: Observation):  # noqa: ANN001
        if self._camera and self._camera in obs.images:
            return obs.images[self._camera]
        if obs.rgb is not None:
            return obs.rgb
        return next(iter(obs.images.values())) if obs.images else None

    def _depth(self, obs: Observation):  # noqa: ANN001
        if self._camera and self._camera in obs.depths:
            return obs.depths[self._camera]
        if obs.depth is not None:
            return obs.depth
        return next(iter(obs.depths.values())) if obs.depths else None
