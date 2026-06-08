"""Diffusion-style sequence policy built on LearnedPolicyBase."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.types import Action, Observation
from robodeploy.policies.learned.base import LearnedPolicyBase
from robodeploy.policies.learned.helpers import action_from_delta, coerce_plan, configure_remote, keyword_delta, resolve_action_space
from robodeploy.policies.learned.loader import ModelLoader


@register_policy("diffusion")
@register_policy("diffusion_stub")
class DiffusionPolicy(LearnedPolicyBase):
    def __init__(self, config: dict | None = None, *args, **kwargs) -> None:
        del args, kwargs
        cfg = dict(config or {})
        configure_remote(cfg)
        batch_plan_fn = cfg.get("predict_batch_plan_fn") or cfg.get("predict_batch_fn")
        super().__init__(
            action_space=resolve_action_space(cfg),
            config=cfg,
            model_spec=cfg.get("model_spec"),
            loader=ModelLoader(
                predict_fn=cfg.get("predict_fn"),
                predict_plan_fn=cfg.get("predict_plan_fn"),
                predict_batch_fn=batch_plan_fn,
            ),
        )
        self._horizon = max(1, int(cfg.get("plan_horizon", 8)))
        self._replan = max(1, int(cfg.get("replan_interval", 4)))
        self._max_delta = float(cfg.get("max_delta", 0.05))
        self._plan_fn = cfg.get("predict_plan_fn")
        self._batch_plan_fn = batch_plan_fn
        self._queue: list[Action] = []
        self._since_plan = 0

    def _reset_impl(self, seed: int | None = None) -> None:
        del seed
        self._queue, self._since_plan = [], 0

    def get_action(self, obs: Observation) -> Action:
        if not self._queue or self._since_plan >= self._replan:
            self._queue, self._since_plan = self._build_plan(obs), 0
        self._since_plan += 1
        return self._queue.pop(0)

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        if callable(self._batch_plan_fn):
            outs = list(self._batch_plan_fn([self._packet(o) for o in obs_batch]))
            return [coerce_plan(v, o, self.action_space, self._adapter)[0] for v, o in zip(outs, obs_batch)]
        return [self._build_plan(o)[0] for o in obs_batch]

    def predict_chunked(self, obs: Observation, *, chunk_size: int = 4) -> Iterator[list[Action]]:
        plan = self._build_plan(obs)
        for idx in range(0, len(plan), chunk_size):
            yield plan[idx : idx + chunk_size]

    def notify_rejected(self, obs: Observation, action: Action) -> None:
        del obs, action
        self._queue, self._since_plan = [], self._replan

    def _build_plan(self, obs: Observation) -> list[Action]:
        packet = self._packet(obs)
        if callable(self._plan_fn):
            return coerce_plan(self._plan_fn(packet), obs, self.action_space, self._adapter)
        if self._model is not None:
            return coerce_plan(self._model.predict_fn(packet), obs, self.action_space, self._adapter)
        direction = keyword_delta(str(obs.language_instruction or self._instruction or "").lower(), max_delta=self._max_delta)
        if not np.any(direction):
            direction[0] = self._max_delta
        return [action_from_delta(obs, direction * (1.0 - i / max(1, self._horizon)), self.action_space) for i in range(self._horizon)]

    def _packet(self, obs: Observation) -> dict[str, Any]:
        return {
            "instruction": str(obs.language_instruction or self._instruction or "").strip(),
            "rgb": obs.rgb,
            "images": dict(obs.images),
            "obs": obs,
            "plan_horizon": self._horizon,
        }
