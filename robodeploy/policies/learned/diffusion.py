"""Diffusion-style sequence policy built on LearnedPolicyBase."""

from __future__ import annotations

from robodeploy.core.registry import register_policy
from robodeploy.core.types import Action, Observation
from robodeploy.policies.learned.base import LearnedPolicyBase
from robodeploy.policies.learned.helpers import PlanQueue, batch_first_actions, build_plan, configure_remote, resolve_action_space
from robodeploy.policies.learned.loader import ModelLoader


@register_policy("diffusion")
@register_policy("diffusion_stub")
class DiffusionPolicy(LearnedPolicyBase):
    def __init__(self, config: dict | None = None, *args, **kwargs) -> None:
        del args, kwargs
        cfg = dict(config or {})
        configure_remote(cfg)
        batch_plan_fn = cfg.get("predict_batch_plan_fn") or cfg.get("predict_batch_fn")
        loader = ModelLoader(predict_fn=cfg.get("predict_fn"), predict_plan_fn=cfg.get("predict_plan_fn"), predict_batch_fn=batch_plan_fn)
        super().__init__(action_space=resolve_action_space(cfg), config=cfg, model_spec=cfg.get("model_spec"), loader=loader)
        self._horizon, self._max_delta = max(1, int(cfg.get("plan_horizon", 8))), float(cfg.get("max_delta", 0.05))
        self._plan_fn, self._batch_plan_fn = cfg.get("predict_plan_fn"), batch_plan_fn
        self._queue = PlanQueue(cfg.get("replan_interval", 4))

    def _reset_impl(self, seed: int | None = None) -> None:
        del seed
        self._queue.reset()

    def get_action(self, obs: Observation) -> Action:
        return self._queue.next_action(lambda: self._build_plan(obs))

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        if callable(self._batch_plan_fn):
            return batch_first_actions(self._batch_plan_fn, obs_batch, instruction=self._instruction, horizon=self._horizon, action_space=self.action_space, adapter=self._adapter)
        return [self._build_plan(o)[0] for o in obs_batch]

    def predict_chunked(self, obs: Observation, *, chunk_size: int = 4):
        plan = self._build_plan(obs)
        for idx in range(0, len(plan), chunk_size):
            yield plan[idx : idx + chunk_size]

    def notify_rejected(self, obs: Observation, action: Action) -> None:
        del obs, action
        self._queue.invalidate()

    def _build_plan(self, obs: Observation) -> list[Action]:
        plan_fn = self._plan_fn or (self._model.predict_fn if self._model is not None else None)
        return build_plan(obs, plan_fn=plan_fn, instruction=self._instruction, horizon=self._horizon, max_delta=self._max_delta, action_space=self.action_space, adapter=self._adapter)
