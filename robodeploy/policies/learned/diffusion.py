"""Concrete diffusion-style sequence policy with lightweight fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase
from robodeploy.policies.remote.http_client import HttpRemotePolicyClient

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]


@register_policy("diffusion")
@register_policy("diffusion_stub")
class DiffusionPolicy(PolicyBase):
    def __init__(self, config: dict | None = None, *args, **kwargs) -> None:
        del args, kwargs
        cfg = dict(config or {})
        super().__init__(action_space=self._resolve_action_space(cfg), config=cfg)
        self._plan_horizon = max(1, int(self.config.get("plan_horizon", 8)))
        self._replan_interval = max(1, int(self.config.get("replan_interval", 4)))
        self._max_delta = float(self.config.get("max_delta", 0.05))
        self._predict_plan_fn = self.config.get("predict_plan_fn")
        self._predict_batch_plan_fn = self.config.get("predict_batch_plan_fn")
        remote_url = str(self.config.get("remote_url", "") or "").strip()
        if remote_url and not callable(self._predict_plan_fn):
            client = HttpRemotePolicyClient(
                remote_url,
                batch_endpoint=str(self.config.get("remote_batch_url", "") or "").strip() or None,
                timeout_s=float(self.config.get("remote_timeout_s", 5.0)),
                transport=self.config.get("remote_transport"),
            )
            self._predict_plan_fn = client.predict
            if not callable(self._predict_batch_plan_fn):
                self._predict_batch_plan_fn = client.predict_batch
        self._queued_plan: list[Action] = []
        self._steps_since_plan = 0

    def _reset_impl(self) -> None:
        self._queued_plan = []
        self._steps_since_plan = 0

    def get_action(self, obs: Observation) -> Action:
        if not self._queued_plan or self._steps_since_plan >= self._replan_interval:
            self._queued_plan = self._build_plan(obs)
            self._steps_since_plan = 0
        action = self._queued_plan.pop(0)
        self._steps_since_plan += 1
        return action

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        if callable(self._predict_batch_plan_fn):
            outputs = list(self._predict_batch_plan_fn([self._packet(obs) for obs in obs_batch]))
            if len(outputs) != len(obs_batch):
                raise RuntimeError(
                    f"DiffusionPolicy.predict_batch_plan_fn returned {len(outputs)} plans "
                    f"for {len(obs_batch)} observations."
                )
            return [self._coerce_plan(output, obs)[0] for output, obs in zip(outputs, obs_batch)]
        return [self._build_plan(obs)[0] for obs in obs_batch]

    def notify_rejected(self, obs: Observation, action: Action) -> None:
        del obs, action
        self._queued_plan = []
        self._steps_since_plan = self._replan_interval

    @staticmethod
    def _resolve_action_space(config: dict) -> ActionSpace:
        raw = str(config.get("action_space", "delta_ee")).strip().upper()
        aliases = {
            "DELTA_EE": ActionSpace.DELTA_EE,
            "CARTESIAN_POSE": ActionSpace.CARTESIAN_POSE,
            "JOINT_POS": ActionSpace.JOINT_POS,
            "JOINT_VEL": ActionSpace.JOINT_VEL,
        }
        return aliases.get(raw, ActionSpace.DELTA_EE)

    def _build_plan(self, obs: Observation) -> list[Action]:
        if callable(self._predict_plan_fn):
            return self._coerce_plan(self._predict_plan_fn(self._packet(obs)), obs)
        return self._fallback_plan(obs)

    def _packet(self, obs: Observation) -> dict[str, Any]:
        instruction = str(obs.language_instruction or self._instruction or "").strip()
        return {
            "instruction": instruction,
            "rgb": obs.rgb,
            "images": dict(obs.images),
            "obs": obs,
            "plan_horizon": self._plan_horizon,
        }

    def _fallback_plan(self, obs: Observation) -> list[Action]:
        direction = self._keyword_delta(str(obs.language_instruction or self._instruction or "").lower())
        if not np.any(direction):
            direction[0] = self._max_delta
        plan: list[Action] = []
        for idx in range(self._plan_horizon):
            scale = 1.0 - (idx / max(1, self._plan_horizon))
            plan.append(self._action_from_delta(obs, direction * scale))
        return plan

    def _coerce_plan(self, value, obs: Observation) -> list[Action]:  # noqa: ANN001
        if isinstance(value, dict) and "actions" in value:
            value = value["actions"]
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"DiffusionPolicy predictor returned unsupported plan value: {type(value).__name__}"
            )
        actions = [self._coerce_action(item, obs) for item in value]
        if not actions:
            raise ValueError("DiffusionPolicy predictor returned an empty action plan.")
        return actions

    def _coerce_action(self, value, obs: Observation) -> Action:  # noqa: ANN001
        if isinstance(value, Action):
            if value.action_space is None:
                value.action_space = self.action_space
            if self.action_space == ActionSpace.DELTA_EE and not value.is_delta_ee:
                value.is_delta_ee = True
            return value
        if not isinstance(value, dict):
            raise TypeError(f"DiffusionPolicy action value must be Action or dict, got {type(value).__name__}")
        payload = dict(value)
        payload.setdefault("action_space", self.action_space)
        if self.action_space == ActionSpace.DELTA_EE:
            payload.setdefault("is_delta_ee", True)
        for field_name in ("joint_positions", "joint_velocities", "ee_position", "ee_orientation", "ee_velocity"):
            if payload.get(field_name) is not None:
                payload[field_name] = jnp.asarray(payload[field_name], dtype=jnp.float32)
        return Action(**payload)

    def _action_from_delta(self, obs: Observation, delta: np.ndarray) -> Action:
        if self.action_space in (ActionSpace.DELTA_EE, ActionSpace.CARTESIAN_POSE):
            return Action(
                ee_position=jnp.asarray(delta, dtype=jnp.float32),
                action_space=self.action_space,
                is_delta_ee=self.action_space == ActionSpace.DELTA_EE,
            )
        base = np.asarray(obs.joint_positions, dtype=np.float32)
        joint_delta = np.zeros_like(base)
        if joint_delta.shape[0] > 0:
            joint_delta[0] = delta[0]
        if joint_delta.shape[0] > 1:
            joint_delta[1] = delta[1]
        if joint_delta.shape[0] > 2:
            joint_delta[2] = delta[2]
        if self.action_space == ActionSpace.JOINT_VEL:
            return Action(joint_velocities=jnp.asarray(joint_delta, dtype=jnp.float32), action_space=self.action_space)
        return Action(
            joint_positions=jnp.asarray(base + joint_delta, dtype=jnp.float32),
            action_space=self.action_space,
        )

    def _keyword_delta(self, instruction: str) -> np.ndarray:
        delta = np.zeros(3, dtype=np.float32)
        step = float(self._max_delta)
        for token, axis, sign in (
            ("forward", 0, 1.0),
            ("back", 0, -1.0),
            ("left", 1, 1.0),
            ("right", 1, -1.0),
            ("up", 2, 1.0),
            ("down", 2, -1.0),
        ):
            if token in instruction:
                delta[axis] += sign * step
        return delta

