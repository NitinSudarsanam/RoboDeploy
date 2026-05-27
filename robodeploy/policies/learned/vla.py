"""Concrete vision-language-action policy with lightweight fallback."""

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


@register_policy("vla")
@register_policy("vla_stub")
class VLAPolicy(PolicyBase):
    def __init__(self, config: dict | None = None, *args, **kwargs) -> None:
        del args, kwargs
        cfg = dict(config or {})
        super().__init__(action_space=self._resolve_action_space(cfg), config=cfg)
        self._camera_name = str(self.config.get("camera_name", "") or "").strip()
        self._max_delta = float(self.config.get("max_delta", 0.05))
        self._predict_fn = self.config.get("predict_fn")
        self._predict_batch_fn = self.config.get("predict_batch_fn")
        remote_url = str(self.config.get("remote_url", "") or "").strip()
        if remote_url and not callable(self._predict_fn):
            client = HttpRemotePolicyClient(
                remote_url,
                batch_endpoint=str(self.config.get("remote_batch_url", "") or "").strip() or None,
                timeout_s=float(self.config.get("remote_timeout_s", 5.0)),
                transport=self.config.get("remote_transport"),
            )
            self._predict_fn = client.predict
            if not callable(self._predict_batch_fn):
                self._predict_batch_fn = client.predict_batch

    def get_action(self, obs: Observation) -> Action:
        packet = self.build_packet(obs)
        if callable(self._predict_fn):
            return self._coerce_action(self._predict_fn(packet), obs)
        return self._heuristic_action(obs, packet)

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        if callable(self._predict_batch_fn):
            packets = [self.build_packet(obs) for obs in obs_batch]
            outputs = list(self._predict_batch_fn(packets))
            if len(outputs) != len(obs_batch):
                raise RuntimeError(
                    f"VLAPolicy.predict_batch_fn returned {len(outputs)} actions "
                    f"for {len(obs_batch)} observations."
                )
            return [self._coerce_action(output, obs) for output, obs in zip(outputs, obs_batch)]
        return super().get_action_batch(obs_batch)

    def build_packet(self, obs: Observation) -> dict[str, Any]:
        instruction = str(obs.language_instruction or self._instruction or "").strip()
        rgb = self._select_image(obs)
        depth = self._select_depth(obs)
        return {
            "instruction": instruction,
            "rgb": rgb,
            "depth": depth,
            "images": dict(obs.images),
            "depths": dict(obs.depths),
            "obs": obs,
        }

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

    def _select_image(self, obs: Observation):  # noqa: ANN001
        if self._camera_name and self._camera_name in obs.images:
            return obs.images[self._camera_name]
        if obs.rgb is not None:
            return obs.rgb
        if obs.images:
            return next(iter(obs.images.values()))
        return None

    def _select_depth(self, obs: Observation):  # noqa: ANN001
        if self._camera_name and self._camera_name in obs.depths:
            return obs.depths[self._camera_name]
        if obs.depth is not None:
            return obs.depth
        if obs.depths:
            return next(iter(obs.depths.values()))
        return None

    def _heuristic_action(self, obs: Observation, packet: dict[str, Any]) -> Action:
        instruction = packet["instruction"].lower()
        delta = np.zeros(3, dtype=np.float32)
        delta += self._keyword_delta(instruction)
        image_delta = self._image_delta(packet.get("rgb"))
        delta[1] += image_delta[0]
        delta[2] += image_delta[1]
        gripper = self._keyword_gripper(instruction)

        if self.action_space in (ActionSpace.DELTA_EE, ActionSpace.CARTESIAN_POSE):
            return Action(
                ee_position=jnp.asarray(delta, dtype=jnp.float32),
                gripper=gripper,
                action_space=self.action_space,
                is_delta_ee=self.action_space == ActionSpace.DELTA_EE,
            )

        base = np.asarray(obs.joint_positions, dtype=np.float32)
        joint_delta = np.zeros_like(base)
        if joint_delta.shape[0] > 0:
            joint_delta[0] = delta[1]
        if joint_delta.shape[0] > 1:
            joint_delta[1] = delta[2]
        if joint_delta.shape[0] > 2:
            joint_delta[2] = delta[0]
        if self.action_space == ActionSpace.JOINT_VEL:
            return Action(
                joint_velocities=jnp.asarray(joint_delta, dtype=jnp.float32),
                gripper=gripper,
                action_space=self.action_space,
            )
        return Action(
            joint_positions=jnp.asarray(base + joint_delta, dtype=jnp.float32),
            gripper=gripper,
            action_space=self.action_space,
        )

    def _coerce_action(self, value, obs: Observation) -> Action:  # noqa: ANN001
        if isinstance(value, Action):
            if value.action_space is None:
                value.action_space = self.action_space
            if self.action_space == ActionSpace.DELTA_EE and not value.is_delta_ee:
                value.is_delta_ee = True
            return value
        if not isinstance(value, dict):
            raise TypeError(f"VLAPolicy predictor returned unsupported value: {type(value).__name__}")
        payload = dict(value)
        payload.setdefault("action_space", self.action_space)
        if self.action_space == ActionSpace.DELTA_EE:
            payload.setdefault("is_delta_ee", True)
        for field_name in ("joint_positions", "joint_velocities", "ee_position", "ee_orientation", "ee_velocity"):
            if payload.get(field_name) is not None:
                payload[field_name] = jnp.asarray(payload[field_name], dtype=jnp.float32)
        return Action(**payload)

    def _keyword_delta(self, instruction: str) -> np.ndarray:
        delta = np.zeros(3, dtype=np.float32)
        step = float(self._max_delta)
        for token, axis, sign in (
            ("forward", 0, 1.0),
            ("reach", 0, 1.0),
            ("back", 0, -1.0),
            ("left", 1, 1.0),
            ("right", 1, -1.0),
            ("up", 2, 1.0),
            ("lift", 2, 1.0),
            ("down", 2, -1.0),
        ):
            if token in instruction:
                delta[axis] += sign * step
        return delta

    @staticmethod
    def _keyword_gripper(instruction: str) -> float | None:
        if "close" in instruction or "grasp" in instruction:
            return 1.0
        if "open" in instruction or "release" in instruction:
            return 0.0
        return None

    def _image_delta(self, rgb) -> np.ndarray:  # noqa: ANN001
        if rgb is None:
            return np.zeros(2, dtype=np.float32)
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return np.zeros(2, dtype=np.float32)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        weights = arr.mean(axis=-1)
        total = float(weights.sum())
        if total <= 0.0:
            return np.zeros(2, dtype=np.float32)
        ys, xs = np.indices(weights.shape)
        cx = float((weights * xs).sum() / total)
        cy = float((weights * ys).sum() / total)
        norm_x = 0.0 if weights.shape[1] <= 1 else (cx / (weights.shape[1] - 1)) * 2.0 - 1.0
        norm_y = 0.0 if weights.shape[0] <= 1 else (cy / (weights.shape[0] - 1)) * 2.0 - 1.0
        scale = float(self._max_delta) * 0.5
        return np.asarray([-norm_x * scale, -norm_y * scale], dtype=np.float32)

