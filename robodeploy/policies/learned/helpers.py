"""Shared helpers for slim learned policy subclasses."""

from __future__ import annotations

from typing import Any

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.remote.http_client import HttpRemotePolicyClient

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]


def resolve_action_space(config: dict, default: ActionSpace = ActionSpace.DELTA_EE) -> ActionSpace:
    raw = str(config.get("action_space", default.name)).strip().upper()
    aliases = {
        "DELTA_EE": ActionSpace.DELTA_EE,
        "CARTESIAN_POSE": ActionSpace.CARTESIAN_POSE,
        "JOINT_POS": ActionSpace.JOINT_POS,
        "JOINT_VEL": ActionSpace.JOINT_VEL,
    }
    return aliases.get(raw, default)


def configure_remote(cfg: dict) -> None:
    remote_url = str(cfg.get("remote_url", "") or "").strip()
    if not remote_url:
        return
    client = HttpRemotePolicyClient(
        remote_url,
        batch_endpoint=str(cfg.get("remote_batch_url", "") or "").strip() or None,
        timeout_s=float(cfg.get("remote_timeout_s", 5.0)),
        transport=cfg.get("remote_transport"),
    )
    if not callable(cfg.get("predict_fn")):
        cfg["predict_fn"] = client.predict
    if not callable(cfg.get("predict_batch_fn")):
        cfg["predict_batch_fn"] = client.predict_batch
    if not callable(cfg.get("predict_plan_fn")):
        cfg["predict_plan_fn"] = client.predict
    if not callable(cfg.get("predict_batch_plan_fn")):
        cfg["predict_batch_plan_fn"] = client.predict_batch


def keyword_delta(instruction: str, *, max_delta: float, extra_tokens: tuple[str, ...] = ()) -> np.ndarray:
    delta = np.zeros(3, dtype=np.float32)
    step = float(max_delta)
    tokens = (
        ("forward", 0, 1.0),
        ("reach", 0, 1.0),
        ("back", 0, -1.0),
        ("left", 1, 1.0),
        ("right", 1, -1.0),
        ("up", 2, 1.0),
        ("lift", 2, 1.0),
        ("down", 2, -1.0),
    )
    for token, axis, sign in tokens:
        if token in instruction:
            delta[axis] += sign * step
    for token in extra_tokens:
        if token in instruction:
            delta[0] += step
    return delta


def coerce_action(value: Any, obs: Observation, action_space: ActionSpace) -> Action:
    if isinstance(value, Action):
        if value.action_space is None:
            value.action_space = action_space
        if action_space == ActionSpace.DELTA_EE and not value.is_delta_ee:
            value.is_delta_ee = True
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Predictor returned unsupported value: {type(value).__name__}")
    payload = dict(value)
    payload.setdefault("action_space", action_space)
    if action_space == ActionSpace.DELTA_EE:
        payload.setdefault("is_delta_ee", True)
    for field_name in ("joint_positions", "joint_velocities", "ee_position", "ee_orientation", "ee_velocity"):
        if payload.get(field_name) is not None:
            payload[field_name] = jnp.asarray(payload[field_name], dtype=jnp.float32)
    return Action(**payload)


def action_from_delta(obs: Observation, delta: np.ndarray, action_space: ActionSpace, gripper: float | None = None) -> Action:
    if action_space in (ActionSpace.DELTA_EE, ActionSpace.CARTESIAN_POSE):
        return Action(
            ee_position=jnp.asarray(delta[:3], dtype=jnp.float32),
            gripper=gripper,
            action_space=action_space,
            is_delta_ee=action_space == ActionSpace.DELTA_EE,
        )
    base = np.asarray(obs.joint_positions, dtype=np.float32)
    joint_delta = np.zeros_like(base)
    for idx in range(min(3, joint_delta.shape[0])):
        joint_delta[idx] = delta[idx]
    if action_space == ActionSpace.JOINT_VEL:
        return Action(joint_velocities=jnp.asarray(joint_delta, dtype=jnp.float32), gripper=gripper, action_space=action_space)
    return Action(joint_positions=jnp.asarray(base + joint_delta, dtype=jnp.float32), gripper=gripper, action_space=action_space)


def coerce_plan(value, obs: Observation, action_space: ActionSpace, adapter=None) -> list[Action]:  # noqa: ANN001
    if isinstance(value, dict) and "actions" in value:
        value = value["actions"]
    if not isinstance(value, (list, tuple)):
        if adapter is not None:
            return [adapter(np.asarray(value), obs)]
        return [coerce_action(value, obs, action_space)]
    actions = [coerce_action(item, obs, action_space) if not isinstance(item, Action) else item for item in value]
    if not actions:
        raise ValueError("Predictor returned an empty action plan.")
    return actions


def robomimic_default_spec(cfg: dict, predict_fn) -> dict:
    framework = "custom" if predict_fn or cfg.get("predict_fn") else "robomimic"
    return {
        "framework": framework,
        "checkpoint": str(cfg.get("checkpoint_path", "") or "missing.ckpt"),
        "expected_action_space": ActionSpace.JOINT_POS,
        "expected_action_dim": int(cfg.get("arm_dof", 7)) + 1,
        "expected_obs_keys": [str(cfg.get("obs_key", "state"))],
        "metadata": {
            "obs_key": cfg.get("obs_key", "state"),
            "arm_dof": cfg.get("arm_dof", 7),
            "use_cuda": cfg.get("use_cuda", True),
            "verbose": cfg.get("verbose", False),
        },
    }


def image_centroid_delta(rgb, max_delta: float) -> np.ndarray:  # noqa: ANN001
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
    scale = float(max_delta) * 0.5
    return np.asarray([-norm_x * scale, -norm_y * scale], dtype=np.float32)
