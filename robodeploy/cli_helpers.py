"""Shared CLI helpers for robodeploy and examples CLIs (no preset loading)."""

from __future__ import annotations

import json
from typing import Any, Callable


def print_json(payload: Any, *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload))


def close_quietly(env) -> None:  # noqa: ANN001
    try:
        env.close()
    except Exception:
        pass


def episode_info_summary(info) -> dict[str, Any]:  # noqa: ANN001
    extra = getattr(info, "extra", {}) or {}
    keep = ("diagnostics", "multi_agent")
    return {
        "episode_id": int(getattr(info, "episode_id", 0)),
        "step": int(getattr(info, "step", 0)),
        "reward": float(getattr(info, "reward", 0.0)),
        "success": bool(getattr(info, "success", False)),
        "failure": bool(getattr(info, "failure", False)),
        "truncated": bool(extra.get("truncated", False)),
        "extra": {k: extra.get(k) for k in keep if k in extra},
    }


def action_fn_for_mode(mode: str, env) -> Callable | None:  # noqa: ANN001
    if mode == "none":
        return None

    try:
        import jax.numpy as jnp
    except Exception:
        import numpy as jnp  # type: ignore[assignment]

    from robodeploy.core.types import Action

    dof = int(getattr(env.primary_robot.description, "dof", 0) or 0)
    home = getattr(env.primary_robot.description, "home_qpos", None)
    if home is not None:
        home_arr = jnp.asarray(home, dtype=jnp.float32)
        dof = int(home_arr.shape[0])
    else:
        home_arr = jnp.zeros((dof,), dtype=jnp.float32)

    if mode == "zero":
        zeros = jnp.zeros((dof,), dtype=jnp.float32)

        def _fn(_obs):  # noqa: ANN001
            return Action(joint_positions=zeros)

        return _fn

    if mode == "hold":

        def _fn(_obs):  # noqa: ANN001
            return Action(joint_positions=home_arr)

        return _fn

    if mode == "sinusoid":
        t = {"i": 0}
        amp = 0.1
        omega = 0.2

        def _fn(_obs):  # noqa: ANN001
            t["i"] += 1
            phase = float(t["i"]) * omega
            delta = amp * jnp.sin(phase) * jnp.ones((dof,), dtype=jnp.float32)
            return Action(joint_positions=home_arr + delta)

        return _fn

    raise ValueError(f"Unknown --action mode: {mode}")
