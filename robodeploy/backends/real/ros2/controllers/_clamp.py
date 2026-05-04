"""Shared joint-space command shaping (slew rate limit)."""

from __future__ import annotations

import numpy as np


def slew_limit_command(
    q_des: np.ndarray,
    q_cur: np.ndarray,
    *,
    max_joint_velocity: np.ndarray | None,
    command_hz: float,
) -> np.ndarray:
    """Clamp ``q_des`` toward ``q_cur`` so per-step motion does not exceed ``max_joint_velocity * dt``.

    Matches previous ``joint_position`` behavior: if ``max_joint_velocity`` is ``None``,
    wrong length, or ``command_hz <= 0``, returns ``q_des`` unchanged.
    """
    q_des = np.asarray(q_des, dtype=np.float64).reshape(-1)
    if max_joint_velocity is None:
        return q_des
    mv = np.asarray(max_joint_velocity, dtype=np.float64).reshape(-1)
    if mv.size != q_des.size or float(command_hz) <= 0.0:
        return q_des
    dt = 1.0 / float(command_hz)
    q_cur = np.asarray(q_cur, dtype=np.float64).reshape(-1)
    if q_cur.size != q_des.size:
        return q_des
    max_step = mv * dt
    return np.minimum(np.maximum(q_des, q_cur - max_step), q_cur + max_step)
