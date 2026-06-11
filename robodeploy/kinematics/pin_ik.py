"""Pinocchio damped least-squares IK for non-MuJoCo backends (URDF-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from robodeploy.kinematics.solver import KinematicsSolver


class PinIkSolver:
    """Position IK using ``KinematicsSolver`` (Pinocchio) from a robot description.

    Mirrors ``MujocoIkSolver`` semantics so policies behave identically across
    backends: position-only DLS (no orientation term), joint limits clamped
    every iteration, best-effort return on non-convergence (never freezes the
    policy by returning ``q_init`` unchanged).
    """

    def __init__(
        self,
        solver: "KinematicsSolver",
        *,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
    ) -> None:
        self._solver = solver
        self._q_min = None if q_min is None else np.asarray(q_min, dtype=np.float64).reshape(-1)
        self._q_max = None if q_max is None else np.asarray(q_max, dtype=np.float64).reshape(-1)

    def solve(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        *,
        max_iter: int = 120,
        pos_tol: float = 0.008,
        step_scale: float = 0.35,
        damping: float = 0.05,
    ) -> np.ndarray:
        q = np.asarray(q_init, dtype=np.float64).reshape(-1).copy()
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        eye = np.eye(3, dtype=np.float64)
        for _ in range(max_iter):
            pos, _ = self._solver.fk(q)
            err = target - np.asarray(pos, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(err)) < pos_tol:
                break
            jac = np.asarray(self._solver.jacobian(q), dtype=np.float64)[:3, : q.shape[0]]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * eye, err)
            q = q + step_scale * dq
            if self._q_min is not None and self._q_max is not None:
                q = np.clip(q, self._q_min, self._q_max)
        return q.astype(np.float32)

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        pos, _ = self._solver.fk(np.asarray(q, dtype=np.float64).reshape(-1))
        return np.asarray(pos, dtype=np.float32).copy()


def attach_pin_ik(policy, description) -> Optional[PinIkSolver]:
    """Bind Pinocchio IK when the description exposes a URDF kinematics solver."""
    if description is None:
        return None
    try:
        solver = description.get_kinematics_solver()
    except Exception:
        return None
    q_min = q_max = None
    try:
        limits = np.asarray(description.joint_position_limits, dtype=np.float64)
        if limits.ndim == 2 and limits.shape[1] == 2:
            q_min, q_max = limits[:, 0], limits[:, 1]
    except Exception:
        pass
    pin_solver = PinIkSolver(solver, q_min=q_min, q_max=q_max)
    if hasattr(policy, "set_ik_solver"):
        policy.set_ik_solver(pin_solver)
    return pin_solver
