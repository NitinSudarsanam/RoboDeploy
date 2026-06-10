"""Pinocchio damped least-squares IK for non-MuJoCo backends (URDF-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from robodeploy.kinematics.solver import KinematicsSolver


class PinIkSolver:
    """Position IK using ``KinematicsSolver`` (Pinocchio) from a robot description."""

    def __init__(self, solver: "KinematicsSolver") -> None:
        self._solver = solver

    def solve(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        *,
        max_iter: int = 120,
        pos_tol: float = 0.008,
    ) -> np.ndarray:
        q = np.asarray(q_init, dtype=np.float64).reshape(-1)
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        _, quat = self._solver.fk(q)
        try:
            q_sol = self._solver.ik(target, quat, q_init=q, max_iter=max_iter, tol=pos_tol)
        except RuntimeError:
            q_sol = q
        return q_sol.astype(np.float32)

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        pos, _ = self._solver.fk(np.asarray(q, dtype=np.float64).reshape(-1))
        return np.asarray(pos, dtype=np.float32).copy()


def attach_pin_ik(policy, description) -> Optional[PinIkSolver]:
    """Bind Pinocchio IK when the description exposes a URDF kinematics solver."""
    if description is None:
        return None
    try:
        solver = PinIkSolver(description.get_kinematics_solver())
    except Exception:
        return None
    if hasattr(policy, "set_ik_solver"):
        policy.set_ik_solver(solver)
    return solver
