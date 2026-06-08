"""Runtime IK binding for teleop cartesian commands."""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np


class IKSolver(Protocol):
    def solve(self, q_init: np.ndarray, target_pos: np.ndarray, **kwargs: Any) -> np.ndarray: ...

    def fk_position(self, q: np.ndarray) -> np.ndarray: ...


def build_ik(backend, description) -> Optional[IKSolver]:  # noqa: ANN001
    """Attach a position IK solver when the active backend supports it."""
    if backend is None or description is None:
        return None

    if hasattr(backend, "_model"):
        try:
            from examples.policies.mujoco_ik import MujocoIkSolver

            return MujocoIkSolver.from_backend(backend, description)
        except Exception:
            pass

    try:
        pin = description.get_kinematics_solver()
    except Exception:
        pin = None
    if pin is None:
        return None

    class _PinWrapper:
        def solve(self, q_init: np.ndarray, target_pos: np.ndarray, **kwargs: Any) -> np.ndarray:
            quat = kwargs.get("target_quat")
            if quat is None:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return pin.ik(target_pos, quat, q_init=q_init).astype(np.float32)

        def fk_position(self, q: np.ndarray) -> np.ndarray:
            pos, _ = pin.fk(q)
            return np.asarray(pos, dtype=np.float32).reshape(3)

    return _PinWrapper()
