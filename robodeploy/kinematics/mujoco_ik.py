"""MuJoCo damped least-squares IK (lazy mujoco import)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend


class MujocoIkSolver:
    """Position IK using the live MuJoCo model attached to ``MuJoCoBackend``."""

    def __init__(
        self,
        *,
        model,
        data,
        qpos_addrs: list[int],
        dof_vel_ids: list[int],
        ee_body_id: int,
        q_min: np.ndarray,
        q_max: np.ndarray,
    ) -> None:
        self._model = model
        self._data = data
        self._qpos_addrs = list(qpos_addrs)
        self._dof_vel_ids = list(dof_vel_ids)
        self._ee_body_id = int(ee_body_id)
        self._q_min = np.asarray(q_min, dtype=np.float64)
        self._q_max = np.asarray(q_max, dtype=np.float64)
        self._mujoco = None

    @classmethod
    def from_backend(cls, backend: "MuJoCoBackend", description) -> "MujocoIkSolver":
        limits = np.asarray(description.joint_position_limits, dtype=np.float64)
        return cls(
            model=backend._model,
            data=backend._data,
            qpos_addrs=backend._qpos_addr,
            dof_vel_ids=backend._dof_addr,
            ee_body_id=backend._ee_body_id,
            q_min=limits[:, 0],
            q_max=limits[:, 1],
        )

    def _write_q(self, q: np.ndarray) -> None:
        for i, addr in enumerate(self._qpos_addrs):
            self._data.qpos[addr] = float(q[i])

    def _saved_q(self) -> np.ndarray:
        return np.array([self._data.qpos[addr] for addr in self._qpos_addrs], dtype=np.float64)

    def _restore_q(self, saved: np.ndarray) -> None:
        # IK iterates on the live sim data; the physical state must be put back
        # exactly, otherwise every solve teleports the robot mid-episode.
        self._write_q(saved)
        self._mujoco.mj_forward(self._model, self._data)

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
        import mujoco

        if self._mujoco is None:
            self._mujoco = mujoco

        q = np.asarray(q_init, dtype=np.float64).reshape(-1).copy()
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        eye = np.eye(3, dtype=np.float64)
        saved = self._saved_q()

        try:
            for _ in range(max_iter):
                self._write_q(q)
                mujoco.mj_forward(self._model, self._data)
                pos = np.asarray(self._data.xpos[self._ee_body_id], dtype=np.float64)
                err = target - pos
                if float(np.linalg.norm(err)) < pos_tol:
                    break

                jacp = np.zeros((3, self._model.nv), dtype=np.float64)
                jacr = np.zeros((3, self._model.nv), dtype=np.float64)
                mujoco.mj_jacBody(self._model, self._data, jacp, jacr, self._ee_body_id)
                j_arm = jacp[:, self._dof_vel_ids]
                dq = j_arm.T @ np.linalg.solve(j_arm @ j_arm.T + damping * eye, err)
                q = np.clip(q + step_scale * dq, self._q_min, self._q_max)
        finally:
            self._restore_q(saved)

        return q.astype(np.float32)

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        import mujoco

        if self._mujoco is None:
            self._mujoco = mujoco
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        saved = self._saved_q()
        try:
            self._write_q(q)
            mujoco.mj_forward(self._model, self._data)
            return np.asarray(self._data.xpos[self._ee_body_id], dtype=np.float32).copy()
        finally:
            self._restore_q(saved)


def attach_mujoco_ik(policy, backend, description) -> Optional[MujocoIkSolver]:
    """Bind IK to a policy if the backend is an initialized MuJoCo backend."""
    if backend is None or not hasattr(backend, "_model"):
        return None
    solver = MujocoIkSolver.from_backend(backend, description)
    if hasattr(policy, "set_ik_solver"):
        policy.set_ik_solver(solver)
    return solver
