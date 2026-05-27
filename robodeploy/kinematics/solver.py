"""
KinematicsSolver — FK and IK via Pinocchio.

Works entirely from a RobotDescription (reads the URDF).
No simulator, no ROS2, no hardware required.

This is intentionally a thin wrapper over Pinocchio. The goal is not to
re-implement kinematics — it is to give every component a single consistent
entry point:

  description.get_kinematics_solver().fk(qpos)   → ee_pose
  description.get_kinematics_solver().ik(target)  → joint_angles

RoboEnv, real backends, and offline trajectory planners all use the same
object, so FK/IK results are identical regardless of context.

Pinocchio is the industry-standard kinematics library for manipulation.
It handles arbitrary kinematic chains, multiple end-effectors, and
supports analytic Jacobians for velocity IK. Install with:
    pip install pin   (Linux/Mac)
    pip install pinocchio  (alternative)
"""

from __future__ import annotations

import warnings

import numpy as np

from robodeploy.core.spaces import AssetFormat

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from robodeploy.description.base import RobotDescription


class KinematicsSolver:
    """FK and IK solver backed by Pinocchio, initialised from a RobotDescription.

    Do not instantiate directly. Access via:
        description.get_kinematics_solver()

    Args:
        description: The robot's static definition. Provides URDF path and
                     joint metadata.
    """

    def __init__(self, description: RobotDescription) -> None:
        self._description = description
        self._model  = None   # pinocchio.Model, lazy-loaded
        self._data   = None   # pinocchio.Data, lazy-loaded

    # ------------------------------------------------------------------
    # Lazy Pinocchio initialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load Pinocchio model on first use. Raises ImportError if not installed."""
        if self._model is not None:
            return
        try:
            import pinocchio as pin  # type: ignore
        except ImportError:
            raise ImportError(
                "Pinocchio is required for KinematicsSolver.\n"
                "Install with: pip install pin"
            )
        urdf_path = str(self._description.asset_path(AssetFormat.URDF))
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data  = self._model.createData()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fk(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: joint angles → end-effector pose.

        Args:
            joint_positions: Joint angles in radians, shape [dof].

        Returns:
            Tuple of:
              position:    np.ndarray [3]  — end-effector position in metres.
              orientation: np.ndarray [4]  — quaternion (w, x, y, z).
        """
        import pinocchio as pin  # type: ignore
        self._ensure_loaded()

        q = np.asarray(joint_positions, dtype=np.float64)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)

        ee_id = self._model.getFrameId(self._description.ee_link_name)
        placement = self._data.oMf[ee_id]

        position    = np.array(placement.translation)
        orientation = pin.Quaternion(placement.rotation)
        quat        = np.array([orientation.w, orientation.x,
                                orientation.y, orientation.z])
        return position, quat

    def ik(
        self,
        target_position:    np.ndarray,
        target_orientation: np.ndarray,
        q_init:             np.ndarray | None = None,
        max_iter:           int    = 200,
        tol:                float  = 1e-4,
    ) -> np.ndarray:
        """Inverse kinematics: target end-effector pose → joint angles.

        Uses Pinocchio's damped least-squares (pseudo-inverse Jacobian) IK.
        Iterative — converges from q_init. If q_init is None, uses the
        robot's home_qpos as the starting point.

        Args:
            target_position:    Desired end-effector position [3] in metres.
            target_orientation: Desired end-effector orientation [4] quaternion
                                (w, x, y, z).
            q_init:             Starting joint configuration [dof]. Defaults to
                                description.home_qpos.
            max_iter:           Maximum IK iterations before giving up.
            tol:                Convergence tolerance on task-space error (metres).

        Returns:
            np.ndarray [dof]: Joint angles in radians that achieve the target pose
                              (within tolerance).

        Raises:
            RuntimeError: If IK fails to converge within max_iter iterations.
        """
        import pinocchio as pin  # type: ignore
        self._ensure_loaded()

        q = np.array(
            q_init if q_init is not None else self._description.home_qpos,
            dtype=np.float64,
        )

        # Build target SE3 from position + quaternion
        w, x, y, z = target_orientation
        R = pin.Quaternion(w, x, y, z).toRotationMatrix()
        target_se3 = pin.SE3(R, np.array(target_position, dtype=np.float64))

        ee_id = self._model.getFrameId(self._description.ee_link_name)

        for _ in range(max_iter):
            pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)

            current = self._data.oMf[ee_id]
            err_se3 = current.actInv(target_se3)
            err     = pin.log6(err_se3).vector

            if np.linalg.norm(err[:3]) < tol:
                return q

            J = pin.computeFrameJacobian(
                self._model, self._data, q, ee_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            lam = 1e-4
            dq  = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            q   = pin.integrate(self._model, q, dq * 0.5)

        raise RuntimeError(
            f"IK failed to converge in {max_iter} iterations. "
            "Try a different q_init or increase max_iter."
        )

    def jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """Geometric Jacobian at the end-effector frame.

        Args:
            joint_positions: Joint angles [dof] in radians.

        Returns:
            np.ndarray [6, dof]: Jacobian matrix (top 3 rows: linear velocity;
                                  bottom 3 rows: angular velocity).
        """
        import pinocchio as pin  # type: ignore
        self._ensure_loaded()

        q     = np.asarray(joint_positions, dtype=np.float64)
        ee_id = self._model.getFrameId(self._description.ee_link_name)
        pin.computeFrameJacobian(
            self._model, self._data, q, ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return np.array(pin.getFrameJacobian(
            self._model, self._data, ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ))

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        *,
        steps: int = 50,
        obstacles=None,  # reserved for future OMPL/MoveIt2 integration
        unsafe_straight_line: bool = False,
    ) -> list[np.ndarray]:
        """Plan a joint-space path from q_start to q_goal.

        This is a minimal, deterministic fallback planner (straight-line in joint space).
        It exists to provide the contract surface required by ARCHITECTURE.md and the
        Arbitrator switch sequencing. Real deployments should replace this with a
        collision-aware planner.
        """
        if obstacles is not None:
            raise NotImplementedError("Collision-aware planning with obstacles is not implemented.")
        if not unsafe_straight_line:
            raise RuntimeError(
                "KinematicsSolver.plan() only has a straight-line joint-space fallback. "
                "Pass unsafe_straight_line=True to acknowledge it is not collision-aware."
            )
        warnings.warn(
            "KinematicsSolver.plan() is using a straight-line joint-space fallback; "
            "it is not collision-aware.",
            RuntimeWarning,
            stacklevel=2,
        )
        qs = np.asarray(q_start, dtype=np.float64).reshape(-1)
        qg = np.asarray(q_goal, dtype=np.float64).reshape(-1)
        if qs.shape != qg.shape:
            raise ValueError("q_start and q_goal must have the same shape.")
        n = int(steps)
        n = max(2, n)
        out: list[np.ndarray] = []
        for i in range(n):
            a = float(i) / float(n - 1)
            out.append(((1.0 - a) * qs + a * qg).copy())
        return out
