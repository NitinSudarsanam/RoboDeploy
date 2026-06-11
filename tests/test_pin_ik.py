"""Pinocchio IK adapter tests (offline)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _PlanarSolver:
    """2-link planar arm (unit links) standing in for KinematicsSolver."""

    def fk(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        x = np.cos(q[0]) + np.cos(q[0] + q[1])
        y = np.sin(q[0]) + np.sin(q[0] + q[1])
        return np.array([x, y, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])

    def jacobian(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        jac = np.zeros((6, 2), dtype=np.float64)
        jac[0] = [-s1 - s12, -s12]
        jac[1] = [c1 + c12, c12]
        return jac


class PinIkTests(unittest.TestCase):
    def test_solve_converges_position_only_dls(self):
        from robodeploy.kinematics.pin_ik import PinIkSolver

        ik = PinIkSolver(_PlanarSolver())
        q = ik.solve(np.array([0.3, 0.3]), np.array([1.2, 0.8, 0.0]), pos_tol=0.005)
        pos = ik.fk_position(q)
        self.assertLess(float(np.linalg.norm(pos[:2] - np.array([1.2, 0.8]))), 0.01)

    def test_solve_clamps_to_joint_limits(self):
        from robodeploy.kinematics.pin_ik import PinIkSolver

        q_min = np.array([-0.5, -0.5])
        q_max = np.array([0.5, 0.5])
        ik = PinIkSolver(_PlanarSolver(), q_min=q_min, q_max=q_max)
        # Target requires q outside +-0.5; result must stay clamped.
        q = ik.solve(np.zeros(2), np.array([-1.5, 1.0, 0.0]))
        self.assertTrue(bool((q >= q_min - 1e-6).all()))
        self.assertTrue(bool((q <= q_max + 1e-6).all()))

    def test_solve_unreachable_returns_best_effort_not_q_init(self):
        from robodeploy.kinematics.pin_ik import PinIkSolver

        ik = PinIkSolver(_PlanarSolver())
        q0 = np.array([0.3, 0.3])
        # Outside the reachable annulus (max reach 2.0): no convergence, but the
        # solver must still make progress toward the target instead of freezing.
        q = ik.solve(q0, np.array([3.0, 0.0, 0.0]))
        d0 = float(np.linalg.norm(ik.fk_position(q0)[:2] - np.array([3.0, 0.0])))
        d1 = float(np.linalg.norm(ik.fk_position(q)[:2] - np.array([3.0, 0.0])))
        self.assertLess(d1, d0)

    def test_kuka_urdf_fk_round_trip(self):
        try:
            import pinocchio  # noqa: F401
        except ImportError:
            self.skipTest("pinocchio not installed")

        from robodeploy.kinematics.pin_ik import PinIkSolver
        from robodeploy.description.kuka.description import KukaDescription

        desc = KukaDescription()
        solver = PinIkSolver(desc.get_kinematics_solver())
        q = np.array(desc.home_qpos, dtype=np.float32)
        pos = solver.fk_position(q)
        q2 = solver.solve(q, pos)
        pos2 = solver.fk_position(q2)
        self.assertLess(float(np.linalg.norm(pos - pos2)), 0.02)

    def test_kuka_pin_ik_reaches_pick_targets_within_limits(self):
        try:
            import pinocchio  # noqa: F401
        except ImportError:
            self.skipTest("pinocchio not installed")

        from robodeploy.description.kuka.description import KukaDescription
        from robodeploy.kinematics.pin_ik import attach_pin_ik

        class _P:
            def set_ik_solver(self, s):
                self.ik = s

        desc = KukaDescription()
        solver = attach_pin_ik(_P(), desc)
        self.assertIsNotNone(solver)
        q0 = np.asarray(desc.home_qpos, dtype=np.float64)
        limits = np.asarray(desc.joint_position_limits, dtype=np.float64)
        # Canonical pick-demo waypoints (pregrasp/lift/transit/place).
        for tgt in ((0.55, 0.0, 0.43), (0.55, 0.0, 0.505), (0.6, 0.2, 0.52), (0.6, 0.2, 0.42)):
            q = solver.solve(q0, np.array(tgt))
            pos = solver.fk_position(q)
            self.assertLess(float(np.linalg.norm(np.array(tgt) - pos)), 0.02, msg=str(tgt))
            self.assertTrue(bool((q >= limits[:, 0] - 1e-6).all()), msg=str(tgt))
            self.assertTrue(bool((q <= limits[:, 1] + 1e-6).all()), msg=str(tgt))

    def test_kuka_jacobian_nonzero(self):
        try:
            import pinocchio  # noqa: F401
        except ImportError:
            self.skipTest("pinocchio not installed")

        from robodeploy.description.kuka.description import KukaDescription

        desc = KukaDescription()
        solver = desc.get_kinematics_solver()
        jac = np.asarray(solver.jacobian(np.asarray(desc.home_qpos, dtype=np.float64)))
        # Regression: getFrameJacobian without computeJointJacobians read zeros.
        self.assertGreater(float(np.linalg.norm(jac[:3])), 0.1)


if __name__ == "__main__":
    unittest.main()
