"""Pinocchio IK adapter tests (offline)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PinIkTests(unittest.TestCase):
    def test_solve_passes_ik_params_without_name_error(self):
        from robodeploy.kinematics.pin_ik import PinIkSolver

        class _FakeSolver:
            def __init__(self):
                self.last_max_iter = None
                self.last_tol = None

            def fk(self, q):
                del q
                return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

            def ik(self, target, quat, *, q_init=None, max_iter=120, tol=0.008):
                del target, quat
                self.last_max_iter = max_iter
                self.last_tol = tol
                return q_init

        fake = _FakeSolver()
        ik = PinIkSolver(fake)
        q = np.zeros(7, dtype=np.float32)
        result = ik.solve(q, np.zeros(3, dtype=np.float32), max_iter=50, pos_tol=0.01)
        self.assertEqual(fake.last_max_iter, 50)
        self.assertAlmostEqual(fake.last_tol, 0.01)
        np.testing.assert_array_equal(result, q)

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


if __name__ == "__main__":
    unittest.main()
