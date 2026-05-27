from __future__ import annotations

import unittest


class MuJoCoSmokeTests(unittest.TestCase):
    def test_mujoco_backend_import_and_instantiate(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend

        backend = MuJoCoBackend()
        self.assertFalse(backend.is_real)


if __name__ == "__main__":
    unittest.main()
