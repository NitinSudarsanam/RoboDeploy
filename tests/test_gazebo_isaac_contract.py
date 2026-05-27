from __future__ import annotations

import unittest


class GazeboIsaacContractTests(unittest.TestCase):
    def test_gazebo_backend_registers(self):
        from robodeploy.builtins import import_builtins
        from robodeploy.core.registry import get_backend

        import_builtins()
        cls = get_backend("ros2_gazebo")
        self.assertFalse(cls.is_real)

    def test_isaac_backend_registers(self):
        from robodeploy.builtins import import_builtins
        from robodeploy.core.registry import get_backend

        import_builtins()
        cls = get_backend("isaacsim")
        self.assertFalse(cls.is_real)


if __name__ == "__main__":
    unittest.main()
