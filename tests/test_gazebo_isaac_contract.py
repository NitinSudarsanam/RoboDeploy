from __future__ import annotations

import unittest


class GazeboIsaacContractTests(unittest.TestCase):
    def test_gazebo_backend_registers(self):
        from robodeploy.builtins import import_builtins
        from robodeploy.core.registry import get_backend

        import_builtins()
        cls = get_backend("ros2_gazebo")
        self.assertFalse(cls.is_real)
        self.assertEqual(cls.sensor_backend_name, "gazebo")

    def test_gazebo_backend_step_multi_contract_on_mock(self):
        from robodeploy.core.types import Action
        from robodeploy.testing import DummyBackend

        class _GazeboLike(DummyBackend):
            sensor_backend_name = "gazebo"

        backend = _GazeboLike()
        backend._initialized = True
        backend._robot_ids = ["robot0"]
        obs_list = backend.step_multi([Action()])
        self.assertEqual(len(obs_list), 1)

    def test_isaac_backend_registers(self):
        from robodeploy.builtins import import_builtins
        from robodeploy.core.registry import get_backend

        import_builtins()
        cls = get_backend("isaacsim")
        self.assertFalse(cls.is_real)


if __name__ == "__main__":
    unittest.main()
