from __future__ import annotations

import unittest


class IsaacSimSmokeTests(unittest.TestCase):
    def test_backend_import_and_register(self):
        from robodeploy.builtins import import_builtins
        from robodeploy.core.registry import get_backend

        import_builtins()
        cls = get_backend("isaacsim")
        self.assertFalse(cls.is_real)
        self.assertEqual(cls.sensor_backend_name, "isaacsim")

    def test_scene_builder_import(self):
        from robodeploy.backends.sim.isaacsim.scene_builder import IsaacSceneBuilder

        builder = IsaacSceneBuilder()
        self.assertIsNotNone(builder)


if __name__ == "__main__":
    unittest.main()
