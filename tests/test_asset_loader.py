from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class AssetLoaderTests(unittest.TestCase):
    def test_list_robots_includes_builtins(self):
        from robodeploy.core.asset_loader import AssetLoader

        robots = AssetLoader().list_robots()
        self.assertIn("franka", robots)

    def test_info_known_robot(self):
        from robodeploy.core.asset_loader import AssetLoader

        info = AssetLoader().info("franka")
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.kind, "robot")

    def test_catalog_lists_formats(self):
        from robodeploy.core.asset_loader import AssetLoader

        rows = AssetLoader().catalog()
        self.assertTrue(any(row["name"] == "franka" for row in rows))

    def test_urdf_to_mjcf_conversion_for_so101(self):
        from robodeploy.core.asset_loader import AssetLoader
        from robodeploy.core.spaces import AssetFormat

        loader = AssetLoader()
        urdf = loader._resolve_format("so101", AssetFormat.URDF)
        if urdf is None:
            urdf = "description/so101/assets/urdf/so101.urdf"
        converted = loader._urdf_to_mjcf(urdf)
        try:
            import mujoco
        except ImportError:
            self.skipTest("mujoco not installed")
        if converted is None:
            self.skipTest("URDF→MJCF conversion unavailable in this environment")
        out = Path(converted)
        self.assertTrue(out.is_file())
        self.assertGreater(out.stat().st_size, 0)
