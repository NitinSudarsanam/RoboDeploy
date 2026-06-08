from __future__ import annotations

import unittest


class AssetManifestTests(unittest.TestCase):
    def test_verify_assets_ok(self):
        from robodeploy.assets import verify_assets

        rows = verify_assets()
        self.assertTrue(rows)
        self.assertTrue(all(r.get("status") == "ok" for r in rows))


if __name__ == "__main__":
    unittest.main()
