"""Offline check that bundled Kuka URDF has arm collision geometry for FT grasp."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KUKA_URDF = REPO_ROOT / "robodeploy" / "description" / "kuka" / "assets" / "urdf" / "kuka.urdf"

ARM_LINKS = tuple(f"link{i}" for i in range(1, 8)) + ("ee_link",)


class KukaUrdfCollisionTests(unittest.TestCase):
    def test_arm_links_have_collision_geometry(self):
        root = ET.parse(KUKA_URDF).getroot()
        links_with_collision: set[str] = set()
        for link in root.findall("link"):
            name = link.get("name", "")
            if link.find("collision") is not None:
                links_with_collision.add(name)

        missing = [name for name in ARM_LINKS if name not in links_with_collision]
        self.assertFalse(missing, msg=f"links missing <collision>: {missing}")

    def test_ee_link_has_geometry_primitive(self):
        root = ET.parse(KUKA_URDF).getroot()
        ee = next(link for link in root.findall("link") if link.get("name") == "ee_link")
        collision = ee.find("collision")
        self.assertIsNotNone(collision)
        geom = collision.find("geometry") if collision is not None else None
        self.assertIsNotNone(geom)
        has_shape = any(geom.find(tag) is not None for tag in ("box", "cylinder", "sphere", "mesh"))
        self.assertTrue(has_shape)


if __name__ == "__main__":
    unittest.main()
