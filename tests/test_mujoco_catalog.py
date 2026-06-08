from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.catalog.load import (
    build_config,
    find_combo,
    get_combo,
    list_combos,
    list_geom_kinds,
    list_robots,
    list_tasks,
    load_catalog,
)
from examples.config import load_example_preset, list_example_presets


class MuJoCoCatalogTests(unittest.TestCase):
    def test_catalog_yaml_parses(self):
        catalog = load_catalog()
        self.assertIn("robots", catalog)
        self.assertIn("tasks", catalog)
        self.assertIn("policies", catalog)
        self.assertIn("sensor_rigs", catalog)
        self.assertIn("geom_kinds", catalog)
        self.assertIn("kuka", catalog["robots"])
        self.assertIn("showcase_scene", catalog["tasks"])

    def test_list_helpers_return_sorted_names(self):
        self.assertIn("kuka", list_robots())
        self.assertIn("showcase_scene", list_tasks())
        self.assertIn("box", list_geom_kinds())

    def test_combo_preset_names_exist_in_presets_yaml(self):
        preset_names = set(list_example_presets())
        for combo_name in list_combos():
            combo = get_combo(combo_name)
            preset = combo.get("preset")
            self.assertIsNotNone(preset, msg=combo_name)
            self.assertIn(preset, preset_names, msg=combo_name)

    def test_find_combo_matches_showcase_kuka(self):
        self.assertEqual(
            find_combo(
                robot="kuka",
                task="showcase_scene",
                policy="example_joint_track",
                rig="full",
            ),
            "mujoco_showcase_kuka",
        )

    def test_build_config_loads_preset_when_combo_exists(self):
        cfg = build_config(
            robot="kuka",
            task="showcase_scene",
            policy="example_joint_track",
            rig="full",
        )
        preset = load_example_preset("mujoco_showcase_kuka")
        self.assertEqual(cfg["robot"], preset["robot"])
        self.assertEqual(cfg["task"], preset["task"])
        rig = cfg["sensor_rigs"][0]
        self.assertIn("wrist_imu", rig)
        self.assertIn("overhead_rgbd", rig)
        self.assertEqual(
            rig["prop_pose"]["prop_names"],
            ["showcase_box", "showcase_cylinder", "showcase_sphere", "showcase_capsule"],
        )

    def test_showcase_presets_match_catalog_combos(self):
        for combo_name in ("mujoco_showcase_kuka", "mujoco_showcase_franka", "mujoco_pick_kuka"):
            combo = get_combo(combo_name)
            preset = load_example_preset(combo["preset"])
            self.assertEqual(preset["robot"], combo["robot"])
            self.assertEqual(preset["task"], combo["task"])
            self.assertEqual(preset["policy"], combo["policy"])


if __name__ == "__main__":
    unittest.main()
