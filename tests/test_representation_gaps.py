"""Tests for REPRESENTATION_UPGRADE_PLAN sections A-F gap items."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.config import load_example_preset, list_example_presets
from robodeploy.backends.camera_presets import get_camera_preset
from robodeploy.backends.sim.gazebo.scene_builder import GazeboSceneBuilder
from robodeploy.backends.sim.mujoco.scene_builder import MjcfSceneBuilder
from robodeploy.core.procedural_terrain import ProceduralTerrainGenerator
from robodeploy.core.scene_ir import (
    SceneIR,
    UnifiedTerrain,
    assert_cross_backend_pose_equivalence,
    world_to_ir,
)
from robodeploy.core.types import GeomSpec, PropConfig, TerrainSpec, WorldSpec
from robodeploy.scene_builder import SceneBuilder
from robodeploy.tasks.choreography import TaskChoreography
from robodeploy.tasks.success_predicates import liquid_in_target, peg_in_hole


class RepresentationGapTests(unittest.TestCase):
    def test_reach_pick_place_policy_under_50_lines(self):
        path = REPO_ROOT / "examples/policies/reach_pick_place.py"
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertLessEqual(len(lines), 50)

    def test_pick_place_task_under_25_lines(self):
        path = REPO_ROOT / "examples/tasks/pick_place.py"
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(lines), 25)

    def test_procedural_generators_ridge_and_stairs(self):
        ridge = ProceduralTerrainGenerator.ridge(resolution=32, seed=1)
        stairs = ProceduralTerrainGenerator.stairs(resolution=32, num_steps=6)
        self.assertEqual(ridge.shape, (32, 32))
        self.assertEqual(stairs.shape, (32, 32))
        self.assertGreater(float(ridge.std()), 0.0)
        self.assertGreater(float(stairs.max() - stairs.min()), 0.0)

    def test_mujoco_procedural_terrain_emits_hfield(self):
        world = WorldSpec(
            terrain=TerrainSpec(
                kind="procedural",
                size=(2.0, 2.0),
                procedural_params={"generator": "stairs", "num_steps": 5, "resolution": 32},
            )
        )
        builder = MjcfSceneBuilder("<mujoco><worldbody/></mujoco>")
        builder.attach_world(world)
        xml = builder.emit()
        self.assertIn('type="hfield"', xml)
        self.assertIn("robodeploy_terrain_", xml)

    def test_gazebo_ridge_procedural_terrain(self):
        ir = SceneIR(
            props=(),
            terrain=UnifiedTerrain(
                kind="procedural",
                size=(2.0, 2.0),
                procedural_params={"generator": "ridge", "seed": 2, "resolution": 32},
            ),
        )
        sdf = GazeboSceneBuilder().from_ir(ir)
        self.assertIn("<heightmap>", sdf)

    def test_prop_collision_masks_round_trip_and_mjcf(self):
        prop = PropConfig(
            name="cube",
            geom=GeomSpec(kind="box", size=(0.02, 0.02, 0.02)),
            collision_layer=2,
            collision_mask=0x0F,
        )
        ir_prop = world_to_ir(WorldSpec(props=[prop])).props[0]
        self.assertEqual(ir_prop.physics.collision_layer, 2)
        self.assertEqual(ir_prop.physics.collision_mask, 0x0F)

        builder = MjcfSceneBuilder("<mujoco><worldbody/></mujoco>")
        builder.attach_world(WorldSpec(props=[prop]))
        xml = builder.emit()
        self.assertIn('contype="4"', xml)
        self.assertIn('conaffinity="15"', xml)

    def test_camera_and_lighting_presets_in_scene_builder(self):
        spec = (
            SceneBuilder()
            .add_box("block", size=(0.03, 0.03, 0.03), pos=(0.5, 0.0, 0.4))
            .set_lighting("bright")
            .set_cameras("overview")
            .build_spec()
        )
        self.assertEqual(spec.lighting, "bright")
        self.assertEqual(len(spec.world.lights), 2)
        self.assertEqual(spec.world.cameras[0].name, "overview")
        self.assertEqual(len(get_camera_preset("tabletop")), 1)

    def test_task_choreography_pour_yaml(self):
        path = REPO_ROOT / "examples/tasks/choreography/pour.yaml"
        choreo = TaskChoreography.from_yaml(path)
        self.assertEqual(len(choreo.phases), 3)
        self.assertEqual(choreo.phases[0].kind, "reach")
        self.assertEqual(choreo.phases[2].params["predicate"], "liquid_in_target")

    def test_task_choreography_insertion_yaml(self):
        path = REPO_ROOT / "examples/tasks/choreography/insertion.yaml"
        choreo = TaskChoreography.from_yaml(path)
        self.assertEqual([p.kind for p in choreo.phases], ["reach", "align", "insert"])

    def test_success_predicates_liquid_and_peg(self):
        class _Obs:
            ee_position = (0.0, 0.0, 0.0)
            objects = {
                "cup": ((0.55, 0.15, 0.08), (1.0, 0.0, 0.0, 0.0)),
                "target": ((0.56, 0.15, 0.08), (1.0, 0.0, 0.0, 0.0)),
                "peg": ((0.6, 0.0, 0.38), (1.0, 0.0, 0.0, 0.0)),
                "hole": ((0.6, 0.0, 0.38), (1.0, 0.0, 0.0, 0.0)),
            }

        obs = _Obs()
        self.assertTrue(liquid_in_target(obs, source="cup", target="target", threshold=0.06))
        self.assertTrue(peg_in_hole(obs, peg="peg", hole="hole", threshold=0.03))

    def test_preset_inheritance_kuka_pick_mujoco(self):
        preset = load_example_preset("kuka_pick_mujoco")
        self.assertEqual(preset["robot"], "kuka")
        self.assertEqual(preset["backend"], "mujoco")
        self.assertEqual(preset["task"], "pick_place")
        self.assertEqual(preset["policy"], "example_sensor_reach_pick")
        self.assertIn("sensor_rigs", preset)
        self.assertNotIn("base_kuka_mujoco", list_example_presets())
        self.assertNotIn("_base_kuka_mujoco", list_example_presets())

    def test_preset_include_fragments_exist(self):
        for name in ("base_sim.yaml", "base_real.yaml", "manipulate.yaml"):
            self.assertTrue((REPO_ROOT / "examples/presets" / name).is_file())

    def test_pick_place_scene_ir_mujoco_gazebo_pose_tolerance(self):
        from examples.scenes.pick_table import build_pick_place_scene

        ir = build_pick_place_scene().to_ir()
        mjcf = MjcfSceneBuilder(
            '<mujoco><worldbody><body name="robot0"><joint name="j1"/></body></worldbody></mujoco>'
        ).from_ir(ir).emit()
        sdf = GazeboSceneBuilder().from_ir(ir)
        assert_cross_backend_pose_equivalence(ir, mjcf=mjcf, sdf=sdf, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
