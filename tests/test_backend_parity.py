from __future__ import annotations

import unittest

import numpy as np

from robodeploy.backends.capabilities import (
    SupportsContactQuery,
    SupportsGraspFollow,
    SupportsGraspWeld,
    SupportsPhysicsTuning,
)
from robodeploy.backends.sim.gazebo.contact import GazeboContactMonitor
from robodeploy.backends.sim.gazebo.scene_builder import GazeboSceneBuilder
from robodeploy.backends.sim.isaacsim.scene_builder import IsaacSceneBuilder
from robodeploy.backends.sim.mujoco.scene_builder import MjcfSceneBuilder
from robodeploy.core.scene_ir import (
    SceneIR,
    UnifiedGeom,
    UnifiedPhysics,
    UnifiedPropSpec,
    Pose3D,
    assert_cross_backend_pose_equivalence,
    count_gazebo_collision_geoms,
    count_mujoco_prop_bodies,
    extract_gazebo_prop_positions,
    extract_mujoco_prop_positions,
    ir_backend_geom_total,
    ir_logical_geom_total,
    ir_prop_count,
    ir_prop_positions,
    world_to_ir,
)
from robodeploy.core.types import GeomSpec, PropConfig, SceneSpec, WorldSpec


def _parity_scene_ir() -> SceneIR:
    return SceneIR(
        props=(
            UnifiedPropSpec(
                name="source",
                geometry=UnifiedGeom(kind="box", size=(0.04, 0.04, 0.04)),
                physics=UnifiedPhysics(mass=0.05, is_fixed=False),
                pose=Pose3D(position=(0.55, 0.0, 0.41)),
            ),
            UnifiedPropSpec(
                name="target",
                geometry=UnifiedGeom(kind="sphere", size=(0.03,)),
                physics=UnifiedPhysics(is_fixed=True),
                pose=Pose3D(position=(0.6, 0.2, 0.41)),
            ),
            UnifiedPropSpec(
                name="capsule_prop",
                geometry=UnifiedGeom(kind="capsule", size=(0.02, 0.08)),
                physics=UnifiedPhysics(mass=0.04),
                pose=Pose3D(position=(0.4, -0.1, 0.45)),
            ),
        ),
        gravity=(0.0, 0.0, -9.81),
    )


class BackendParityTests(unittest.TestCase):
    def test_scene_ir_round_trip_pose_tolerance(self):
        scene = SceneSpec(
            props=[
                PropConfig(name="source", geom=GeomSpec(kind="box", size=(0.04, 0.04, 0.04)), position=(0.55, 0.0, 0.41)),
                PropConfig(name="target", geom=GeomSpec(kind="sphere", size=(0.03,)), position=(0.6, 0.2, 0.41), is_fixed=True),
            ]
        )
        ir = scene.to_ir()
        rebuilt = SceneSpec(world=scene.to_world())
        rebuilt_ir = rebuilt.to_ir()
        for left, right in zip(ir.props, rebuilt_ir.props):
            self.assertEqual(left.name, right.name)
            np.testing.assert_allclose(left.pose.position, right.pose.position, atol=1e-3)
            np.testing.assert_allclose(left.pose.orientation, right.pose.orientation, atol=1e-3)

    def test_gazebo_builder_from_ir_prop_count_and_geoms(self):
        ir = _parity_scene_ir()
        sdf = GazeboSceneBuilder().from_ir(ir)
        self.assertEqual(sdf.count('model name="source"'), 1)
        self.assertEqual(sdf.count('model name="target"'), 1)
        self.assertEqual(sdf.count('model name="capsule_prop"'), 1)
        self.assertIn("<cylinder>", sdf)
        self.assertIn("cap_top", sdf)
        self.assertIn("cap_bot", sdf)

    def test_gazebo_procedural_terrain_emits_heightmap(self):
        from robodeploy.core.scene_ir import UnifiedTerrain

        ir = SceneIR(
            props=(),
            terrain=UnifiedTerrain(kind="procedural", size=(2.0, 2.0), procedural_params={"seed": 3, "resolution": 32}),
        )
        sdf = GazeboSceneBuilder().from_ir(ir)
        self.assertIn("<heightmap>", sdf)
        self.assertIn("robodeploy_terrain_", sdf)

    def test_mujoco_builder_from_ir_emits_props(self):
        ir = _parity_scene_ir()
        builder = MjcfSceneBuilder(
            '<mujoco><worldbody><body name="robot0"><joint name="j1"/></body></worldbody></mujoco>'
        )
        builder.from_ir(ir)
        xml = builder.emit()
        self.assertIn('body name="source"', xml)
        self.assertIn('body name="target"', xml)
        self.assertIn('body name="capsule_prop"', xml)

    def test_isaac_builder_from_ir_offline_warns_gracefully(self):
        builder = IsaacSceneBuilder()
        paths = builder.from_ir(_parity_scene_ir())
        self.assertEqual(paths, {})
        self.assertTrue(builder._warnings)

    def test_isaac_procedural_terrain_no_plane_fallback_warning(self):
        from robodeploy.core.scene_ir import UnifiedTerrain

        ir = SceneIR(
            props=(),
            terrain=UnifiedTerrain(kind="procedural", size=(2.0, 2.0), procedural_params={"seed": 1, "resolution": 16}),
        )
        builder = IsaacSceneBuilder()
        builder.from_ir(ir)
        joined = " ".join(builder._warnings)
        self.assertNotIn("approximated as a large plane", joined)

    def test_cross_backend_geom_count_equivalence(self):
        ir = _parity_scene_ir()
        prop_names = [p.name for p in ir.props]
        self.assertEqual(ir_prop_count(ir), 3)
        self.assertEqual(ir_logical_geom_total(ir), 3)

        mjcf = MjcfSceneBuilder(
            '<mujoco><worldbody><body name="robot0"><joint name="j1"/></body></worldbody></mujoco>'
        ).from_ir(ir).emit()
        mujoco_counts = count_mujoco_prop_bodies(mjcf, prop_names)
        self.assertEqual(sum(mujoco_counts.values()), ir_logical_geom_total(ir))

        sdf = GazeboSceneBuilder().from_ir(ir)
        gazebo_counts = count_gazebo_collision_geoms(sdf, prop_names)
        self.assertEqual(sum(gazebo_counts.values()), ir_backend_geom_total(ir, backend="gazebo"))
        self.assertEqual(gazebo_counts["capsule_prop"], 3)

        isaac_counts = IsaacSceneBuilder.planned_prop_geom_counts(ir)
        self.assertEqual(sum(isaac_counts.values()), ir_logical_geom_total(ir))

    def test_cross_backend_pose_equivalence(self):
        ir = _parity_scene_ir()
        prop_names = [p.name for p in ir.props]
        expected = ir_prop_positions(ir)

        mjcf = MjcfSceneBuilder(
            '<mujoco><worldbody><body name="robot0"><joint name="j1"/></body></worldbody></mujoco>'
        ).from_ir(ir).emit()
        sdf = GazeboSceneBuilder().from_ir(ir)

        mujoco_pos = extract_mujoco_prop_positions(mjcf, prop_names)
        gazebo_pos = extract_gazebo_prop_positions(sdf, prop_names)
        self.assertEqual(set(mujoco_pos.keys()), set(expected.keys()))
        self.assertEqual(set(gazebo_pos.keys()), set(expected.keys()))

        assert_cross_backend_pose_equivalence(ir, mjcf=mjcf, sdf=sdf, atol=1e-3)

    def test_capability_protocol_matrix(self):
        from robodeploy.backends.sim.mujoco.backend import MuJoCoBackend
        from robodeploy.backends.sim.isaacsim.backend import IsaacSimBackend
        from robodeploy.backends.sim.gazebo.backend import ROS2GazeboBackend

        self.assertTrue(isinstance(MuJoCoBackend({}), SupportsGraspWeld))
        self.assertTrue(isinstance(MuJoCoBackend({}), SupportsContactQuery))
        self.assertFalse(isinstance(ROS2GazeboBackend({}), SupportsGraspWeld))
        self.assertTrue(isinstance(ROS2GazeboBackend({}), SupportsGraspFollow))
        self.assertTrue(isinstance(ROS2GazeboBackend({}), SupportsContactQuery))
        self.assertFalse(isinstance(IsaacSimBackend({}), SupportsGraspWeld))
        self.assertTrue(isinstance(IsaacSimBackend({}), SupportsPhysicsTuning))

    def test_gazebo_contact_monitor_fixture_pairs(self):
        monitor = GazeboContactMonitor()
        monitor.inject_contacts([("source", "robot0/ee_link")])
        self.assertTrue(monitor.has_contact("source", "robot0/ee_link"))
        self.assertFalse(monitor.has_contact("target", "robot0/ee_link"))

    def test_world_to_ir_preserves_prop_count(self):
        world = WorldSpec(
            props=[
                PropConfig(name="a", geom=GeomSpec(kind="mesh", size=(), mesh_path="cube.obj")),
                PropConfig(name="b", geom=GeomSpec(kind="plane", size=(2.0, 2.0))),
            ]
        )
        ir = world_to_ir(world)
        self.assertEqual(len(ir.props), 2)
        self.assertEqual(ir.props[0].geometry.kind, "mesh")
        self.assertEqual(ir.props[1].geometry.kind, "plane")


if __name__ == "__main__":
    unittest.main()
