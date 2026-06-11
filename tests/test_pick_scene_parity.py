"""Pick-place scene parity across MuJoCo, Gazebo, and RViz scene IR."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PickSceneParityTests(unittest.TestCase):
    def test_canonical_scene_has_table_source_target(self):
        from robodeploy.demos.scenes.pick_table import build_pick_place_scene

        names = {p.name for p in build_pick_place_scene().to_world().props}
        self.assertEqual(names, {"table", "source", "target"})

    def test_mujoco_gazebo_prop_poses_match_ir(self):
        from robodeploy.backends.sim.gazebo.scene_builder import GazeboSceneBuilder
        from robodeploy.backends.sim.mujoco.scene_builder import MjcfSceneBuilder
        from robodeploy.core.scene_ir import assert_cross_backend_pose_equivalence
        from robodeploy.demos.scenes.pick_table import build_pick_place_scene

        ir = build_pick_place_scene().to_ir()
        mjcf = MjcfSceneBuilder(
            '<mujoco><worldbody><body name="robot0"><joint name="j1"/></body></worldbody></mujoco>'
        ).from_ir(ir).emit()
        sdf = GazeboSceneBuilder().from_ir(ir)
        assert_cross_backend_pose_equivalence(ir, mjcf=mjcf, sdf=sdf, atol=1e-3)

    def test_kuka_mjcf_has_no_legacy_pick_geometry(self):
        from robodeploy.core.spaces import AssetFormat
        from robodeploy.description.kuka.description import KukaDescription

        text = KukaDescription().asset_path(AssetFormat.MJCF, variant="sim").read_text(encoding="utf-8")
        for legacy in ("pick_cube", "pick_target", 'name="table"'):
            self.assertNotIn(legacy, text, msg=f"legacy scene fragment still in kuka.xml: {legacy}")

    def test_kuka_rviz_extra_config_uses_world_frame(self):
        from robodeploy.description.kuka.description import KukaDescription

        extra = KukaDescription().ros2_rviz_extra_config("robot0")
        assert extra is not None
        self.assertEqual(extra["robot0.base_frame"], "world")
        self.assertTrue(extra["prefer_fk_ee_pose"])
        self.assertEqual(extra["rviz"]["fixed_frame"], "world")

    def test_ee_pose_sensor_prefers_world_fk_when_configured(self):
        import numpy as np

        from robodeploy.demos.sensors.ee_pose import EePoseSensor, _prefer_world_fk

        class _Backend:
            config = {"prefer_fk_ee_pose": True, "robot0.base_frame": "world"}
            _drivers = {}
            _latest_obs = {}

        self.assertTrue(_prefer_world_fk(_Backend()))

        sensor = EePoseSensor(name="ee_pose")
        sensor._backend = _Backend()
        sensor._prefer_mujoco_fk = False

        class _Fk:
            def fk_position(self, q):
                return np.array([0.55, 0.0, 0.6], dtype=np.float32)

            class _solver:
                @staticmethod
                def fk(q):
                    return np.array([0.55, 0.0, 0.6]), np.array([1.0, 0.0, 0.0, 0.0])

        sensor._fk = _Fk()
        sensor._backend._latest_obs = {
            "robot0": type("O", (), {"joint_positions": np.zeros(7)})(),
        }
        data = sensor._read_impl()
        self.assertEqual(getattr(data, "status", ""), "ok")
        self.assertAlmostEqual(float(data.ee_pose[2]), 0.6, places=3)

    def test_rviz_set_prop_pose_republishes_markers(self):
        from robodeploy.backends.real.ros2.backend import ROS2RvizBackend

        backend = ROS2RvizBackend.__new__(ROS2RvizBackend)
        backend._scene_prop_poses = {
            "source": ((0.55, 0.0, 0.405), (1.0, 0.0, 0.0, 0.0)),
        }
        backend._prop_marker_specs = {
            "source": {"kind": "box", "size": (0.025, 0.025, 0.025), "rgba": (1.0, 0.0, 0.0, 1.0)},
        }
        backend._scene_terrain_size = (4.0, 4.0)
        calls: list[dict] = []

        class _Rviz:
            def publish_prop_poses(self, poses, *, prop_specs=None, terrain_size=(4.0, 4.0), include_ground=True):
                calls.append({"poses": dict(poses), "prop_specs": prop_specs})

        backend._rviz = _Rviz()
        backend.set_prop_pose("source", (0.56, 0.01, 0.5), (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0]["poses"]["source"][0][0], 0.56, places=4)

    def test_place_snap_off_by_default_in_policy_config(self):
        from robodeploy.policies.reach_dsl import ReachTrajectoryPolicy

        class _Policy(ReachTrajectoryPolicy):
            def __init__(self):
                self.config = {"gazebo_place_snap": False}

        policy = _Policy()
        self.assertFalse(policy._gazebo_place_snap_enabled())


if __name__ == "__main__":
    unittest.main()
