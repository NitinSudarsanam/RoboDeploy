"""Packaging tests for examples benchmark dependencies."""

from __future__ import annotations

import unittest


class DemosPackagingTests(unittest.TestCase):
    def test_benchmark_task_imports_from_demos(self):
        from examples.tasks import (
            PegInsertionTask,
            PegTask,
            PickPlaceTask,
            PourTask,
            ShowcaseSceneTask,
        )

        self.assertEqual(PickPlaceTask.__name__, "PickPlaceTask")
        self.assertIs(PegInsertionTask, PegTask)
        self.assertEqual(PourTask.__name__, "PourTask")
        self.assertEqual(ShowcaseSceneTask.__name__, "ShowcaseSceneTask")

    def test_benchmark_policy_and_sensor_imports_from_demos(self):
        from examples.policies.sensor_reach_pick import SensorReachPickPlacePolicy
        from examples.policies.joint_track import JointTrackPolicy
        from robodeploy.sensors.pose.sim.prop_pose import SimPropPoseSensor
        from robodeploy.sensors.pose.sim.ee_pose import EePoseSensor

        self.assertEqual(SensorReachPickPlacePolicy.__name__, "SensorReachPickPlacePolicy")
        self.assertEqual(JointTrackPolicy.__name__, "JointTrackPolicy")
        self.assertEqual(SimPropPoseSensor.__name__, "SimPropPoseSensor")
        self.assertEqual(EePoseSensor.__name__, "EePoseSensor")

    def test_pose_sensors_resolve_via_builtins_without_custom_modules(self):
        from robodeploy.builtins import import_builtins
        from robodeploy.core.registry import resolve_sensor_class

        import_builtins()
        prop_cls = resolve_sensor_class("sim_prop_pose", is_real=False, backend_name="mujoco")
        ee_cls = resolve_sensor_class("ee_pose", is_real=False, backend_name="mujoco")
        self.assertEqual(prop_cls.__name__, "SimPropPoseSensor")
        self.assertEqual(ee_cls.__name__, "EePoseSensor")

    def test_reach_pick_place_yaml_in_examples(self):
        from pathlib import Path

        yaml_path = Path(__file__).resolve().parents[1] / "examples" / "policies" / "reach_pick_place.yaml"
        self.assertTrue(yaml_path.is_file(), "reach_pick_place.yaml must live under examples/policies")
