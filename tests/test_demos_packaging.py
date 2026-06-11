"""Packaging tests for robodeploy.demos benchmark dependencies."""

from __future__ import annotations

import unittest


class DemosPackagingTests(unittest.TestCase):
    def test_benchmark_task_imports_from_demos(self):
        from robodeploy.demos.tasks import (
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
        from robodeploy.demos.policies.sensor_reach_pick import SensorReachPickPlacePolicy
        from robodeploy.demos.policies.joint_track import JointTrackPolicy
        from robodeploy.demos.sensors.prop_pose import SimPropPoseSensor

        self.assertEqual(SensorReachPickPlacePolicy.__name__, "SensorReachPickPlacePolicy")
        self.assertEqual(JointTrackPolicy.__name__, "JointTrackPolicy")
        self.assertEqual(SimPropPoseSensor.__name__, "SimPropPoseSensor")

    def test_reach_pick_place_yaml_packaged(self):
        from importlib.resources import files

        yaml_path = files("robodeploy.demos.policies").joinpath("reach_pick_place.yaml")
        self.assertTrue(yaml_path.is_file(), "reach_pick_place.yaml must ship in wheel")
