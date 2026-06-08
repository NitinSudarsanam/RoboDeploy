"""Calibration and live-path helpers testable without hardware."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robodeploy.calibration import cli as cal_cli
from robodeploy.calibration.base import CameraIntrinsics
from robodeploy.calibration.extrinsic.aruco import ArUcoExtrinsicCalibrator
from robodeploy.calibration.extrinsic.tf_lookup import TfExtrinsicLookup
from robodeploy.calibration.store import CalibrationStore
from robodeploy.sim2real.calibration import (
    load_calibration_template,
    seed_calibration_artifacts,
    validate_calibration_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class CalibrationCliTests(unittest.TestCase):
    def test_cmd_calibrate_kinematic_so101_dry_run(self):
        result = cal_cli.cmd_calibrate_kinematic(robot="so101", port="/dev/ttyACM0", as_json=True)
        self.assertEqual(result["robot"], "so101")
        self.assertIn("message", result)

    def test_cmd_calibrate_extrinsic_checkerboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os

            os.environ["ROBODEPLOY_CALIBRATION_ROOT"] = tmp
            result = cal_cli.cmd_calibrate_extrinsic(
                camera="wrist",
                pattern="checkerboard",
                board="7x5x0.025",
                robot_id="franka",
                as_json=True,
            )
            self.assertEqual(result["camera"], "wrist")
            self.assertIn("pose", result)
            self.assertTrue(Path(result["path"]).is_file())

    def test_cmd_calibrate_handeye_aruco(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os

            os.environ["ROBODEPLOY_CALIBRATION_ROOT"] = tmp
            result = cal_cli.cmd_calibrate_handeye(
                robot="franka",
                pattern="aruco",
                method="park",
                as_json=True,
            )
            self.assertIn("T_camera_to_ee", result)
            if "path" in result:
                self.assertTrue(Path(result["path"]).is_file())

    def test_cmd_calibrate_system_id_dummy(self):
        result = cal_cli.cmd_calibrate_system_id(
            robot="franka",
            joint=0,
            dummy=True,
            as_json=True,
        )
        self.assertEqual(result["robot"], "franka")
        self.assertIn("system_id", result)

    def test_parse_board_spec(self):
        size, square = cal_cli.parse_board_spec("7x5x0.025")
        self.assertEqual(size, (7, 5))
        self.assertAlmostEqual(square, 0.025)


class ArUcoExtrinsicTests(unittest.TestCase):
    def test_fit_with_mocked_detection(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv not installed")

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        corners = np.array(
            [[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]],
            dtype=np.float32,
        )
        ids = np.array([[0]], dtype=np.int32)
        rvecs = np.zeros((1, 1, 3), dtype=np.float64)
        tvecs = np.array([[[0.0, 0.0, 0.5]]], dtype=np.float64)

        calibrator = ArUcoExtrinsicCalibrator(dictionary="DICT_4X4_50", marker_size_m=0.05)
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

        with mock.patch.object(cv2.aruco.ArucoDetector, "detectMarkers", return_value=(corners, ids, None)):
            with mock.patch(
                "robodeploy.calibration.extrinsic.aruco._estimate_marker_poses",
                return_value=(rvecs, tvecs),
            ):
                poses = calibrator.fit([img], intrinsics)

        self.assertIn(0, poses)
        self.assertAlmostEqual(poses[0].position[2], 0.5, places=2)

    def test_estimate_marker_poses_solvepnp_fallback(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv not installed")

        from robodeploy.calibration.extrinsic.aruco import _estimate_marker_poses

        corners = np.array(
            [[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]],
            dtype=np.float32,
        )
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        rvecs, tvecs = _estimate_marker_poses(
            cv2,
            corners,
            marker_size_m=0.05,
            camera_matrix=K,
            dist_coeffs=dist,
        )
        self.assertEqual(rvecs.shape[0], 1)
        self.assertEqual(tvecs.shape[0], 1)


class TfExtrinsicLookupTests(unittest.TestCase):
    def test_lookup_wraps_tf_buffer(self):
        tf_stamped = SimpleNamespace(
            child_frame_id="wrist_camera",
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.1, y=0.2, z=0.3),
                rotation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        )
        buffer = mock.Mock()
        buffer.lookup_transform.return_value = tf_stamped
        lookup = TfExtrinsicLookup()
        pose = lookup.lookup(
            buffer,
            target_frame="world",
            source_frame="wrist_camera",
            stamp=object(),
        )
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose.position, (0.1, 0.2, 0.3))

    def test_intrinsics_from_camera_info(self):
        msg = SimpleNamespace(
            width=640,
            height=480,
            k=[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
        )
        lookup = TfExtrinsicLookup()
        intrinsics = lookup.intrinsics_from_camera_info(msg)
        self.assertIsNotNone(intrinsics)
        assert intrinsics is not None
        self.assertEqual(intrinsics.fx, 500.0)

    def test_to_store_payload_marks_tf_source(self):
        from robodeploy.core.types import Pose3D

        lookup = TfExtrinsicLookup()
        payload = lookup.to_store_payload(
            Pose3D(position=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0)),
            frame_id="cam",
            parent_link="ee",
        )
        self.assertEqual(payload["source"], "tf_lookup")


class CalibrationTemplateTests(unittest.TestCase):
    def test_load_reach_template(self):
        path = REPO_ROOT / "benchmarks" / "sim2real" / "reach_to_target" / "calibration_template.json"
        template = load_calibration_template(path)
        self.assertEqual(template.robot_id, "default")
        self.assertIn("kinematic", template.artifacts)

    def test_seed_and_validate_round_trip(self):
        path = REPO_ROOT / "benchmarks" / "sim2real" / "reach_to_target" / "calibration_template.json"
        template = load_calibration_template(path)
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(root=Path(tmp))
            missing_before = validate_calibration_artifacts(template, store)
            self.assertEqual(set(missing_before), {"kinematic", "extrinsic", "system_id"})
            seed_calibration_artifacts(template, store)
            missing_after = validate_calibration_artifacts(template, store)
            self.assertEqual(missing_after, [])

    def test_peg_template_includes_ft_calibration(self):
        path = REPO_ROOT / "benchmarks" / "sim2real" / "peg_insert" / "calibration_template.json"
        template = load_calibration_template(path)
        self.assertIn("ft_calibration", template.artifacts)


if __name__ == "__main__":
    unittest.main()
