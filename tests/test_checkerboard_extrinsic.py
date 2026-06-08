from __future__ import annotations

import unittest

import numpy as np

from robodeploy.calibration.base import CameraIntrinsics
from robodeploy.calibration.extrinsic.checkerboard import CheckerboardExtrinsicCalibrator
from robodeploy.calibration.extrinsic.handeye import HandEyeCalibrator
from robodeploy.core.types import Pose3D


class CheckerboardExtrinsicTests(unittest.TestCase):
    def test_fit_from_synthetic_samples(self):
        calibrator = CheckerboardExtrinsicCalibrator(board_size=(7, 5), square_size_m=0.025)
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        from robodeploy.calibration.extrinsic.checkerboard import CheckerboardSample

        samples = [
            CheckerboardSample(
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                rvec=np.zeros(3),
                tvec=np.array([0.0, 0.0, 0.5]),
                corners=np.zeros((35, 1, 2)),
            )
            for _ in range(3)
        ]
        pose = calibrator.fit(samples, intrinsics)
        self.assertAlmostEqual(pose.position[2], 0.5, places=2)

    def test_handeye_synthetic_poses(self):
        calibrator = HandEyeCalibrator()
        robot_poses = [
            Pose3D(position=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0)),
            Pose3D(position=(0.1, 0.0, 0.0), orientation=(0.996, 0.0, 0.087, 0.0)),
            Pose3D(position=(0.0, 0.1, 0.0), orientation=(0.996, 0.087, 0.0, 0.0)),
        ]
        marker_poses = [
            Pose3D(position=(0.0, 0.0, 0.3), orientation=(1.0, 0.0, 0.0, 0.0)),
            Pose3D(position=(0.05, 0.0, 0.3), orientation=(0.996, 0.0, 0.087, 0.0)),
            Pose3D(position=(0.0, 0.05, 0.3), orientation=(0.996, 0.087, 0.0, 0.0)),
        ]
        try:
            T = calibrator.fit(robot_poses, marker_poses, method="park")
            self.assertEqual(len(T.orientation), 4)
        except ImportError:
            self.skipTest("opencv not installed")


if __name__ == "__main__":
    unittest.main()
