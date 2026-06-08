from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from robodeploy.backends.real.ros2.sensors.tf_extrinsics import (
    camera_info_to_intrinsics,
    compose_extrinsics,
    extrinsics_dict,
    lookup_camera_extrinsics,
    quat_multiply_wxyz,
    quat_rotate_wxyz,
)


class TfExtrinsicsTests(unittest.TestCase):
    def test_quat_compose_and_rotate(self):
        parent_pos = (1.0, 0.0, 0.0)
        parent_quat = (1.0, 0.0, 0.0, 0.0)
        child_pos = (0.0, 0.0, 0.5)
        child_quat = (1.0, 0.0, 0.0, 0.0)
        pos, quat = compose_extrinsics(parent_pos, parent_quat, child_pos, child_quat)
        self.assertEqual(pos, (1.0, 0.0, 0.5))
        self.assertEqual(quat, (1.0, 0.0, 0.0, 0.0))
        rotated = quat_rotate_wxyz((1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
        self.assertAlmostEqual(rotated[0], 0.1, places=5)

    def test_quat_multiply_identity(self):
        q = (0.70710678, 0.0, 0.70710678, 0.0)
        out = quat_multiply_wxyz(q, (1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(out[0], q[0], places=5)

    def test_camera_info_to_intrinsics(self):
        msg = SimpleNamespace(
            width=640,
            height=480,
            k=[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
        )
        intrinsics = camera_info_to_intrinsics(msg)
        self.assertIsNotNone(intrinsics)
        assert intrinsics is not None
        self.assertEqual(intrinsics["fx"], 500.0)
        self.assertEqual(intrinsics["cx"], 320.0)

    def test_lookup_camera_extrinsics_mocked_tf(self):
        tf_stamped = SimpleNamespace(
            child_frame_id="wrist_camera",
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.1, y=0.2, z=0.3),
                rotation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        )
        buffer = mock.Mock()
        buffer.lookup_transform.return_value = tf_stamped
        out = lookup_camera_extrinsics(buffer, "world", "wrist_camera", stamp=object())
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["position"], (0.1, 0.2, 0.3))
        self.assertEqual(out["source"], "tf")
        self.assertEqual(out["parent_link"], "world")

    def test_extrinsics_dict_shape(self):
        out = extrinsics_dict((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), frame_id="cam", parent_link="ee")
        self.assertEqual(out["frame_id"], "cam")
        self.assertEqual(out["parent_link"], "ee")


if __name__ == "__main__":
    unittest.main()
