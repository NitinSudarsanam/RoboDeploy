from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from robodeploy.core.registry import resolve_sensor_class
from robodeploy.sensors.imu.real.xsens import XsensIMUSensor, _decode_mtdata2_accel_gyro


def _hardware_available() -> bool:
    port = os.environ.get("ROBODEPLOY_XSENS_PORT", "").strip()
    if not port:
        return False
    try:
        import serial  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        ser.close()
        return True
    except Exception:
        return False


@unittest.skipUnless(_hardware_available(), "Set ROBODEPLOY_XSENS_PORT to a live Xsens serial device")
class LiveRealImuTests(unittest.TestCase):
    def test_live_xsens_read(self):
        port = os.environ["ROBODEPLOY_XSENS_PORT"]
        sensor = XsensIMUSensor(config={"name": "wrist_imu", "port": port})
        sensor.initialize(mock.Mock())
        reading = sensor.read()
        self.assertIsNotNone(reading.imu_acceleration)
        self.assertIsNotNone(reading.imu_angular_velocity)
        accel = np.asarray(reading.imu_acceleration, dtype=np.float32)
        self.assertEqual(accel.shape, (3,))
        sensor.close()


class RealImuStubTests(unittest.TestCase):
    def test_resolve_real_wrist_imu(self):
        self.assertIs(
            resolve_sensor_class("wrist_imu", is_real=True, backend_name=None),
            XsensIMUSensor,
        )

    def test_mtdata2_decode_roundtrip(self):
        payload = np.array([0.0, 0.0, 9.81, 0.1, 0.0, 0.0], dtype=np.float32).tobytes()
        decoded = _decode_mtdata2_accel_gyro(payload)
        self.assertIsNotNone(decoded)
        accel, gyro = decoded
        np.testing.assert_allclose(accel, [0.0, 0.0, 9.81], rtol=1e-3)
        np.testing.assert_allclose(gyro, [0.1, 0.0, 0.0], rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
