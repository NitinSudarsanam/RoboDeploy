from __future__ import annotations

import struct
import unittest
from unittest import mock

import numpy as np

from robodeploy.sensors.ft_sensor.real.ati_ft import ATIFTSensor


def _netft_packet(*, status: int = 0, forces: tuple[int, int, int, int, int, int] = (0,) * 6) -> bytes:
    return struct.pack("!HHII", 0x1234, 2, 0, status) + struct.pack("!6i", *forces)


class AtiFtOverflowTests(unittest.TestCase):
    def test_rejects_invalid_header(self):
        sensor = ATIFTSensor(config={"host": "127.0.0.1"})
        sensor._sock = mock.Mock()
        sensor._sock.recvfrom.return_value = (b"\x00" * 36, None)
        sensor._scale = 1.0
        with self.assertRaises(RuntimeError) as ctx:
            sensor._read_impl()
        self.assertIn("header", str(ctx.exception).lower())

    def test_rejects_overflow_status(self):
        sensor = ATIFTSensor(config={"host": "127.0.0.1"})
        sensor._sock = mock.Mock()
        sensor._sock.recvfrom.return_value = (_netft_packet(status=0x01, forces=(1_000_000, 0, 0, 0, 0, 0)), None)
        sensor._scale = 1.0 / 1_000_000.0
        with self.assertRaises(RuntimeError) as ctx:
            sensor._read_impl()
        self.assertIn("overflow", str(ctx.exception).lower())

    def test_valid_packet_decodes_wrench(self):
        sensor = ATIFTSensor(config={"host": "127.0.0.1"})
        sensor._sock = mock.Mock()
        sensor._sock.recvfrom.return_value = (_netft_packet(forces=(2_000_000, 0, 0, 0, 0, 0)), None)
        sensor._scale = 1.0 / 1_000_000.0
        reading = sensor._read_impl()
        np.testing.assert_allclose(reading.ft_force, [2.0, 0.0, 0.0], rtol=1e-4)
        self.assertEqual(reading.status, "ok")


if __name__ == "__main__":
    unittest.main()
