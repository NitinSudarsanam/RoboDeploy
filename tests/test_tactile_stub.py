from __future__ import annotations

import unittest
from unittest import mock

from robodeploy.core.registry import resolve_sensor_class
from robodeploy.sensors.tactile.stub import TactileArrayStubSensor


@unittest.skip("Tactile pressure-array hardware deferred — stub only for API parity")
class TactileHardwareDeferredTests(unittest.TestCase):
    """Placeholder for future tactile-array hardware integration."""

    def test_hardware_not_integrated(self):
        self.fail("Tactile array driver not yet available.")


class TactileStubTests(unittest.TestCase):
    def test_resolve_tactile_array_pair(self):
        cls = resolve_sensor_class("tactile_array", is_real=False, backend_name="mujoco")
        self.assertIs(cls, TactileArrayStubSensor)

    def test_stub_returns_deferred_status(self):
        sensor = TactileArrayStubSensor(config={"name": "tactile_array", "rows": 2, "cols": 2})
        sensor.initialize(mock.Mock())
        reading = sensor.read()
        self.assertEqual(reading.status, "deferred")
        self.assertIn("not integrated", sensor.deferred_reason.lower())


if __name__ == "__main__":
    unittest.main()
