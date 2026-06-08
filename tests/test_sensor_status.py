from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.backends.base import BackendBase
from robodeploy.core.types import Observation, SensorData
from robodeploy.sensors.base import SensorBase


class _OkSensor(SensorBase):
    def _init_impl(self, backend) -> None:
        return

    def _read_impl(self) -> SensorData:
        return SensorData(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            status="ok",
            timestamp=0.0,
            timestamp_hw=0.0,
        )

    def _close_impl(self) -> None:
        return


class _ErrorSensor(SensorBase):
    def _init_impl(self, backend) -> None:
        return

    def read(self) -> SensorData:
        raise RuntimeError("read failed")

    def _read_impl(self) -> SensorData:
        raise RuntimeError("read failed")

    def _close_impl(self) -> None:
        return


def _base_obs() -> Observation:
    return Observation(
        joint_positions=jnp.zeros((2,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
        joint_torques=jnp.zeros((2,), dtype=jnp.float32),
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        timestamp=0.0,
        timestamp_hw=0.0,
    )


class SensorStatusTests(unittest.TestCase):
    def test_merge_records_ok_and_error_status(self):
        backend = mock.Mock(spec=BackendBase)
        backend.config = {"sensor_read_policy": "warn"}
        backend._record_sensor_error = BackendBase._record_sensor_error.__get__(backend, BackendBase)
        backend._sensor_error_warned = set()
        backend._sensor_errors = {}
        backend._pending_sensor_reads = []
        ok = _OkSensor(name="wrist_camera", is_real=False)
        bad = _ErrorSensor(name="wrist_ft", is_real=False)
        ok._initialized = True
        bad._initialized = True
        merged = BackendBase._merge_sensor_data(backend, _base_obs(), [ok, bad])
        self.assertEqual(merged.sensor_status.get("wrist_camera"), "ok")
        self.assertEqual(merged.sensor_status.get("wrist_ft"), "error")
        self.assertIn("wrist_camera", merged.images)

    def test_stale_status_propagates_through_buffer(self):
        from robodeploy.obs_pipeline import ObsPipeline

        pipeline = ObsPipeline(sync_window_s=0.1)
        pipeline.buffer_sensor(
            "wrist_ft",
            SensorData(
                ft_force=np.ones(3, dtype=np.float32),
                status="stale",
                timestamp_hw=0.0,
                timestamp=0.0,
            ),
        )
        merged = pipeline.process(_base_obs())
        self.assertEqual(merged.sensor_status.get("wrist_ft"), "stale")


if __name__ == "__main__":
    unittest.main()
