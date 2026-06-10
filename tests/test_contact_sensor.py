from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from robodeploy.backends.base import BackendBase
from robodeploy.core.registry import resolve_sensor_class
from robodeploy.core.types import Observation, SensorData
from robodeploy.obs_pipeline import ObsPipeline
from robodeploy.sensors.contact.sim.gazebo_contact import GazeboContactSensor
from robodeploy.sensors.contact.sim.mujoco_contact import MuJoCoContactSensor

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]


class ContactSensorTests(unittest.TestCase):
    def test_resolve_wrist_contact_mujoco(self):
        self.assertIs(
            resolve_sensor_class("wrist_contact", is_real=False, backend_name="mujoco"),
            MuJoCoContactSensor,
        )

    def test_resolve_wrist_contact_gazebo(self):
        self.assertIs(
            resolve_sensor_class("wrist_contact", is_real=False, backend_name="gazebo"),
            GazeboContactSensor,
        )

    def test_gazebo_contact_reads_monitor(self):
        sensor = GazeboContactSensor("wrist_contact", config={"prop_name": "source", "ee_link": "ee"})
        backend = mock.Mock()
        backend._ee_link = "robot0/ee_link"
        backend.has_prop_contact = mock.Mock(return_value=True)
        sensor.initialize(backend)
        reading = sensor.read()
        self.assertTrue(reading.contact_state.get("wrist_contact"))

    def test_gazebo_contact_binds_world_scoped_topic(self):
        sensor = GazeboContactSensor("wrist_contact", config={"prop_name": "source"})
        monitor = mock.Mock()
        monitor._subscriber = None
        monitor.bind_transport = mock.Mock()
        backend = mock.Mock()
        backend._contact_monitor = monitor
        backend._gz_transport_node = object()
        backend._gz_world_name = "robodeploy_world"
        sensor.initialize(backend)
        monitor.bind_transport.assert_called_once_with(
            backend._gz_transport_node,
            topic="/world/robodeploy_world/contacts",
        )

    def test_gazebo_contact_skips_rebind_when_monitor_ready(self):
        sensor = GazeboContactSensor("wrist_contact", config={"prop_name": "source"})
        monitor = mock.Mock()
        monitor._subscriber = object()
        monitor.bind_transport = mock.Mock()
        backend = mock.Mock()
        backend._contact_monitor = monitor
        backend._gz_transport_node = object()
        sensor.initialize(backend)
        monitor.bind_transport.assert_not_called()

    def test_mujoco_contact_reads_backend_state(self):
        sensor = MuJoCoContactSensor("wrist_contact", config={"prop_name": "source"})
        backend = mock.Mock()
        backend.has_prop_contact = mock.Mock(return_value=True)
        backend.prop_near_ee = mock.Mock(return_value=False)
        backend._data = mock.Mock(time=0.5)
        sensor.initialize(backend)
        reading = sensor.read()
        self.assertTrue(reading.contact_state.get("wrist_contact"))
        backend.has_prop_contact.assert_called_with("source")

    def test_contact_state_merged_into_observation(self):
        backend = mock.Mock(spec=BackendBase)
        backend.config = {"sensor_read_policy": "warn"}
        backend._record_sensor_error = BackendBase._record_sensor_error.__get__(backend, BackendBase)
        backend._sensor_error_warned = set()
        backend._sensor_errors = {}
        backend._pending_sensor_reads = []

        sensor = MuJoCoContactSensor("wrist_contact", config={"prop_name": "source"})
        sensor._backend = backend
        sensor._prop_name = "source"
        sensor._ee_dist = 0.04
        sensor._initialized = True
        sensor._read_impl = lambda: SensorData(
            contact_state={"wrist_contact": True},
            timestamp=0.0,
            timestamp_hw=0.0,
        )

        base = Observation(
            joint_positions=jnp.zeros((2,), dtype=jnp.float32),
            joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
            joint_torques=jnp.zeros((2,), dtype=jnp.float32),
            ee_position=jnp.zeros((3,), dtype=jnp.float32),
            ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
            ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        )
        merged = BackendBase._merge_sensor_data(backend, base, [sensor])
        self.assertTrue(merged.contact_state.get("wrist_contact"))

    def test_contact_state_propagates_through_pipeline_buffer(self):
        pipeline = ObsPipeline(sync_window_s=0.1)
        pipeline.buffer_sensor(
            "wrist_contact",
            SensorData(
                contact_state={"wrist_contact": True},
                timestamp_hw=0.0,
                timestamp=0.0,
            ),
        )
        base = Observation(
            joint_positions=jnp.zeros((2,), dtype=jnp.float32),
            joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
            joint_torques=jnp.zeros((2,), dtype=jnp.float32),
            ee_position=jnp.zeros((3,), dtype=jnp.float32),
            ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
            ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
        )
        merged = pipeline.process(base)
        self.assertTrue(merged.contact_state.get("wrist_contact"))


if __name__ == "__main__":
    unittest.main()
