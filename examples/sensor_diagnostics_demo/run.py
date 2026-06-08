"""Print obs.sensor_status while one sensor read fails."""

from __future__ import annotations

from examples._bootstrap import ensure_repo_on_path

ensure_repo_on_path()
from pathlib import Path
from unittest import mock

import numpy as np



from robodeploy.backends.base import BackendBase  # noqa: E402
from robodeploy.core.types import Observation, SensorData  # noqa: E402
from robodeploy.sensors.base import SensorBase  # noqa: E402


class _FailingSensor(SensorBase):
    def _init_impl(self, backend) -> None:
        return

    def _read_impl(self) -> SensorData:
        raise RuntimeError("simulated sensor fault")

    def _close_impl(self) -> None:
        return


class _OkSensor(SensorBase):
    def _init_impl(self, backend) -> None:
        return

    def _read_impl(self) -> SensorData:
        return SensorData(
            ft_force=np.asarray([0.1, 0.0, -0.5], dtype=np.float32),
            status="ok",
            timestamp=0.0,
            timestamp_hw=0.0,
        )

    def _close_impl(self) -> None:
        return


def main() -> None:
    try:
        import jax.numpy as jnp
    except Exception:
        import numpy as jnp  # type: ignore[assignment]

    backend = mock.Mock(spec=BackendBase)
    backend.config = {"sensor_read_policy": "warn"}
    backend._record_sensor_error = BackendBase._record_sensor_error.__get__(backend, BackendBase)
    backend._sensor_error_warned = set()
    backend._pending_sensor_reads = []

    obs = Observation(
        joint_positions=jnp.zeros((2,), dtype=jnp.float32),
        joint_velocities=jnp.zeros((2,), dtype=jnp.float32),
        joint_torques=jnp.zeros((2,), dtype=jnp.float32),
        ee_position=jnp.zeros((3,), dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ee_angular_velocity=jnp.zeros((3,), dtype=jnp.float32),
    )
    merged = BackendBase._merge_sensor_data(
        backend,
        obs,
        [_OkSensor(name="wrist_ft", is_real=False), _FailingSensor(name="wrist_camera", is_real=False)],
    )
    print("sensor_status:", merged.sensor_status)
    print("ft_forces keys:", list(merged.ft_forces.keys()))
    print("images keys:", list(merged.images.keys()))


if __name__ == "__main__":
    main()
