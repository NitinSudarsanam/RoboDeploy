from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp
except Exception:
    import numpy as jnp  # type: ignore[assignment]

from robodeploy.core.types import Action, Observation
from robodeploy.safety import EStop, SafetyError, SafetyMonitor


def _obs() -> Observation:
    return Observation(
        joint_positions=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_velocities=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        joint_torques=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        ee_position=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_orientation=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
        ee_angular_velocity=jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32),
    )


class EStopTests(unittest.TestCase):
    def test_trip_raises_on_check(self):
        estop = EStop(signal_handlers=False)
        estop.trip("test")
        with self.assertRaises(SafetyError) as ctx:
            estop.check()
        self.assertIn("test", str(ctx.exception))

    def test_reset_clears_trip(self):
        estop = EStop(signal_handlers=False)
        estop.trip("test")
        estop.reset()
        estop.check()

    def test_monitor_halts_on_tripped_estop(self):
        monitor = SafetyMonitor(estop=EStop(signal_handlers=False))
        monitor.estop.trip("operator")
        with self.assertRaises(SafetyError):
            monitor.check_action(Action(), _obs(), dt=0.05)


if __name__ == "__main__":
    unittest.main()
