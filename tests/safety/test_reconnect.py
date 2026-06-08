from __future__ import annotations

import unittest

from robodeploy.backends.real.ros2.recovery import ROS2RecoveryManager
from robodeploy.safety import Hazard, SafetyError


class ROS2RecoveryTests(unittest.TestCase):
    def test_reconnect_succeeds_within_retries(self):
        calls = {"n": 0}

        def reconnect() -> bool:
            calls["n"] += 1
            return calls["n"] >= 2

        mgr = ROS2RecoveryManager(
            reconnect_fn=reconnect,
            state_timeout_s=0.1,
            max_retries=5,
            initial_backoff_s=0.0,
            sleep_fn=lambda _s: None,
        )
        mgr.on_state_stale(1.0)
        self.assertFalse(mgr.connection_lost)
        self.assertEqual(calls["n"], 2)

    def test_reconnect_failure_raises_connection_lost(self):
        mgr = ROS2RecoveryManager(
            reconnect_fn=lambda: False,
            state_timeout_s=0.1,
            max_retries=3,
            initial_backoff_s=0.0,
            sleep_fn=lambda _s: None,
        )
        with self.assertRaises(SafetyError) as ctx:
            mgr.on_state_stale(1.0)
        self.assertEqual(ctx.exception.violation.hazard, Hazard.CONNECTION_LOST)


if __name__ == "__main__":
    unittest.main()
