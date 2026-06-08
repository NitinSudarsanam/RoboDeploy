from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

import numpy as np

from robodeploy.backends.real.ros2.backend import ROS2RealBackend
from robodeploy.backends.real.ros2.controllers.base import CommandAck, ControllerConfig
from robodeploy.backends.real.ros2.controllers.joint_position import JointPositionControllerAdapter
from robodeploy.core.types import Action
from robodeploy.safety import Hazard


class CommandAckTests(unittest.TestCase):
    def test_ack_defaults_unacked(self):
        ack = CommandAck(published_at=time.time(), expected_ack_within_s=0.01, sequence_id=1)
        self.assertFalse(ack.acked)

    def test_timed_out_after_window(self):
        ack = CommandAck(published_at=time.time() - 1.0, expected_ack_within_s=0.01, sequence_id=1)
        self.assertTrue(ack.timed_out)

    def test_mark_received(self):
        now = time.time()
        ack = CommandAck(published_at=now, expected_ack_within_s=0.5, sequence_id=7)
        ack.received_at = now + 0.01
        self.assertTrue(ack.acked)
        self.assertFalse(ack.timed_out)


class CommandAckE2ETests(unittest.TestCase):
    def _make_adapter(self, *, ack_timeout_s: float = 0.05) -> JointPositionControllerAdapter:
        cfg = ControllerConfig(
            robot_id="robot0",
            joint_names=["j0", "j1"],
            ack_timeout_s=ack_timeout_s,
        )
        adapter = JointPositionControllerAdapter(cfg)
        adapter._joint_names = ["j0", "j1"]  # noqa: SLF001
        adapter._ensure_buffers(2)  # noqa: SLF001
        adapter._has_joint_state = True
        adapter._last_joint_state_wall_s = time.time()  # noqa: SLF001
        adapter._commander._send_fn = lambda _payload: None  # noqa: SLF001
        return adapter

    def test_send_action_returns_ack_and_joint_state_clears_timeout(self):
        adapter = self._make_adapter()
        action = Action(joint_positions=np.array([0.1, 0.2], dtype=np.float32))
        ack = adapter.send_action(action)
        self.assertIsNotNone(ack)
        self.assertEqual(ack.sequence_id, 1)
        self.assertFalse(ack.acked)

        msg = SimpleNamespace(
            name=["j0", "j1"],
            position=[0.1, 0.2],
            velocity=[],
            effort=[],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
        )
        adapter._on_joint_state(msg)

        self.assertTrue(ack.acked)
        self.assertFalse(ack.timed_out)
        self.assertEqual(adapter.get_diagnostics()["pending_ack_timeouts"], 0)

    def test_command_ack_timeout_detected(self):
        adapter = self._make_adapter(ack_timeout_s=0.001)
        action = Action(joint_positions=np.array([0.1, 0.2], dtype=np.float32))
        ack = adapter.send_action(action)
        self.assertIsNotNone(ack)
        time.sleep(0.01)
        self.assertTrue(ack.timed_out)
        self.assertEqual(adapter.get_diagnostics()["pending_ack_timeouts"], 1)

    def test_backend_surfaces_command_rejected_hazard(self):
        adapter = self._make_adapter(ack_timeout_s=0.001)
        adapter.send_action(Action(joint_positions=np.array([0.1, 0.2], dtype=np.float32)))
        time.sleep(0.01)
        adapter._last_joint_state_wall_s = time.time()  # noqa: SLF001

        backend = ROS2RealBackend(config={})
        backend._drivers = {"robot0": adapter}
        backend._diagnostics = {"backend": "ros2", "robots": {}, "warnings": []}
        backend._recovery_managers = {}
        backend._capture_driver_diagnostics("robot0")

        self.assertEqual(
            backend._diagnostics.get("safety", {}).get("last_hazard"),
            Hazard.COMMAND_REJECTED.name,
        )
        warnings = backend._diagnostics.get("warnings", [])
        self.assertTrue(any("without joint-state ack" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
