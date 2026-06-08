from __future__ import annotations

import unittest

from robodeploy.observability.health import HealthMonitor, summarize_sensor_health


class HealthMonitorTests(unittest.TestCase):
    def test_summarize_sensor_health(self):
        summary = summarize_sensor_health({"cam": "ok", "ft": "stale"})
        self.assertEqual(summary["overall"], "degraded")
        self.assertEqual(summary["counts"]["ok"], 1)

    def test_fail_threshold_triggers_callback(self):
        calls: list[tuple[str, dict]] = []

        def on_failure(name: str, status: dict) -> None:
            calls.append((name, status))

        monitor = HealthMonitor(fail_threshold_per_sensor=2, on_failure=on_failure)
        self.assertEqual(monitor.observe({"ft": "error"}), "degraded")
        self.assertEqual(monitor.observe({"ft": "error"}), "degraded")
        self.assertEqual(monitor.observe({"ft": "error"}), "failed")
        self.assertEqual(calls[0][0], "ft")


if __name__ == "__main__":
    unittest.main()
