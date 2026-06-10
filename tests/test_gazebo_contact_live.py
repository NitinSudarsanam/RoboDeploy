"""Live + offline Gazebo contact validation."""

from __future__ import annotations

import os
import unittest

from robodeploy.backends.sim.gazebo.contact import GazeboContactMonitor


class _FakeContactEntity:
    def __init__(self, name: str):
        self.name = name


class _FakeContact:
    def __init__(self, a: str, b: str):
        self.collision1 = _FakeContactEntity(a)
        self.collision2 = _FakeContactEntity(b)


class _FakeContactsMsg:
    def __init__(self, pairs: list[tuple[str, str]]):
        self.contact = [_FakeContact(a, b) for a, b in pairs]


class GazeboContactOfflineTests(unittest.TestCase):
    def test_on_contacts_parses_gz_message_shape(self):
        monitor = GazeboContactMonitor()
        monitor._on_contacts(_FakeContactsMsg([("source", "robot0/ee_link"), ("table", "floor")]))
        self.assertTrue(monitor.has_contact("source", "robot0/ee_link"))
        self.assertFalse(monitor.has_contact("source", "floor"))

    def test_has_contact_any_body_match(self):
        monitor = GazeboContactMonitor()
        monitor.inject_contacts([("target", "robot0/gripper")])
        self.assertTrue(monitor.has_contact("target"))
        self.assertTrue(monitor.has_contact("robot0/gripper"))

    def test_fuzzy_harmonic_contact_names(self):
        monitor = GazeboContactMonitor()
        monitor.inject_contacts([("robot0::source::collision", "robot0::ee_link::collision")])
        self.assertTrue(monitor.has_contact("source", "ee_link"))
        self.assertTrue(monitor.has_contact("source", "robot0/ee_link"))
        self.assertFalse(monitor.has_contact("target", "ee_link"))


LIVE = os.environ.get("ROBODEPLOY_LIVE_GAZEBO", "").strip() in {"1", "true", "yes"}


@unittest.skipUnless(LIVE, "set ROBODEPLOY_LIVE_GAZEBO=1 to run live Gazebo contact tests")
class GazeboContactLiveTests(unittest.TestCase):
    def test_contact_monitor_subscribes_when_transport_available(self):
        class _FakeNode:
            def __init__(self):
                self.subscriptions: list[tuple[str, object]] = []

            def subscribe(self, topic, callback):
                self.subscriptions.append((topic, callback))

        node = _FakeNode()
        monitor = GazeboContactMonitor()
        monitor.bind_transport(node, topic="contacts")
        self.assertEqual(len(node.subscriptions), 1)
        topic, callback = node.subscriptions[0]
        self.assertEqual(topic, "contacts")
        callback(_FakeContactsMsg([("cube", "robot0/ee_link")]))
        self.assertTrue(monitor.has_contact("cube", "robot0/ee_link"))


if __name__ == "__main__":
    unittest.main()
