"""Topic helpers for ROS2 namespacing.

RoboDeploy uses per-robot namespaces like `/robot0`. This helper builds the
absolute topic for a (namespace, topic) pair consistently.
"""

from __future__ import annotations


def resolve_ros_topic(namespace: str, topic: str) -> str:
    ns = str(namespace or "").strip()
    t = str(topic or "").strip()
    if not t:
        raise ValueError("topic must be non-empty")
    if t.startswith("/"):
        return t
    if ns:
        if not ns.startswith("/"):
            ns = "/" + ns
        return f"{ns.rstrip('/')}/{t}"
    return "/" + t

