"""Backward-compatible shim for RViz publishing.

New code should import `RvizPublisher` from `robodeploy.viz.rviz_publisher`.
"""

from robodeploy.viz.rviz_publisher import RvizPublisher  # noqa: F401

