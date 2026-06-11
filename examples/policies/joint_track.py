"""Joint-space waypoint policy — thin re-export of packaged demo policy."""

from robodeploy.demos.policies.joint_track import JointTrackPolicy  # noqa: F401

__all__ = ["JointTrackPolicy"]
