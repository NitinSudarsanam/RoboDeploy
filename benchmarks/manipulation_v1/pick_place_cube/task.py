"""Tier-2 pick_place_cube — wraps the example PickPlaceTask template."""

from __future__ import annotations

# Re-export registers pick_place via robodeploy.demos.tasks on preset import.
from robodeploy.demos.tasks.pick_place import PickPlaceTask as PickPlaceTask  # noqa: F401
