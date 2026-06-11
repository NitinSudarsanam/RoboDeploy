"""PegTask — thin re-export of packaged demo task."""

from robodeploy.demos.tasks.peg_insertion import PegTask  # noqa: F401

PegInsertionTask = PegTask

__all__ = ["PegTask", "PegInsertionTask"]
