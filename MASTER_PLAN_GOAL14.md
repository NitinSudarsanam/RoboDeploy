# RoboDeploy — Master Plan (Iteration 13)

**Canonical state**: `history.json`

## Goals

1. Move shared dummy test stubs into the package as `robodeploy.testing`.
2. Make `robodeploy export-episode --dummy` work without MuJoCo/ROS2/Gazebo installed.
3. Update tests to import stubs from `robodeploy.testing` (no cross-test coupling).

## Subtasks

| ID | Title |
|----|--------|
| goal14-subtask1 | Add `robodeploy/testing/dummies.py` + exports |
| goal14-subtask2 | Add `--dummy` mode to `robodeploy export-episode` + tests |
| goal14-subtask3 | Update history.json; commit/push; cleanup |

