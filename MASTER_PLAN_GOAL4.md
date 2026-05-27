# RoboDeploy — Master Plan (Iteration 3)

**Canonical state**: `history.json`

## Goals

1. **Demonstration I/O** — record/replay trajectories through RoboEnv (no parallel control API).
2. **VecEnv batching** — honest batched obs/action path for training loops.
3. **Hydra configs** — structured YAML for robot/backend/task/policy presets.
4. **Hardware gates** — SO101 / ROS2 smoke tests behind markers, skipped in default CI.
5. **Policy composition** — sequential policy chains on the normalized runtime.

## Subtasks

| ID | Title |
|----|--------|
| goal4-subtask1 | Demonstration record/replay |
| goal4-subtask2 | VecEnv batching contract |
| goal4-subtask3 | Hydra preset wiring |
| goal4-subtask4 | Policy composition chains |
| goal4-subtask5 | Hardware-marked integration tests |
