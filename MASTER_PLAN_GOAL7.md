# RoboDeploy — Master Plan (Iteration 6)

**Canonical state**: `history.json`

## Goals

1. MuJoCo one-step smoke (reset + step) in CI or scripted check.
2. Wire ObsPipeline.buffer_sensor from backend sensor merge path.
3. HDF5 dataset export (optional h5py extra).
4. Env.auto_record_demo helper on RoboEnv.
5. Gazebo/Isaac contract tests only (no live sim required).

## Subtasks

| ID | Title |
|----|--------|
| goal7-subtask1 | MuJoCo step smoke |
| goal7-subtask2 | Auto buffer_sensor wiring |
| goal7-subtask3 | HDF5 export optional |
| goal7-subtask4 | RoboEnv demo recording helper |
| goal7-subtask5 | Gazebo Isaac contract regression |
