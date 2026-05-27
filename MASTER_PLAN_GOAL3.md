# RoboDeploy — Master Plan (Iteration 2)

**Canonical state**: `history.json` (this file is navigation only).

## Goals (priority order)

1. **Observation pipeline** — multi-camera `Observation` schema, `ObsPipeline` sync/buffering, timestamp alignment.
2. **Domain randomization** — wire `tasks/randomization.py` into env reset and backend scene props with reproducible seeds.
3. **Robomimic / BC policy** — thin `IPolicy` adapter on the normalized runtime (injectable checkpoint loader).
4. **CI & packaging** — `pyproject.toml` test job, optional extras for mujoco/isaac/ros2, registry honesty gate.
5. **Docs trim** — fold superseded audit/plan markdown into `history.json` entries; keep `ARCHITECTURE.md` + `README.md` current.

## Execution model

Same as iteration 1: subtask → implement → `conda run -n ros2_env python -m unittest discover -s tests` → record in `history.json` → push. No broad repo search; read known paths only. Gazebo/Isaac live E2E only where installed; mock otherwise.

## Subtasks (iteration 2)

| ID | Title |
|----|--------|
| goal3-subtask1 | Observation pipeline sync and multi-camera schema |
| goal3-subtask2 | Domain randomization on reset |
| goal3-subtask3 | Robomimic BC policy adapter |
| goal3-subtask4 | CI extras and doc trim |
