# RoboDeploy — Master Plan (Iteration 8)

**Canonical state**: `history.json`

## Goals

1. CI job runs MuJoCo reset+step unittest (not just import).
2. SequentialVecEnv + from_preset example snippet.
3. Episode diagnostics test for failed_builtin_imports.
4. PolicyChain via registry in RoboEnv.make config path.
5. Trim duplicate MASTER_PLAN nav files into history-only.

## Subtasks

| ID | Title |
|----|--------|
| goal9-subtask1 | CI mujoco reset+step job |
| goal9-subtask2 | VecEnv preset example |
| goal9-subtask3 | Diagnostics unittest |
| goal9-subtask4 | PolicyChain registry env wiring doc |
| goal9-subtask5 | Master plan doc trim |
