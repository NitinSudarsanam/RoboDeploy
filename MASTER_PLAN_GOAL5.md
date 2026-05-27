# RoboDeploy — Master Plan (Iteration 4)

**Canonical state**: `history.json`

## Standing process

- Switch plan/agent freely; no permission prompts.
- Subtask → test → `history.json` → push after each master-plan iteration.
- No subagents; no broad repo search; commands under 2 minutes.

## Goals

1. MuJoCo demo record/replay E2E through RoboEnv.
2. Observation pipeline per-sensor timestamp buffers (if needed).
3. True batched backend `step_multi` (not sequential VecEnv only).
4. Robomimic example with injectable `predict_fn` fallback.
5. Builtin import hygiene for optional dependencies.

## Subtasks

| ID | Title |
|----|--------|
| goal5-subtask1 | MuJoCo demo replay E2E |
| goal5-subtask2 | Obs pipeline sensor buffers |
| goal5-subtask3 | Batched backend contract |
| goal5-subtask4 | Robomimic example path |
| goal5-subtask5 | Builtin optional-dep guards |
