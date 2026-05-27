# RoboDeploy — Master Plan (Iteration 14)

**Canonical state**: `history.json`

## Goals

1. Extend the `robodeploy` CLI to load extensions via entry points (`--discover`) and explicit imports (`--custom-module`).
2. Keep extension loading opt-in and safe (no required deps).
3. Record in `history.json`, commit, push.

## Subtasks

| ID | Title |
|----|--------|
| goal15-subtask1 | CLI flags: `list-registry --discover`, `serve-policy --custom-module`, `export-episode --custom-module` |
| goal15-subtask2 | Unit tests for new flags |
| goal15-subtask3 | Update history.json; commit/push; cleanup |

