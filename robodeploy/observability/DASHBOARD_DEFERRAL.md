# D9 Hot-Reload Dashboard — Deferred (Phase 10.5)

**Status**: Deferred per `plans/GOAL_10_OBSERVABILITY_REPLAY.md` Phase 10.5.

## Rationale

The stretch dashboard (`FastAPI` + `WebSocket` live charts) depends on optional
`fastapi`, `uvicorn`, and `websockets` and duplicates tooling many teams already
use (W&B, TensorBoard, MLflow). Core observability deliverables (D1–D8, D10–D12)
ship first via JSONL sinks and CLI (`robodeploy logs tail`, `robodeploy logs summary`).

## Workaround

1. Log with `RoboDeployLogger(sinks=[JsonlSink("runs/<name>/run.jsonl")])`.
2. Tail locally: `robodeploy logs tail runs/<name>/`.
3. Summarize: `robodeploy logs summary runs/<name>/`.
4. Optional external sinks: `WandbSink`, `TensorBoardSink`, `MlflowSink`.

## Planned CLI (not implemented)

```bash
robodeploy dashboard --logs runs/2026-06-08-1410/
```

Revisit when benchmark eval HTML reports (Goal 11 D9) and JSONL volume justify
an in-repo live viewer.
