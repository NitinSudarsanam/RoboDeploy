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

## Revisited wave 2 (2026-06-09)

Wave 2.05 reaffirms deferral of D9 hot-reload dashboard
(`plans/WAVE2_05_POLISH.md` closeout). Benchmark nightly JSONL +
`robodeploy logs tail/summary` and optional W&B/TensorBoard/MLflow sinks
remain the supported observability path for wave 2.

**Revisit trigger**: when eval HTML reports (Goal 11 D9) need a live step stream
without external tooling, or JSONL volume makes CLI-only workflows painful for
demo stakeholders.

**GOAL 10 D9** checkbox stays `[ ]` — intentional defer, not a gap.
