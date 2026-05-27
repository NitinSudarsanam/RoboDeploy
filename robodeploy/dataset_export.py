"""Minimal trajectory export for training datasets (JSONL stub)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from robodeploy.demo_recording import DemoRecorder


def export_demo_jsonl(recorder: DemoRecorder, path: str | Path) -> None:
    """Write demo frames as JSON lines (one frame per line)."""
    out = Path(path)
    with out.open("w", encoding="utf-8") as handle:
        for frame in recorder.frames:
            handle.write(json.dumps(asdict(frame)) + "\n")
