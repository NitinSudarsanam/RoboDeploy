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


def export_recorded_episode(env, steps: int, path: str | Path, *, action_fn=None) -> DemoRecorder:  # noqa: ANN001
    """Run an episode with recording and write JSONL in one call."""
    recorder = env.run_episode(steps, action_fn=action_fn, record=True)
    export_demo_jsonl(recorder, path)
    return recorder


def export_demo_hdf5(recorder: DemoRecorder, path: str | Path) -> None:
    """Write demo frames to HDF5 when h5py is installed."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("export_demo_hdf5 requires h5py. pip install h5py") from exc

    out = Path(path)
    with h5py.File(out, "w") as handle:
        handle.attrs["frame_count"] = len(recorder.frames)
        for index, frame in enumerate(recorder.frames):
            grp = handle.create_group(f"frame_{index}")
            grp.attrs["reward"] = frame.reward
            grp.attrs["done"] = int(frame.done)
            grp.create_dataset("observation_json", data=json.dumps(frame.observation))
            grp.create_dataset("action_json", data=json.dumps(frame.action))
