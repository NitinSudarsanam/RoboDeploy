"""Trajectory export for imitation-learning datasets.

Schema (version 1)
------------------
JSON bundle (``DemoRecorder.save``)::

    {"version": 1, "frames": [{"observation": {...}, "action": {...}, "reward": float, "done": bool}, ...]}

JSONL (``export_demo_jsonl``) — one frame dict per line, same frame keys.

HDF5 (``export_demo_hdf5``) — groups ``frame_{i}`` with attrs ``reward``, ``done`` and datasets
``observation_json``, ``action_json`` (UTF-8 JSON strings).

Observation/action dicts mirror ``robodeploy.core.types.Observation`` / ``Action`` field names.
Training loaders (GOAL 02 ``DemoDataset``) consume these formats.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from robodeploy.demo_recording import DemoFrame, DemoRecorder

DATASET_SCHEMA_VERSION = 1


def _extract_proprio_vector(obs: dict[str, Any]) -> np.ndarray:
    if "proprio" in obs and obs["proprio"] is not None:
        return np.asarray(obs["proprio"], dtype=np.float32).reshape(-1)
    parts: list[np.ndarray] = []
    for key in ("joint_positions", "joint_velocities", "joint_torques"):
        if key in obs and obs[key] is not None:
            parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    if not parts:
        raise ValueError("Observation missing proprio fields for export.")
    return np.concatenate(parts, dtype=np.float32)


def _extract_action_vector(action: dict[str, Any]) -> np.ndarray:
    for key in ("joint_positions", "joint_velocities", "joint_torques"):
        if action.get(key) is not None:
            vec = np.asarray(action[key], dtype=np.float32).reshape(-1)
            if action.get("gripper") is not None:
                vec = np.concatenate([vec, np.asarray([float(action["gripper"])], dtype=np.float32)])
            return vec
    raise ValueError("Action missing joint command fields for export.")


def _first_rgb(obs: dict[str, Any]) -> np.ndarray | None:
    rgb = obs.get("rgb")
    if rgb is None and obs.get("images"):
        images = obs["images"]
        if isinstance(images, dict) and images:
            rgb = next(iter(images.values()))
    if rgb is None:
        return None
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 3:
        return arr
    return None


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


def _lerobot_features_from_frames(frames: list[DemoFrame]) -> dict[str, dict[str, Any]]:
    if not frames:
        raise ValueError("Cannot export empty recorder to LeRobot.")
    proprio = _extract_proprio_vector(frames[0].observation)
    action = _extract_action_vector(frames[0].action)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (int(proprio.shape[0]),),
        },
        "action": {
            "dtype": "float32",
            "shape": (int(action.shape[0]),),
        },
    }
    sample_rgb = _first_rgb(frames[0].observation)
    if sample_rgb is not None:
        h, w, c = sample_rgb.shape
        features["observation.images.camera"] = {
            "dtype": "image",
            "shape": (int(h), int(w), int(c)),
            "names": ["height", "width", "channel"],
        }
    return features


def export_to_lerobot(
    recorder: DemoRecorder,
    *,
    repo_id: str,
    fps: int = 30,
    push_to_hub: bool = False,
    root: str | Path | None = None,
    task: str = "robodeploy_demo",
    robot_type: str | None = "robodeploy",
) -> Any:
    """Export to HuggingFace LeRobot format (loadable via ``DemoDataset.from_lerobot``)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "export_to_lerobot requires the lerobot package. "
            "Install with: pip install robodeploy[teleop] or pip install lerobot."
        ) from exc

    frames = list(recorder.frames)
    if not frames:
        raise ValueError("Cannot export empty recorder to LeRobot.")

    features = _lerobot_features_from_frames(frames)
    has_rgb = "observation.images.camera" in features
    dataset = LeRobotDataset.create(
        repo_id=str(repo_id),
        fps=int(fps),
        features=features,
        root=Path(root) if root is not None else None,
        robot_type=robot_type,
        use_videos=has_rgb,
    )
    for frame in frames:
        lerobot_frame: dict[str, Any] = {
            "observation.state": _extract_proprio_vector(frame.observation),
            "action": _extract_action_vector(frame.action),
            "task": str(task),
        }
        rgb = _first_rgb(frame.observation)
        if rgb is not None:
            lerobot_frame["observation.images.camera"] = rgb
        dataset.add_frame(lerobot_frame)
    dataset.save_episode()
    dataset.finalize()
    if push_to_hub:
        dataset.push_to_hub()
    return dataset


def export_to_rlds(recorder: DemoRecorder, *, output_dir: str | Path) -> Path:
    """Export an RLDS-compatible episode bundle (JSON manifest + numpy arrays)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    episode_dir = out / "episode_0"
    episode_dir.mkdir(parents=True, exist_ok=True)

    observations: list[dict[str, Any]] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[bool] = []

    for frame in recorder.frames:
        observations.append(frame.observation)
        actions.append(_extract_action_vector(frame.action))
        rewards.append(float(frame.reward))
        terminals.append(bool(frame.done))

    np.savez_compressed(
        episode_dir / "steps.npz",
        action=np.stack(actions, axis=0),
        reward=np.asarray(rewards, dtype=np.float32),
        is_terminal=np.asarray(terminals, dtype=bool),
    )
    (episode_dir / "observations.json").write_text(
        json.dumps(observations, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "format": "rlds_compat_v1",
        "frame_count": len(recorder.frames),
        "episode_paths": [str(episode_dir.relative_to(out))],
    }
    manifest_path = out / "rlds_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def export_to_robomimic(recorder: DemoRecorder, *, output_path: str | Path) -> Path:
    """Export robomimic-style HDF5 (loadable via ``DemoDataset.from_robomimic``)."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("export_to_robomimic requires h5py. pip install h5py") from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = list(recorder.frames)
    if not frames:
        raise ValueError("Cannot export empty recorder to robomimic.")

    proprio = np.stack([_extract_proprio_vector(f.observation) for f in frames], axis=0)
    actions = np.stack([_extract_action_vector(f.action) for f in frames], axis=0)
    rewards = np.asarray([f.reward for f in frames], dtype=np.float32)
    dones = np.asarray([f.done for f in frames], dtype=np.uint8)

    with h5py.File(out, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        obs_grp = demo.create_group("obs")
        obs_grp.create_dataset("proprio", data=proprio)
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("rewards", data=rewards)
        demo.create_dataset("dones", data=dones)
        handle.attrs["schema_version"] = DATASET_SCHEMA_VERSION
        handle.attrs["frame_count"] = len(frames)
    return out
