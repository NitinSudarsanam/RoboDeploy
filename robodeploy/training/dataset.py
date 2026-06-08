"""Demo dataset loaders for behavior cloning and offline RL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from robodeploy.demo_recording import DemoFrame


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Training datasets require PyTorch. Install with: pip install 'robodeploy[training]'"
        ) from exc
    return torch


def _extract_proprio(obs: dict[str, Any]) -> np.ndarray:
    if "proprio" in obs:
        return np.asarray(obs["proprio"], dtype=np.float32)
    parts = []
    for key in ("joint_positions", "joint_velocities", "joint_torques"):
        if key in obs and obs[key] is not None:
            parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    if not parts:
        raise ValueError("Observation dict missing proprio fields.")
    return np.concatenate(parts, dtype=np.float32)


def _extract_action(action: dict[str, Any], *, action_key: str = "joint_positions") -> np.ndarray:
    raw = action.get(action_key)
    if raw is None:
        for key in ("joint_positions", "joint_velocities", "joint_torques"):
            if action.get(key) is not None:
                raw = action[key]
                break
    if raw is None:
        raise ValueError("Action dict missing joint command fields.")
    return np.asarray(raw, dtype=np.float32).reshape(-1)


def _load_jsonl_frames(path: Path) -> list[DemoFrame]:
    frames: list[DemoFrame] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return frames
    if len(lines) == 1:
        payload = json.loads(lines[0])
        if isinstance(payload, dict) and "frames" in payload:
            for item in payload["frames"]:
                frames.append(DemoFrame(**item))
            return frames
        frames.append(DemoFrame(**payload))
        return frames
    for line in lines:
        frames.append(DemoFrame(**json.loads(line)))
    return frames


def _load_hdf5_frames(path: Path) -> list[DemoFrame]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 datasets require h5py. pip install h5py") from exc
    frames: list[DemoFrame] = []
    with h5py.File(path, "r") as handle:
        keys = sorted(
            (k for k in handle.keys() if k.startswith("frame_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for key in keys:
            grp = handle[key]
            obs = json.loads(grp["observation_json"][()].decode("utf-8"))
            act = json.loads(grp["action_json"][()].decode("utf-8"))
            frames.append(
                DemoFrame(
                    observation=obs,
                    action=act,
                    reward=float(grp.attrs.get("reward", 0.0)),
                    done=bool(grp.attrs.get("done", 0)),
                )
            )
    return frames


class DemoDataset:
    """Loads recorded demos (JSONL / HDF5) for supervised learning."""

    def __init__(
        self,
        frames: Sequence[DemoFrame],
        *,
        obs_keys: list[str] | None = None,
        action_key: str = "joint_positions",
    ) -> None:
        self._frames = list(frames)
        self._obs_keys = obs_keys or ["proprio"]
        self._action_key = action_key
        self._proprio_dim = len(_extract_proprio(self._frames[0].observation)) if self._frames else 0
        self._action_dim = len(_extract_action(self._frames[0].action, action_key=action_key)) if self._frames else 0

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        obs_keys: list[str] | None = None,
        action_key: str = "joint_positions",
    ) -> "DemoDataset":
        return cls(_load_hdf5_frames(Path(path)), obs_keys=obs_keys, action_key=action_key)

    @classmethod
    def from_teleop_jsonl(
        cls,
        path: str | Path,
        *,
        obs_keys: list[str] | None = None,
        action_key: str = "joint_positions",
    ) -> "DemoDataset":
        """Load teleop / InteractiveDemoSession JSONL exports (GOAL 04 schema)."""
        from robodeploy.demo_recording import load_demo_jsonl

        return cls(load_demo_jsonl(path), obs_keys=obs_keys, action_key=action_key)

    @classmethod
    def from_jsonl(cls, path: str | Path, *, obs_keys: list[str] | None = None, action_key: str = "joint_positions") -> "DemoDataset":
        """Alias-friendly loader; teleop JSONL uses the same frame schema."""
        path_obj = Path(path)
        try:
            from robodeploy.demo_recording import load_demo_jsonl

            frames = load_demo_jsonl(path_obj)
            if frames:
                return cls(frames, obs_keys=obs_keys, action_key=action_key)
        except Exception:
            pass
        return cls(_load_jsonl_frames(path_obj), obs_keys=obs_keys, action_key=action_key)

    @classmethod
    def from_lerobot(
        cls,
        repo_id: str,
        *,
        root: str | Path | None = None,
        obs_keys: list[str] | None = None,
        action_key: str = "joint_positions",
    ) -> "DemoDataset":
        """Load HuggingFace LeRobot dataset (requires ``lerobot`` package)."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "DemoDataset.from_lerobot requires lerobot. pip install robodeploy[teleop]"
            ) from exc
        dataset = LeRobotDataset(repo_id=str(repo_id), root=Path(root) if root is not None else None)
        frames: list[DemoFrame] = []
        for index in range(len(dataset)):
            item = dataset[index]
            state = item.get("observation.state")
            if state is None:
                state = item.get("observation")
            proprio = np.asarray(state, dtype=np.float32).reshape(-1)
            action_vec = np.asarray(item["action"], dtype=np.float32).reshape(-1)
            obs_dict: dict[str, Any] = {"proprio": proprio.tolist()}
            rgb = item.get("observation.images.camera")
            if rgb is not None:
                arr = np.asarray(rgb)
                if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))
                obs_dict["rgb"] = np.asarray(arr, dtype=np.uint8).tolist()
            act_dict = {action_key: action_vec.tolist()}
            frames.append(
                DemoFrame(
                    observation=obs_dict,
                    action=act_dict,
                    reward=float(item.get("next.reward", item.get("reward", 0.0)) or 0.0),
                    done=bool(item.get("next.done", item.get("done", False))),
                )
            )
        return cls(frames, obs_keys=obs_keys, action_key=action_key)

    @classmethod
    def from_robomimic(
        cls,
        path: str | Path,
        *,
        demo_key: str | None = None,
        obs_keys: list[str] | None = None,
        action_key: str = "joint_positions",
    ) -> "DemoDataset":
        """Load Robomimic HDF5 demonstrations."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("from_robomimic requires h5py") from exc
        frames: list[DemoFrame] = []
        with h5py.File(Path(path), "r") as handle:
            data_grp = handle.get("data")
            if data_grp is None:
                raise ValueError(f"Robomimic HDF5 missing 'data' group: {path}")
            demo_names = sorted(data_grp.keys())
            if demo_key is not None:
                demo_names = [demo_key]
            for name in demo_names:
                demo = data_grp[name]
                obs_grp = demo["obs"]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                rewards = np.asarray(demo.get("rewards", np.zeros(len(actions))), dtype=np.float32)
                dones = np.asarray(demo.get("dones", np.zeros(len(actions))), dtype=np.float32)
                n = len(actions)
                proprio_keys = [k for k in obs_grp.keys() if k in ("robot0_joint_pos", "joint_positions")]
                proprio = (
                    np.asarray(obs_grp[proprio_keys[0]], dtype=np.float32)
                    if proprio_keys
                    else np.zeros((n, actions.shape[-1]), dtype=np.float32)
                )
                for i in range(n):
                    obs_dict = {"joint_positions": proprio[i].tolist()}
                    act_dict = {action_key: actions[i].tolist()}
                    frames.append(
                        DemoFrame(
                            observation=obs_dict,
                            action=act_dict,
                            reward=float(rewards[i]),
                            done=bool(dones[i]),
                        )
                    )
        return cls(frames, obs_keys=obs_keys, action_key=action_key)

    @classmethod
    def from_rlds(
        cls,
        builder: Any,
        *,
        split: str = "train",
        obs_keys: list[str] | None = None,
        action_key: str = "joint_positions",
        max_episodes: int | None = None,
    ) -> "DemoDataset":
        """Load RLDS dataset via TFDS builder or local ``export_to_rlds`` bundle."""
        local_root = Path(builder)
        manifest_path = local_root / "rlds_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            episode_rel = manifest["episode_paths"][0]
            episode_dir = local_root / episode_rel
            steps = np.load(episode_dir / "steps.npz")
            observations = json.loads((episode_dir / "observations.json").read_text(encoding="utf-8"))
            frames: list[DemoFrame] = []
            for index, obs in enumerate(observations):
                frames.append(
                    DemoFrame(
                        observation=obs,
                        action={action_key: steps["action"][index].tolist()},
                        reward=float(steps["reward"][index]),
                        done=bool(steps["is_terminal"][index]),
                    )
                )
            return cls(frames, obs_keys=obs_keys, action_key=action_key)

        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError("from_rlds requires tensorflow_datasets or a local RLDS bundle path") from exc
        ds = tfds.load(builder, split=split)
        frames: list[DemoFrame] = []
        for ep_index, episode in enumerate(ds):
            if max_episodes is not None and ep_index >= max_episodes:
                break
            steps = episode["steps"]
            for step in steps:
                obs_step = step["observation"]
                obs_dict: dict[str, Any] = {}
                if "state" in obs_step:
                    obs_dict["joint_positions"] = np.asarray(obs_step["state"], dtype=np.float32).tolist()
                act_dict = {action_key: np.asarray(step["action"], dtype=np.float32).tolist()}
                frames.append(
                    DemoFrame(
                        observation=obs_dict,
                        action=act_dict,
                        reward=float(step.get("reward", 0.0)),
                        done=bool(step.get("is_terminal", False) or step.get("is_last", False)),
                    )
                )
        return cls(frames, obs_keys=obs_keys, action_key=action_key)

    def to_hdf5(self, path: str | Path) -> None:
        from robodeploy.dataset_export import export_demo_hdf5
        from robodeploy.demo_recording import DemoRecorder

        recorder = DemoRecorder()
        recorder.frames = list(self._frames)
        export_demo_hdf5(recorder, path)

    def to_jsonl(self, path: str | Path) -> None:
        from robodeploy.dataset_export import export_demo_jsonl
        from robodeploy.demo_recording import DemoRecorder

        recorder = DemoRecorder()
        recorder.frames = list(self._frames)
        export_demo_jsonl(recorder, path)

    @property
    def proprio_dim(self) -> int:
        return self._proprio_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch = _require_torch()
        frame = self._frames[int(idx)]
        obs_tensors: dict[str, Any] = {}
        if "proprio" in self._obs_keys:
            obs_tensors["proprio"] = torch.from_numpy(
                _extract_proprio(frame.observation)
            )
        if "rgb" in self._obs_keys:
            rgb = frame.observation.get("rgb")
            if rgb is None and frame.observation.get("images"):
                rgb = next(iter(frame.observation["images"].values()))
            if rgb is not None:
                arr = np.asarray(rgb, dtype=np.uint8)
                obs_tensors["rgb"] = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        action = torch.from_numpy(
            _extract_action(frame.action, action_key=self._action_key)
        )
        return {"obs": obs_tensors, "action": action, "reward": float(frame.reward), "done": bool(frame.done)}


class SequenceDataset(DemoDataset):
    """Windowed trajectories for sequence policies."""

    def __init__(
        self,
        base: DemoDataset,
        *,
        horizon: int,
        pad_strategy: str = "last",
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        super().__init__(
            base._frames,
            obs_keys=base._obs_keys,
            action_key=base._action_key,
        )
        self._horizon = int(horizon)
        self._pad_strategy = pad_strategy

    def __len__(self) -> int:
        return max(0, len(self._frames) - self._horizon + 1)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch = _require_torch()
        end = int(idx) + self._horizon
        window = [super().__getitem__(i) for i in range(int(idx), end)]
        obs: dict[str, list] = {}
        for key in window[0]["obs"]:
            obs[key] = torch.stack([item["obs"][key] for item in window], dim=0)
        actions = torch.stack([item["action"] for item in window], dim=0)
        return {"obs": obs, "action": actions, "reward": window[-1]["reward"], "done": window[-1]["done"]}


class DemoCollator:
    """Pads variable-length trajectory batches."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        torch = _require_torch()
        obs_keys = batch[0]["obs"].keys()
        collated_obs: dict[str, Any] = {}
        for key in obs_keys:
            tensors = [item["obs"][key] for item in batch]
            if tensors[0].ndim == 1:
                collated_obs[key] = torch.stack(tensors, dim=0)
            else:
                collated_obs[key] = torch.stack(tensors, dim=0)
        actions = torch.stack([item["action"] for item in batch], dim=0)
        return {
            "obs": collated_obs,
            "action": actions,
            "reward": torch.tensor([item["reward"] for item in batch], dtype=torch.float32),
            "done": torch.tensor([item["done"] for item in batch], dtype=torch.bool),
        }
