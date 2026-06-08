"""ModelLoader — checkpoint resolution and learned-model contract validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

import numpy as np

from robodeploy.core.spaces import ActionSpace

FrameworkName = Literal["robomimic", "diffusion", "openvla", "pi0", "octo", "custom"]
PredictFn = Callable[[dict[str, np.ndarray]], np.ndarray]
PredictBatchFn = Callable[[list[dict[str, np.ndarray]]], np.ndarray]


class ModelContractError(ValueError):
    """Raised when a loaded model does not match its ModelSpec contract."""


class ModelSpec(TypedDict, total=False):
    framework: FrameworkName
    checkpoint: str | Path
    config_path: str | None
    expected_action_space: ActionSpace
    expected_action_dim: int
    expected_obs_keys: list[str]
    obs_normalization: dict[str, Any] | None
    action_normalization: dict[str, Any] | None
    metadata: dict[str, Any]


@dataclass
class LoadedModel:
    predict_fn: PredictFn
    predict_batch_fn: PredictBatchFn | None
    action_space: ActionSpace
    action_dim: int
    required_obs_keys: list[str]
    framework: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _action_space_from_value(value: ActionSpace | str) -> ActionSpace:
    if isinstance(value, ActionSpace):
        return value
    raw = str(value).strip().upper()
    aliases = {
        "JOINT_POS": ActionSpace.JOINT_POS,
        "JOINT_VEL": ActionSpace.JOINT_VEL,
        "JOINT_TORQUE": ActionSpace.JOINT_TORQUE,
        "CARTESIAN_POSE": ActionSpace.CARTESIAN_POSE,
        "DELTA_EE": ActionSpace.DELTA_EE,
    }
    if raw not in aliases:
        raise ValueError(f"Unknown action space: {value}")
    return aliases[raw]


class ModelLoader:
    """Resolve checkpoint paths, instantiate models, and validate I/O contracts."""

    def __init__(
        self,
        *,
        search_paths: list[Path] | None = None,
        hf_cache: Path | None = None,
        s3_client: Any = None,
        predict_fn: PredictFn | None = None,
        predict_batch_fn: PredictBatchFn | None = None,
        predict_plan_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._paths = list(search_paths or [Path.home() / ".robodeploy" / "models"])
        self._hf_cache = hf_cache or (Path.home() / ".robodeploy" / "hf_cache")
        self._s3_client = s3_client
        self._predict_fn = predict_fn
        self._predict_batch_fn = predict_batch_fn
        self._predict_plan_fn = predict_plan_fn

    def resolve(self, ref: str | Path) -> Path:
        ref_str = str(ref)
        if ref_str.startswith("hf://"):
            return self._download_hf(ref_str)
        if ref_str.startswith("s3://"):
            return self._download_s3(ref_str)
        path = Path(ref)
        if path.is_file():
            return path
        for root in self._paths:
            candidate = root / ref
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Model {ref} not found in {self._paths}")

    def load(self, spec: ModelSpec) -> LoadedModel:
        framework = str(spec.get("framework", "custom"))
        checkpoint = spec.get("checkpoint", "")
        ckpt_path = self._resolve_checkpoint(checkpoint, framework, spec)
        loaders = {
            "robomimic": self._load_robomimic,
            "diffusion": self._load_diffusion,
            "openvla": self._load_openvla,
            "pi0": self._load_pi0,
            "octo": self._load_octo,
            "custom": self._load_custom,
        }
        loader = loaders.get(framework)
        if loader is None:
            raise ValueError(f"Unknown framework: {framework}")
        model = loader(ckpt_path, spec)
        self._validate(model, spec)
        return model

    def load_from_callables(
        self,
        *,
        predict_fn: PredictFn,
        action_space: ActionSpace,
        action_dim: int,
        obs_keys: list[str] | None = None,
        predict_batch_fn: PredictBatchFn | None = None,
        framework: str = "custom",
        metadata: dict[str, Any] | None = None,
    ) -> LoadedModel:
        return LoadedModel(
            predict_fn=predict_fn,
            predict_batch_fn=predict_batch_fn,
            action_space=action_space,
            action_dim=int(action_dim),
            required_obs_keys=list(obs_keys or ["state"]),
            framework=framework,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _probe_action_dim(predict_fn: PredictFn, obs_keys: list[str]) -> int:
        sample = {key: np.zeros(8, dtype=np.float32) for key in obs_keys}
        try:
            output = predict_fn(sample)
        except Exception:
            return 7
        return int(np.asarray(output, dtype=np.float64).reshape(-1).size)

    def _resolve_checkpoint(self, checkpoint: str | Path, framework: str, spec: ModelSpec) -> Path:
        if not checkpoint:
            return Path(".")
        metadata = dict(spec.get("metadata") or {})
        if framework == "custom" and (
            self._predict_fn is not None
            or metadata.get("predict_fn") is not None
            or callable(metadata.get("predict_fn"))
        ):
            path = Path(str(checkpoint))
            if path.is_file():
                return path
            try:
                return self.resolve(checkpoint)
            except FileNotFoundError:
                return path
        return self.resolve(checkpoint)

    def _validate(self, model: LoadedModel, spec: ModelSpec) -> None:
        expected_dim = spec.get("expected_action_dim")
        if expected_dim is not None and model.action_dim != int(expected_dim):
            raise ModelContractError(
                f"Model action_dim={model.action_dim} does not match "
                f"expected_action_dim={expected_dim}."
            )
        expected_keys = spec.get("expected_obs_keys")
        if expected_keys is not None:
            missing = set(model.required_obs_keys) - set(expected_keys)
            if missing:
                raise ModelContractError(
                    f"Model requires obs keys {sorted(missing)} not listed in expected_obs_keys."
                )
        expected_space = spec.get("expected_action_space")
        if expected_space is not None and model.action_space != _action_space_from_value(expected_space):
            raise ModelContractError(
                f"Model action_space={model.action_space.name} does not match "
                f"expected_action_space={_action_space_from_value(expected_space).name}."
            )

    def _download_hf(self, ref: str) -> Path:
        repo_path = ref[len("hf://") :]
        if "/" not in repo_path:
            raise ValueError(f"Invalid hf:// ref (expected hf://org/repo[/path]): {ref}")
        repo_id, _, subpath = repo_path.partition("/")
        filename = subpath or "model.pt"
        cache_dir = self._hf_cache / repo_id.replace("/", "__")
        target = cache_dir / filename
        if target.exists():
            return target
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for hf:// checkpoints. "
                "Install with: pip install huggingface_hub"
            ) from exc
        cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(cache_dir))
        return Path(downloaded)

    def _download_s3(self, ref: str) -> Path:
        if self._s3_client is None:
            try:
                import boto3

                self._s3_client = boto3.client("s3")
            except ImportError as exc:
                raise ImportError("boto3 is required for s3:// checkpoints.") from exc
        without_scheme = ref[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        target = self._hf_cache / "s3" / bucket / key
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        self._s3_client.download_file(bucket, key, str(target))
        return target

    def _load_custom(self, ckpt_path: Path, spec: ModelSpec) -> LoadedModel:
        metadata = dict(spec.get("metadata") or {})
        predict_fn = self._predict_fn or metadata.get("predict_fn")
        if predict_fn is None and ckpt_path.is_file():
            predict_fn = self._predict_fn_from_checkpoint(ckpt_path, metadata)
        if predict_fn is None:
            raise ValueError(
                "custom framework requires predict_fn in ModelLoader, spec metadata, or a torch checkpoint."
            )
        action_space = _action_space_from_value(
            spec.get("expected_action_space", metadata.get("action_space", ActionSpace.JOINT_POS))
        )
        obs_keys = list(spec.get("expected_obs_keys") or metadata.get("obs_keys") or ["state"])
        if "action_dim" in metadata:
            action_dim = int(metadata["action_dim"])
        elif callable(predict_fn):
            action_dim = self._probe_action_dim(predict_fn, obs_keys)
        else:
            action_dim = int(spec.get("expected_action_dim", 7))
        return LoadedModel(
            predict_fn=predict_fn,
            predict_batch_fn=self._predict_batch_fn or metadata.get("predict_batch_fn"),
            action_space=action_space,
            action_dim=action_dim,
            required_obs_keys=obs_keys,
            framework="custom",
            metadata=metadata,
        )

    def _predict_fn_from_checkpoint(self, ckpt_path: Path, metadata: dict[str, Any]) -> PredictFn | None:
        suffix = ckpt_path.suffix.lower()
        if suffix not in {".pt", ".pth", ".ckpt"}:
            return None
        try:
            import torch
        except ImportError:
            return None

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and callable(payload.get("predict_fn")):
            return payload["predict_fn"]
        if isinstance(payload, dict) and "state_dict" in payload:
            metadata.setdefault("checkpoint_payload", "torch_state_dict")
        config_path = metadata.get("config_path")
        if config_path and Path(config_path).exists():
            metadata.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
        return metadata.get("predict_fn")

    def _load_robomimic(self, ckpt_path: Path, spec: ModelSpec) -> LoadedModel:
        metadata = dict(spec.get("metadata") or {})
        if self._predict_fn is not None:
            return self._load_custom(ckpt_path, spec)
        try:
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.torch_utils as TorchUtils
        except ImportError as exc:
            raise ImportError(
                "robomimic is not installed. Install with: pip install robomimic "
                "or pass predict_fn via ModelLoader."
            ) from exc
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Robomimic checkpoint not found: {ckpt_path}")
        device = TorchUtils.get_torch_device(try_to_use_cuda=bool(metadata.get("use_cuda", True)))
        policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=str(ckpt_path),
            device=device,
            verbose=bool(metadata.get("verbose", False)),
        )

        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            return np.asarray(policy(ob=obs_dict), dtype=np.float64).reshape(-1)

        action_dim = int(spec.get("expected_action_dim", metadata.get("arm_dof", 8)))
        return LoadedModel(
            predict_fn=predict_fn,
            predict_batch_fn=None,
            action_space=ActionSpace.JOINT_POS,
            action_dim=action_dim,
            required_obs_keys=list(spec.get("expected_obs_keys") or [metadata.get("obs_key", "state")]),
            framework="robomimic",
            metadata=metadata,
        )

    def _load_diffusion(self, ckpt_path: Path, spec: ModelSpec) -> LoadedModel:
        metadata = dict(spec.get("metadata") or {})
        plan_fn = self._predict_plan_fn or metadata.get("predict_plan_fn") or self._predict_fn
        if plan_fn is None:
            raise ValueError("diffusion framework requires predict_plan_fn or predict_fn.")
        action_space = _action_space_from_value(
            spec.get("expected_action_space", metadata.get("action_space", ActionSpace.DELTA_EE))
        )
        action_dim = int(spec.get("expected_action_dim", metadata.get("action_dim", 7)))

        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            plan = plan_fn(obs_dict)
            if isinstance(plan, dict) and "actions" in plan:
                plan = plan["actions"]
            if isinstance(plan, (list, tuple)) and plan:
                first = plan[0]
                if isinstance(first, dict) and first.get("ee_position") is not None:
                    return np.asarray(first["ee_position"], dtype=np.float64).reshape(-1)
                if isinstance(first, dict) and first.get("joint_positions") is not None:
                    return np.asarray(first["joint_positions"], dtype=np.float64).reshape(-1)
            return np.asarray(plan, dtype=np.float64).reshape(-1)

        return LoadedModel(
            predict_fn=predict_fn,
            predict_batch_fn=self._predict_batch_fn,
            action_space=action_space,
            action_dim=action_dim,
            required_obs_keys=list(spec.get("expected_obs_keys") or ["instruction", "rgb"]),
            framework="diffusion",
            metadata=metadata,
        )

    def _load_openvla(self, ckpt_path: Path, spec: ModelSpec) -> LoadedModel:
        metadata = dict(spec.get("metadata") or {})
        if self._predict_fn is not None:
            return self._load_custom(ckpt_path, spec)
        model, predict_fn = self._build_openvla_predictor(ckpt_path, metadata)
        action_dim = int(spec.get("expected_action_dim", metadata.get("action_dim", 7)))
        return LoadedModel(
            predict_fn=predict_fn,
            predict_batch_fn=self._predict_batch_fn,
            action_space=_action_space_from_value(spec.get("expected_action_space", ActionSpace.DELTA_EE)),
            action_dim=action_dim,
            required_obs_keys=list(spec.get("expected_obs_keys") or ["rgb", "instruction"]),
            framework="openvla",
            metadata={**metadata, "model": model},
        )

    def _load_pi0(self, ckpt_path: Path, spec: ModelSpec) -> LoadedModel:
        metadata = dict(spec.get("metadata") or {})
        if self._predict_fn is not None:
            return self._load_custom(ckpt_path, spec)
        model, predict_fn = self._build_pi0_predictor(ckpt_path, metadata)
        action_dim = int(spec.get("expected_action_dim", metadata.get("action_dim", 7)))
        return LoadedModel(
            predict_fn=predict_fn,
            predict_batch_fn=self._predict_batch_fn,
            action_space=_action_space_from_value(spec.get("expected_action_space", ActionSpace.DELTA_EE)),
            action_dim=action_dim,
            required_obs_keys=list(spec.get("expected_obs_keys") or ["rgb", "instruction"]),
            framework="pi0",
            metadata={**metadata, "model": model},
        )

    def _load_octo(self, ckpt_path: Path, spec: ModelSpec) -> LoadedModel:
        metadata = dict(spec.get("metadata") or {})
        if self._predict_fn is not None:
            return self._load_custom(ckpt_path, spec)
        model, predict_fn = self._build_octo_predictor(ckpt_path, metadata)
        action_dim = int(spec.get("expected_action_dim", metadata.get("action_dim", 7)))
        return LoadedModel(
            predict_fn=predict_fn,
            predict_batch_fn=self._predict_batch_fn,
            action_space=_action_space_from_value(spec.get("expected_action_space", ActionSpace.DELTA_EE)),
            action_dim=action_dim,
            required_obs_keys=list(spec.get("expected_obs_keys") or ["rgb", "instruction"]),
            framework="octo",
            metadata={**metadata, "model": model},
        )

    def _build_openvla_predictor(self, ckpt_path: Path, metadata: dict[str, Any]) -> tuple[Any, PredictFn]:
        try:
            from openvla import OpenVLA  # type: ignore[import-untyped]
        except ImportError:
            try:
                from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction as OpenVLA  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "openvla is not installed. Install with: pip install openvla "
                    f"or pass predict_fn via ModelLoader (checkpoint={ckpt_path})."
                ) from exc
        model = OpenVLA.from_pretrained(str(ckpt_path), **metadata.get("model_kwargs", {}))
        if hasattr(model, "eval"):
            model.eval()

        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            instruction = str(obs_dict.get("instruction", metadata.get("instruction", "")))
            rgb = obs_dict.get("rgb")
            if hasattr(model, "predict_action"):
                out = model.predict_action(rgb=rgb, instruction=instruction)
            elif hasattr(model, "predict"):
                out = model.predict(obs_dict)
            else:
                out = model(obs_dict)
            return np.asarray(out, dtype=np.float64).reshape(-1)

        return model, predict_fn

    def _build_pi0_predictor(self, ckpt_path: Path, metadata: dict[str, Any]) -> tuple[Any, PredictFn]:
        try:
            from pi0.policy import Pi0Policy  # type: ignore[import-untyped]
        except ImportError:
            try:
                from openpi.policies.pi0_policy import Pi0Policy  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "pi0 is not installed. Install physical-intelligence pi0 "
                    f"or pass predict_fn via ModelLoader (checkpoint={ckpt_path})."
                ) from exc
        model = Pi0Policy.from_checkpoint(str(ckpt_path), **metadata.get("model_kwargs", {}))

        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            out = model.predict(obs_dict) if hasattr(model, "predict") else model(obs_dict)
            return np.asarray(out, dtype=np.float64).reshape(-1)

        return model, predict_fn

    def _build_octo_predictor(self, ckpt_path: Path, metadata: dict[str, Any]) -> tuple[Any, PredictFn]:
        try:
            from octo.model.octo_model import OctoModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "octo is not installed. Install with: pip install octo "
                f"or pass predict_fn via ModelLoader (checkpoint={ckpt_path})."
            ) from exc
        model = OctoModel.load_pretrained(str(ckpt_path), **metadata.get("model_kwargs", {}))

        def predict_fn(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
            if hasattr(model, "sample_actions"):
                out = model.sample_actions(obs_dict)
            elif hasattr(model, "predict"):
                out = model.predict(obs_dict)
            else:
                out = model(obs_dict)
            return np.asarray(out, dtype=np.float64).reshape(-1)

        return model, predict_fn
