"""LearnedPolicyBase — shared scaffold for checkpoint-backed policies."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.base import PolicyBase
from robodeploy.policies.learned.adapter import LearnedActionAdapter
from robodeploy.policies.learned.loader import LoadedModel, ModelLoader, ModelSpec

ObsPreprocessFn = Callable[[Observation], dict[str, np.ndarray]]


class LearnedPolicyBase(PolicyBase):
    """Common scaffold: ModelLoader + LearnedActionAdapter + obs preprocessing."""

    def __init__(
        self,
        *,
        action_space: ActionSpace,
        config: dict | None = None,
        model_spec: ModelSpec | None = None,
        obs_keys: list[str] | None = None,
        adapter_kwargs: dict | None = None,
        loader: ModelLoader | None = None,
    ) -> None:
        super().__init__(action_space=action_space, config=config or {})
        self._loader = loader or ModelLoader()
        self._model: LoadedModel | None = None
        self._adapter: LearnedActionAdapter | None = None
        self._obs_preprocess: ObsPreprocessFn | None = None
        self._ik_solver: Any | None = None

        if model_spec is not None:
            self._setup_from_spec(model_spec, obs_keys=obs_keys, adapter_kwargs=adapter_kwargs)
        elif self.config.get("model_spec"):
            self._setup_from_spec(dict(self.config["model_spec"]), obs_keys=obs_keys, adapter_kwargs=adapter_kwargs)

    def _setup_from_spec(
        self,
        model_spec: ModelSpec,
        *,
        obs_keys: list[str] | None,
        adapter_kwargs: dict | None,
    ) -> None:
        predict_fn = self.config.get("predict_fn")
        predict_batch_fn = self.config.get("predict_batch_fn")
        predict_plan_fn = self.config.get("predict_plan_fn")
        if predict_fn is not None or predict_plan_fn is not None:
            self._loader = ModelLoader(
                predict_fn=predict_fn,
                predict_batch_fn=predict_batch_fn,
                predict_plan_fn=predict_plan_fn,
            )
        self._model = self._loader.load(model_spec)
        keys = obs_keys or self._model.required_obs_keys
        self._obs_preprocess = self._build_preprocess(keys)
        adapter_cfg = dict(adapter_kwargs or self.config.get("adapter_kwargs") or {})
        self._adapter = LearnedActionAdapter(
            source_space=self._model.action_space,
            target_space=self._action_space,
            source_dim=self._model.action_dim,
            target_dim=self._infer_target_dim(),
            normalization=model_spec.get("action_normalization"),
            ik_solver=self._ik_solver,
            arm_dof=int(self.config.get("arm_dof", self._infer_target_dim())),
            **adapter_cfg,
        )

    def bind_runtime(self, backend, description=None) -> None:
        del backend
        if description is None:
            return
        try:
            self._ik_solver = description.get_kinematics_solver()
        except Exception:
            self._ik_solver = None
        if self._adapter is not None:
            self._adapter._ik_solver = self._ik_solver  # noqa: SLF001

    def reset(self, *, seed: int | None = None) -> None:
        super().reset(seed=seed)
        if self._adapter is not None:
            self._adapter.reset()

    def get_action(self, obs: Observation) -> Action:
        if self._model is None or self._adapter is None or self._obs_preprocess is None:
            raise RuntimeError(f"{type(self).__name__} model is not loaded.")
        obs_dict = self._obs_preprocess(obs)
        model_output = self._model.predict_fn(obs_dict)
        return self._adapter(np.asarray(model_output), obs)

    def get_action_batch(self, obs_batch: list[Observation]) -> list[Action]:
        if self._model is None or self._adapter is None or self._obs_preprocess is None:
            raise RuntimeError(f"{type(self).__name__} model is not loaded.")
        if self._model.predict_batch_fn is None:
            return [self.get_action(o) for o in obs_batch]
        batch = [self._obs_preprocess(o) for o in obs_batch]
        outputs = self._model.predict_batch_fn(batch)
        return [self._adapter(np.asarray(out), obs) for out, obs in zip(outputs, obs_batch)]

    def _infer_target_dim(self) -> int:
        return int(self.config.get("target_dim", self.config.get("arm_dof", 7)))

    def _build_preprocess(self, obs_keys: list[str]) -> ObsPreprocessFn:
        obs_key = str(self.config.get("obs_key", "state"))
        arm_dof = int(self.config.get("arm_dof", 7))

        def preprocess(obs: Observation) -> dict[str, np.ndarray]:
            payload: dict[str, np.ndarray] = {}
            for key in obs_keys:
                if key in {"state", obs_key}:
                    joint_pos = np.asarray(obs.joint_positions, dtype=np.float32)[:arm_dof]
                    joint_vel = np.asarray(obs.joint_velocities, dtype=np.float32)[:arm_dof]
                    if obs.gripper_state is not None:
                        gripper = np.array([obs.gripper_state], dtype=np.float32)
                    else:
                        gripper = np.zeros(1, dtype=np.float32)
                    if obs.rgb is not None:
                        rgb = np.asarray(obs.rgb, dtype=np.float32)
                        if rgb.ndim == 3:
                            rgb = rgb.reshape(-1)
                        payload[key] = np.concatenate([joint_pos, joint_vel, gripper, rgb[:32]])
                    else:
                        payload[key] = np.concatenate([joint_pos, joint_vel, gripper])
                elif key == "rgb" and obs.rgb is not None:
                    payload[key] = np.asarray(obs.rgb, dtype=np.float32)
                elif key == "instruction":
                    text = str(obs.language_instruction or self._instruction or "")
                    payload[key] = np.asarray([len(text)], dtype=np.float32)
            if not payload and obs_key not in payload:
                payload[obs_key] = np.zeros(arm_dof, dtype=np.float32)
            return payload

        return preprocess
