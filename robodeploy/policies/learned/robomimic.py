"""RobomimicPolicy — checkpoint or injectable predict_fn via ModelLoader."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.learned.base import LearnedPolicyBase
from robodeploy.policies.learned.helpers import robomimic_default_spec
from robodeploy.policies.learned.loader import ModelLoader, ModelSpec

PredictFn = Callable[[dict[str, np.ndarray]], np.ndarray]


@register_policy("robomimic")
class RobomimicPolicy(LearnedPolicyBase):
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config: dict | None = None,
        *,
        obs_key: str = "state",
        action_smooth: float = 0.2,
        use_cuda: bool = True,
        arm_dof: int = 7,
        predict_fn: PredictFn | None = None,
        model_spec: ModelSpec | None = None,
    ) -> None:
        cfg = dict(config or {})
        if checkpoint_path is not None:
            cfg.setdefault("checkpoint_path", checkpoint_path)
        cfg.update({"obs_key": obs_key, "action_smooth": action_smooth, "use_cuda": use_cuda, "arm_dof": arm_dof})
        if predict_fn is not None:
            cfg["predict_fn"] = predict_fn
        spec = model_spec or cfg.get("model_spec") or robomimic_default_spec(cfg, predict_fn)
        super().__init__(action_space=ActionSpace.JOINT_POS, config=cfg, model_spec=spec, loader=ModelLoader(predict_fn=predict_fn or cfg.get("predict_fn")))
        self._smooth = float(np.clip(float(cfg.get("action_smooth", action_smooth)), 0.0, 1.0))
        self._arm_dof = int(cfg.get("arm_dof", arm_dof))
        self._prev: Optional[np.ndarray] = None

    def _reset_impl(self, seed: int | None = None) -> None:
        del seed
        self._prev = None

    def get_action(self, obs: Observation) -> Action:
        raw = np.asarray(self._model.predict_fn(self._obs_preprocess(obs)), dtype=np.float64).reshape(-1)
        out = raw if self._prev is None or self._smooth <= 0 else (1.0 - self._smooth) * self._prev + self._smooth * raw
        self._prev = out.copy()
        gripper = float(np.clip(out[self._arm_dof], 0.0, 1.0)) if out.size > self._arm_dof else None
        return Action(joint_positions=out[: self._arm_dof].astype(np.float32), gripper=gripper)
