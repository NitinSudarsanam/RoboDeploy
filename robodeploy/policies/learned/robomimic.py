"""RobomimicPolicy — checkpoint or injectable predict_fn via ModelLoader."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation
from robodeploy.policies.learned.base import LearnedPolicyBase
from robodeploy.policies.learned.helpers import ActionSmoother, arm_gripper_action, robomimic_default_spec
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
        self._smoother, self._arm_dof = ActionSmoother(cfg.get("action_smooth", action_smooth)), int(cfg.get("arm_dof", arm_dof))

    def _reset_impl(self, seed: int | None = None) -> None:
        del seed
        self._smoother.reset()

    def get_action(self, obs: Observation) -> Action:
        raw = np.asarray(self._model.predict_fn(self._obs_preprocess(obs)), dtype=np.float64)
        return arm_gripper_action(self._smoother(raw.reshape(-1)), self._arm_dof)
