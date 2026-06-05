"""Cartesian reach policy for PickPlaceTask (MuJoCo IK + joint-space tracking)."""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np

from robodeploy.core.registry import register_policy
from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation, SceneSpec
from robodeploy.policies.base import PolicyBase

from examples.policies.mujoco_ik import MujocoIkSolver, attach_mujoco_ik


class _Phase(Enum):
    SETTLE_HOME = auto()
    PREGRASP = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSIT = auto()
    PLACE = auto()
    RETREAT = auto()
    HOLD = auto()


def _waypoints_from_scene(scene: SceneSpec) -> dict[str, np.ndarray]:
    """Build EE targets from task prop layout (table-top pick-place)."""
    props = {p.name: p for p in scene.props}
    source = props.get("source")
    target = props.get("target")
    src = np.array(source.position if source else (0.55, 0.0, 0.41), dtype=np.float32)
    tgt = np.array(target.position if target else (0.60, 0.20, 0.41), dtype=np.float32)
    if source is not None and source.geom is not None and source.geom.kind == "box":
        half_z = float(source.geom.size[2]) if len(source.geom.size) >= 3 else 0.025
        src[2] += half_z
    if target is not None and target.geom is not None and target.geom.kind == "box":
        half_z = float(target.geom.size[2]) if len(target.geom.size) >= 3 else 0.003
        tgt[2] += half_z
    return {
        "source": src,
        "target": tgt,
    }


@register_policy("example_reach_pick")
class ReachPickPlacePolicy(PolicyBase):
    """Reach-and-place using MuJoCo IK when bound, else joint-space fallback."""

    def __init__(
        self,
        *,
        home_qpos: list[float] | None = None,
        scene: SceneSpec | None = None,
        blend: float = 0.22,
        settle_dist: float = 0.025,
        steps_per_phase: int = 180,
        carry_mode: str = "kinematic",
        description=None,
        config: dict | None = None,
    ) -> None:
        policy_cfg = {"action_hz": 50.0, "carry_mode": str(carry_mode)}
        if config:
            policy_cfg.update(dict(config))
        super().__init__(action_space=ActionSpace.JOINT_POS, config=policy_cfg)
        self._description = description
        self._home = np.array(
            home_qpos if home_qpos is not None else [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0],
            dtype=np.float32,
        )
        self._blend = float(blend)
        self._settle_dist = float(settle_dist)
        self._steps_per_phase = int(steps_per_phase)
        self._ik: Optional[MujocoIkSolver] = None
        self._pin_solver = None
        if description is not None:
            try:
                self._pin_solver = description.get_kinematics_solver()
            except Exception:
                self._pin_solver = None

        wps = _waypoints_from_scene(scene if scene is not None else SceneSpec())
        src, tgt = wps["source"], wps["target"]
        self._ee_targets = {
            _Phase.PREGRASP: src + np.array([0.0, 0.0, 0.10], dtype=np.float32),
            _Phase.GRASP: src + np.array([0.0, 0.0, 0.015], dtype=np.float32),
            _Phase.LIFT: src + np.array([0.0, 0.0, 0.14], dtype=np.float32),
            _Phase.TRANSIT: tgt + np.array([0.0, 0.0, 0.12], dtype=np.float32),
            _Phase.PLACE: tgt + np.array([0.0, 0.0, 0.02], dtype=np.float32),
            _Phase.RETREAT: tgt + np.array([0.0, 0.0, 0.12], dtype=np.float32),
            _Phase.HOLD: tgt + np.array([0.0, 0.0, 0.12], dtype=np.float32),
        }
        self._phase = _Phase.SETTLE_HOME
        self._phase_step = 0
        self._q_goal = self._home.copy()
        self._backend = None
        self._carrying = False
        self._carry_offset = np.array([0.0, 0.0, 0.03], dtype=np.float32)
        self._kinematic_carry = str(self.config.get("carry_mode", "kinematic")).lower() != "none"

    def set_ik_solver(self, solver: MujocoIkSolver) -> None:
        self._ik = solver

    def _set_ee_targets(self, source: np.ndarray, target: np.ndarray) -> None:
        src = np.asarray(source, dtype=np.float32).reshape(3)
        tgt = np.asarray(target, dtype=np.float32).reshape(3)
        self._ee_targets = {
            _Phase.PREGRASP: src + np.array([0.0, 0.0, 0.10], dtype=np.float32),
            _Phase.GRASP: src + np.array([0.0, 0.0, 0.015], dtype=np.float32),
            _Phase.LIFT: src + np.array([0.0, 0.0, 0.14], dtype=np.float32),
            _Phase.TRANSIT: tgt + np.array([0.0, 0.0, 0.12], dtype=np.float32),
            _Phase.PLACE: tgt + np.array([0.0, 0.0, 0.02], dtype=np.float32),
            _Phase.RETREAT: tgt + np.array([0.0, 0.0, 0.12], dtype=np.float32),
            _Phase.HOLD: tgt + np.array([0.0, 0.0, 0.12], dtype=np.float32),
        }

    def bind_runtime(self, backend, description=None) -> None:
        self.attach_mujoco(backend, description)

    def attach_mujoco(self, backend, description=None) -> None:
        desc = description or self._description
        if desc is None:
            return
        self._backend = backend
        attach_mujoco_ik(self, backend, desc)
        if hasattr(backend, "get_prop_pose"):
            try:
                src_pos, _ = backend.get_prop_pose("source")
                tgt_pos, _ = backend.get_prop_pose("target")
                self._set_ee_targets(np.array(src_pos, dtype=np.float32), np.array(tgt_pos, dtype=np.float32))
            except (KeyError, NotImplementedError, RuntimeError):
                pass

    def _reset_impl(self) -> None:
        self._phase = _Phase.SETTLE_HOME
        self._phase_step = 0
        self._q_goal = self._home.copy()
        self._carrying = False

    def _sync_carried_object(self, ee: np.ndarray) -> None:
        if not self._kinematic_carry or not self._carrying or self._backend is None:
            return
        if not hasattr(self._backend, "set_prop_pose"):
            return
        pos = tuple(float(v) for v in (ee + self._carry_offset))
        quat = (1.0, 0.0, 0.0, 0.0)
        try:
            self._backend.set_prop_pose("source", pos, quat)
        except KeyError:
            pass

    def _advance_phase(self) -> None:
        order = [
            _Phase.SETTLE_HOME,
            _Phase.PREGRASP,
            _Phase.GRASP,
            _Phase.LIFT,
            _Phase.TRANSIT,
            _Phase.PLACE,
            _Phase.RETREAT,
            _Phase.HOLD,
        ]
        idx = order.index(self._phase)
        if idx + 1 < len(order):
            self._phase = order[idx + 1]
        self._phase_step = 0

    def _solve_ik(self, q_init: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        if self._ik is not None:
            return self._ik.solve(q_init, target_pos)
        if self._pin_solver is not None:
            _, quat = self._pin_solver.fk(q_init)
            return self._pin_solver.ik(target_pos, quat, q_init=q_init).astype(np.float32)
        return self._fallback_delta(q_init, target_pos)

    def _fallback_delta(self, q: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Joint-space fallback when IK is unavailable (e.g. unit tests)."""
        del target_pos
        return self._track_toward(q, self._home)

    def _track_toward(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        err = q_goal - q
        step = np.clip(err, -0.12, 0.12)
        return (q + self._blend * step).astype(np.float32)

    def get_action(self, obs: Observation) -> Action:
        self._phase_step += 1
        q = np.asarray(obs.joint_positions, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._home.shape[0]:
            q = self._home.copy()

        if self._phase is _Phase.SETTLE_HOME:
            self._q_goal = self._home.copy()
            if self._phase_step >= 40:
                self._advance_phase()
                self._q_goal = self._solve_ik(q, self._ee_targets[_Phase.PREGRASP])
            return Action(joint_positions=self._track_toward(q, self._q_goal))

        ee = np.asarray(obs.ee_position, dtype=np.float32).reshape(3)
        target_ee = self._ee_targets[self._phase]
        dist = float(np.linalg.norm(target_ee - ee))

        if self._phase is _Phase.GRASP and dist < self._settle_dist * 1.5:
            self._carrying = True

        if self._carrying and self._phase in (
            _Phase.LIFT,
            _Phase.TRANSIT,
            _Phase.PLACE,
            _Phase.RETREAT,
            _Phase.HOLD,
        ):
            self._sync_carried_object(ee)

        if self._phase_step == 1 or self._phase_step % 25 == 0:
            self._q_goal = self._solve_ik(q, target_ee)

        q_cmd = self._track_toward(q, self._q_goal)

        if dist < self._settle_dist or self._phase_step >= self._steps_per_phase:
            if self._phase is _Phase.PLACE and self._carrying:
                self._carrying = False
            self._advance_phase()
            if self._phase is not _Phase.SETTLE_HOME and self._phase is not _Phase.HOLD:
                self._q_goal = self._solve_ik(q, self._ee_targets[self._phase])

        return Action(joint_positions=q_cmd)
