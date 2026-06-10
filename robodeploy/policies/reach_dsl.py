"""YAML-driven reach trajectory policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

_logger = logging.getLogger(__name__)

from robodeploy.core.spaces import ActionSpace
from robodeploy.core.types import Action, Observation, SceneSpec
from robodeploy.policies.base import PolicyBase

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def waypoints_from_scene(scene: SceneSpec) -> dict[str, np.ndarray]:
    """Build EE waypoint bases from task prop layout."""
    props = {p.name: p for p in scene.to_world().props}
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
    return {"source": src, "target": tgt}


@dataclass
class _PhaseSpec:
    name: str
    kind: str
    target: str | None = None
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hold_steps: int = 0
    steps: int = 0
    tracking_blend: float | None = None
    settle_threshold: float | None = None
    max_steps: int = 180
    gripper_command: str | None = None
    engage_carry: bool = False
    release_carry: bool = False
    policy: str | None = None
    instruction: str | None = None
    fallback_to: str | None = None
    fallback_target: str | None = None


class _CompiledPhase:
    def __init__(
        self,
        spec: _PhaseSpec,
        ee_target: np.ndarray | None,
        policy: "ReachTrajectoryPolicy",
        *,
        learned_policy=None,
        fallback_ee_target: np.ndarray | None = None,
    ) -> None:
        self.spec = spec
        self.ee_target = ee_target
        self.fallback_ee_target = fallback_ee_target
        self._policy = policy
        self._learned = learned_policy
        self._fallback_active = False

    @property
    def max_steps(self) -> int:
        if self.spec.hold_steps:
            return int(self.spec.hold_steps)
        if self.spec.steps:
            return int(self.spec.steps)
        return int(self.spec.max_steps)

    def gripper_value(self) -> float | None:
        if self.spec.kind != "gripper":
            return None
        cmd = str(self.spec.gripper_command or "").lower()
        if cmd == "close":
            return 1.0
        if cmd == "open":
            return 0.0
        return None

    def compute(self, obs: Observation, q: np.ndarray) -> np.ndarray:
        kind = self.spec.kind
        if kind == "learned" and not self._fallback_active:
            return q
        if kind == "settle":
            return self._policy._home.copy()
        if kind == "gripper":
            return self._policy._q_goal.copy()
        if kind == "hold" and self.ee_target is None:
            return self._policy._q_goal.copy()
        target = self.fallback_ee_target if self._fallback_active and self.fallback_ee_target is not None else self.ee_target
        if target is not None:
            if self._policy._phase_step == 1 or self._policy._phase_step % 25 == 0:
                self._policy._q_goal = self._policy._solve_ik(q, target)
            return self._policy._q_goal.copy()
        return self._policy._home.copy()

    def learned_action(self, obs: Observation) -> Action | None:
        if self.spec.kind != "learned" or self._learned is None or self._fallback_active:
            return None
        try:
            if self.spec.instruction:
                from dataclasses import replace

                obs = replace(obs, language_instruction=self.spec.instruction)
            return self._learned.get_action(obs)
        except Exception:
            self._fallback_active = True
            return None

    def settled(self, obs: Observation) -> bool:
        kind = self.spec.kind
        if kind == "learned":
            return self._policy._phase_step >= self.max_steps
        if kind in ("settle", "hold", "gripper"):
            return self._policy._phase_step >= self.max_steps
        if self.ee_target is None:
            return False
        ee = np.asarray(obs.ee_position, dtype=np.float32).reshape(3)
        dist = float(np.linalg.norm(self.ee_target - ee))
        threshold = float(
            self.spec.settle_threshold
            if self.spec.settle_threshold is not None
            else self._policy._settle_dist
        )
        return dist < threshold and self._policy._imu_settled(obs)


def _compile_phases(spec: dict[str, Any], policy: "ReachTrajectoryPolicy") -> list[_CompiledPhase]:
    phases_raw = spec.get("phases") or []
    compiled: list[_CompiledPhase] = []
    for entry in phases_raw:
        if not isinstance(entry, dict):
            continue
        phase = _PhaseSpec(
            name=str(entry.get("name", "phase")),
            kind=str(entry.get("kind", "reach")),
            target=entry.get("target"),
            offset=tuple(entry.get("offset", (0.0, 0.0, 0.0))),
            hold_steps=int(entry.get("hold_steps", 0)),
            steps=int(entry.get("steps", 0)),
            tracking_blend=entry.get("tracking_blend"),
            settle_threshold=entry.get("settle_threshold"),
            max_steps=int(entry.get("max_steps", spec.get("steps_per_phase", 180))),
            gripper_command=entry.get("command"),
            engage_carry=bool(entry.get("engage_carry", False)),
            release_carry=bool(entry.get("release_carry", False)),
            policy=entry.get("policy"),
            instruction=entry.get("instruction"),
            fallback_to=str(entry.get("fallback_to")) if entry.get("fallback_to") is not None else None,
            fallback_target=entry.get("fallback_target"),
        )
        ee_target = None
        fallback_ee_target = None
        if phase.kind in ("reach", "hold") and phase.target:
            base = policy._waypoints.get(str(phase.target))
            if base is not None:
                ee_target = base + np.array(phase.offset, dtype=np.float32)
        if phase.fallback_to == "reach" and phase.fallback_target:
            base = policy._waypoints.get(str(phase.fallback_target))
            if base is not None:
                fallback_ee_target = base + np.array(phase.offset, dtype=np.float32)
        learned_policy = None
        if phase.kind == "learned" and phase.policy:
            from robodeploy.policies.learned.factory import load_policy_from_ref

            cfg = {"instruction": phase.instruction} if phase.instruction else {}
            learned_policy = load_policy_from_ref(phase.policy, action_space=policy._action_space, config=cfg)
            learned_policy.reset()
        compiled.append(
            _CompiledPhase(
                phase,
                ee_target,
                policy,
                learned_policy=learned_policy,
                fallback_ee_target=fallback_ee_target,
            )
        )
    return compiled


def _load_spec_dict(data: dict[str, Any]) -> dict[str, Any]:
    if len(data) == 1:
        return next(iter(data.values()))
    return data


class ReachTrajectoryPolicy(PolicyBase):
    """Phase-machine reach policy compiled from YAML or dict spec."""

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        action_space: ActionSpace | None = None,
        scene: SceneSpec | None = None,
        description=None,
        config: dict | None = None,
    ) -> None:
        space_name = spec.get("action_space", "JOINT_POS")
        resolved_space = action_space or ActionSpace[space_name]
        policy_cfg = {
            "action_hz": float(spec.get("action_hz", 50.0)),
            "carry_mode": str((spec.get("carry") or {}).get("mode", spec.get("carry_mode", "kinematic"))),
        }
        if config:
            policy_cfg.update(dict(config))
        super().__init__(action_space=resolved_space, config=policy_cfg)

        self._description = description
        self._home = np.array(spec.get("home", [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0]), dtype=np.float32)
        carry = spec.get("carry") or {}
        self._blend = float(carry.get("follow_blend", spec.get("tracking_blend", 0.22)))
        self._settle_dist = float(spec.get("settle_threshold", 0.025))
        self._steps_per_phase = int(spec.get("steps_per_phase", 180))
        self._ik = None

        self._waypoints = waypoints_from_scene(scene if scene is not None else SceneSpec())
        self._phase_specs_raw = list(spec.get("phases") or [])
        self._phases = _compile_phases(spec, self)
        if not self._phases:
            raise ValueError("Reach trajectory spec must include at least one phase.")

        self._phase_idx = 0
        self._phase_step = 0
        self._q_goal = self._home.copy()
        self._backend = None
        self._carrying = False
        self._carry_offset = np.array([0.0, 0.0, 0.03], dtype=np.float32)
        mode = str(self.config.get("carry_mode", "kinematic")).lower()
        self._carry_mode = mode
        self._kinematic_carry = mode == "kinematic"
        self._backend_follow_carry = mode == "follow"
        self._backend_weld_carry = mode == "weld"
        self._contact_carry = mode == "contact"
        grasp_default = "contact" if self._contact_carry else "distance"
        raw_grasp = self.config.get("grasp_detection", grasp_default)
        self._grasp_detection = str(raw_grasp).lower()
        if self._grasp_detection == "backend_contact":
            self._grasp_detection = "contact"
        self._grasp_force_threshold = float(
            self.config.get("force_threshold", self.config.get("grasp_force_threshold", 2.0))
        )
        self._grasp_force_window = int(self.config.get("grasp_force_window", 5))
        self._grasp_force_loss_threshold = float(self.config.get("grasp_force_loss_threshold", 0.5))
        self._contact_sensor_name = str(self.config.get("contact_sensor", "wrist_contact"))
        self._imu_omega_max = self.config.get("imu_omega_max")
        self._imu_settle_steps = int(self.config.get("imu_settle_steps", 5))
        self._force_history: list[float] = []
        self._imu_settle_count = 0
        self._last_sensor_health: dict[str, object] = {"overall": "ok"}
        self._halt_on_sensor_failure = bool(self.config.get("halt_on_sensor_failure", True))
        self._critical_sensors = set(self.config.get("critical_sensors", ["wrist_ft"]))
        self._gripper_state = 0.0

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        action_space: ActionSpace | None = None,
        scene: SceneSpec | None = None,
        description=None,
        config: dict | None = None,
    ) -> "ReachTrajectoryPolicy":
        if yaml is None:
            raise ImportError("PyYAML is required for ReachTrajectoryPolicy.from_yaml(). Install pyyaml.")
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid reach DSL YAML: {path}")
        spec = _load_spec_dict(raw)
        return cls(spec, action_space=action_space, scene=scene, description=description, config=config)

    def bind_runtime(self, backend, description=None) -> None:
        desc = description or self._description
        if desc is None:
            return
        self._backend = backend
        self._map_gazebo_carry_mode(backend)
        from robodeploy.kinematics.policy_ik import attach_policy_ik

        attach_policy_ik(self, backend, desc)

    def _map_gazebo_carry_mode(self, backend) -> None:
        if not self._backend_weld_carry:
            return
        backend_name = str(getattr(backend, "sensor_backend_name", "") or "").lower()
        if backend_name != "gazebo":
            return
        _logger.warning(
            "carry_mode 'weld' is not supported on Gazebo; using 'follow' instead."
        )
        self._carry_mode = "follow"
        self._backend_weld_carry = False
        self._backend_follow_carry = True

    def attach_mujoco(self, backend, description=None) -> None:
        self.bind_runtime(backend, description)

    def set_ik_solver(self, solver) -> None:  # noqa: ANN001
        self._ik = solver

    def _reset_impl(self, *, seed: int | None = None) -> None:
        del seed
        self._phase_idx = 0
        self._phase_step = 0
        self._q_goal = self._home.copy()
        self._carrying = False
        self._force_history = []
        self._imu_settle_count = 0
        self._last_sensor_health = {"overall": "ok"}
        self._gripper_state = 0.0
        if self._backend is not None and hasattr(self._backend, "set_grasp_prop"):
            self._backend.set_grasp_prop(None)

    def _set_waypoints(self, source: np.ndarray, target: np.ndarray) -> None:
        self._waypoints = {
            "source": np.asarray(source, dtype=np.float32).reshape(3),
            "target": np.asarray(target, dtype=np.float32).reshape(3),
        }
        self._phases = _compile_phases(
            {"phases": self._phase_specs_raw, "steps_per_phase": self._steps_per_phase},
            self,
        )

    def _ft_force_norm(self, obs: Observation) -> float:
        force = obs.ft_force
        if force is None and getattr(obs, "ft_forces", None):
            forces = obs.ft_forces
            if forces:
                force = next(iter(forces.values()))
        if force is None:
            return 0.0
        return float(np.linalg.norm(np.asarray(force, dtype=np.float32)))

    def _contact_sensor_active(self, obs: Observation) -> bool:
        contact = getattr(obs, "contact_state", None) or {}
        return bool(contact.get(self._contact_sensor_name, False))

    def _grasp_engage(self, obs: Observation) -> bool:
        mode = self._grasp_detection
        if mode == "ft":
            self._force_history.append(self._ft_force_norm(obs))
            if len(self._force_history) < self._grasp_force_window:
                return False
            avg = float(np.mean(self._force_history[-self._grasp_force_window :]))
            return avg >= self._grasp_force_threshold
        if mode == "contact":
            return self._contact_sensor_active(obs)
        return True

    def _observe_sensor_health(self, obs: Observation) -> dict[str, object]:
        """Read per-sensor status from the observation (not info.extra)."""
        from robodeploy.observability.health import summarize_sensor_health

        status = dict(getattr(obs, "sensor_status", {}) or {})
        summary = summarize_sensor_health(status)
        self._last_sensor_health = summary
        return summary

    def _sensor_health_blocks_action(self, obs: Observation) -> bool:
        if not self._halt_on_sensor_failure:
            return False
        status = dict(getattr(obs, "sensor_status", {}) or {})
        for name in self._critical_sensors:
            if status.get(name) in {"error", "stale"}:
                return True
        summary = self._observe_sensor_health(obs)
        return str(summary.get("overall", "ok")) == "failed"

    def _imu_settled(self, obs: Observation) -> bool:
        if self._imu_omega_max is None:
            return True
        omega = getattr(obs, "imu_angular_velocity", None)
        if omega is None:
            return True
        settled = float(np.linalg.norm(np.asarray(omega, dtype=np.float32))) <= float(self._imu_omega_max)
        if settled:
            self._imu_settle_count += 1
        else:
            self._imu_settle_count = 0
        return self._imu_settle_count >= self._imu_settle_steps

    def _update_targets_from_obs(self, obs: Observation) -> None:
        objects = getattr(obs, "objects", None) or {}
        if "source" not in objects or "target" not in objects:
            return
        src_pos, _ = objects["source"]
        tgt_pos, _ = objects["target"]
        self._waypoints = {
            "source": np.array(src_pos, dtype=np.float32),
            "target": np.array(tgt_pos, dtype=np.float32),
        }
        self._phases = _compile_phases(
            {"phases": self._phase_specs_raw, "steps_per_phase": self._steps_per_phase},
            self,
        )

    def _solve_ik(self, q_init: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        if self._ik is not None:
            return self._ik.solve(q_init, target_pos)
        return self._fallback_delta(q_init, target_pos)

    def _fallback_delta(self, q: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        del target_pos
        q_goal = self._q_goal
        if np.allclose(q_goal, self._home):
            q_goal = q
        return self._track_toward(q, q_goal)

    def _track_toward(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        err = q_goal - q
        step = np.clip(err, -0.12, 0.12)
        blend = self._blend
        phase = self._phases[self._phase_idx]
        if phase.spec.tracking_blend is not None:
            blend = float(phase.spec.tracking_blend)
        return (q + blend * step).astype(np.float32)

    def _sync_carried_object(self, ee: np.ndarray) -> None:
        if not self._kinematic_carry or not self._carrying or self._backend is None:
            return
        if not hasattr(self._backend, "set_prop_pose"):
            return
        pos = tuple(float(v) for v in (ee + self._carry_offset))
        try:
            self._backend.set_prop_pose("source", pos, (1.0, 0.0, 0.0, 0.0))
        except KeyError:
            pass

    def _maybe_engage_carry(self, obs: Observation, dist: float) -> None:
        phase = self._phases[self._phase_idx]
        if phase.spec.kind != "reach" or "grasp" not in phase.spec.name.lower():
            return
        if dist >= self._settle_dist * 1.5:
            return
        engage = self._grasp_engage(obs)
        if engage:
            self._carrying = True
        if self._backend is None or not hasattr(self._backend, "set_grasp_prop"):
            return
        offset = tuple(float(v) for v in self._carry_offset)
        if engage and self._backend_weld_carry:
            backend_name = str(getattr(self._backend, "sensor_backend_name", "") or "").lower()
            mode = "follow" if backend_name == "gazebo" else "weld"
            self._backend.set_grasp_prop("source", offset=offset, mode=mode)
        elif engage and (self._backend_follow_carry or self._contact_carry):
            self._backend.set_grasp_prop("source", offset=offset, mode="follow")

    def _grasp_phase_idx(self) -> int | None:
        for idx, phase in enumerate(self._phases):
            spec = phase.spec
            if spec.kind == "reach" and (spec.engage_carry or "grasp" in spec.name.lower()):
                return idx
        return None

    def _close_gripper_phase_idx(self) -> int | None:
        for idx, phase in enumerate(self._phases):
            spec = phase.spec
            if spec.name == "close_gripper":
                return idx
            if spec.kind == "gripper" and str(spec.gripper_command or "").lower() == "close":
                return idx
        return None

    def _rewind_to_grasp_phase(self) -> None:
        grasp_idx = self._grasp_phase_idx()
        if grasp_idx is None or grasp_idx >= self._phase_idx:
            return
        close_idx = self._close_gripper_phase_idx()
        self._phase_idx = close_idx if close_idx is not None else grasp_idx
        self._phase_step = 0
        self._force_history = []
        self._gripper_state = 1.0
        for phase in self._phases:
            phase._fallback_active = False

    def _maybe_drop_carry(self, obs: Observation) -> None:
        if not self._carrying or self._grasp_detection != "ft":
            return
        phase = self._phases[self._phase_idx]
        if "lift" not in phase.spec.name.lower() and "transit" not in phase.spec.name.lower():
            return
        if self._ft_force_norm(obs) >= self._grasp_force_loss_threshold:
            return
        self._carrying = False
        self._force_history = []
        if self._backend is not None and hasattr(self._backend, "set_grasp_prop"):
            self._backend.set_grasp_prop(None)
        self._rewind_to_grasp_phase()

    def get_action(self, obs: Observation) -> Action:
        self._observe_sensor_health(obs)
        if self._sensor_health_blocks_action(obs):
            q_hold = np.asarray(obs.joint_positions, dtype=np.float32).reshape(-1)
            if q_hold.shape[0] != self._home.shape[0]:
                q_hold = self._home.copy()
            return Action(joint_positions=q_hold, gripper=self._gripper_state)
        self._update_targets_from_obs(obs)
        self._phase_step += 1
        q = np.asarray(obs.joint_positions, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._home.shape[0]:
            q = self._home.copy()

        phase = self._phases[self._phase_idx]
        learned_action = phase.learned_action(obs)
        if learned_action is not None:
            if phase.settled(obs) or self._phase_step >= phase.max_steps:
                self._phase_idx = min(self._phase_idx + 1, len(self._phases) - 1)
                self._phase_step = 0
            if learned_action.gripper is not None:
                self._gripper_state = float(learned_action.gripper)
            return learned_action

        ee = np.asarray(obs.ee_position, dtype=np.float32).reshape(3)
        if phase.ee_target is not None:
            dist = float(np.linalg.norm(phase.ee_target - ee))
            if phase.spec.engage_carry or "grasp" in phase.spec.name.lower():
                self._maybe_engage_carry(obs, dist)
            self._maybe_drop_carry(obs)
            if self._carrying and phase.spec.kind == "reach":
                self._sync_carried_object(ee)

        gripper_cmd = phase.gripper_value()
        if gripper_cmd is not None:
            self._gripper_state = float(gripper_cmd)
            if gripper_cmd >= 0.5 and phase.spec.kind == "gripper":
                self._carrying = True
            elif gripper_cmd < 0.5 and phase.spec.kind == "gripper":
                self._carrying = False
                if self._backend is not None and hasattr(self._backend, "set_grasp_prop"):
                    self._backend.set_grasp_prop(None)

        q_goal = phase.compute(obs, q)
        q_cmd = self._track_toward(q, q_goal)

        if phase.settled(obs) or self._phase_step >= phase.max_steps:
            if phase.spec.release_carry or ("place" in phase.spec.name.lower() and self._carrying):
                self._carrying = False
                if self._backend is not None and hasattr(self._backend, "set_grasp_prop"):
                    self._backend.set_grasp_prop(None)
            self._phase_idx = min(self._phase_idx + 1, len(self._phases) - 1)
            self._phase_step = 0

        return Action(joint_positions=q_cmd, gripper=self._gripper_state)

    @staticmethod
    def default_pick_place_spec() -> dict[str, Any]:
        """Built-in pick-place phase list when no YAML is available."""
        return {
            "action_space": "JOINT_POS",
            "action_hz": 50.0,
            "home": [0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0],
            "blend": 0.22,
            "settle_threshold": 0.025,
            "steps_per_phase": 180,
            "carry": {"mode": "kinematic"},
            "phases": [
                {"name": "settle_home", "kind": "settle", "hold_steps": 40},
                {
                    "name": "pregrasp",
                    "kind": "reach",
                    "target": "source",
                    "offset": [0.0, 0.0, 0.10],
                    "tracking_blend": 0.22,
                    "settle_threshold": 0.025,
                },
                {
                    "name": "grasp",
                    "kind": "reach",
                    "target": "source",
                    "offset": [0.0, 0.0, 0.015],
                    "settle_threshold": 0.025,
                    "engage_carry": True,
                },
                {"name": "close_gripper", "kind": "gripper", "command": "close", "hold_steps": 10},
                {"name": "lift", "kind": "reach", "target": "source", "offset": [0.0, 0.0, 0.14]},
                {"name": "transit", "kind": "reach", "target": "target", "offset": [0.0, 0.0, 0.12]},
                {"name": "place", "kind": "reach", "target": "target", "offset": [0.0, 0.0, 0.02]},
                {"name": "open_gripper", "kind": "gripper", "command": "open", "hold_steps": 10},
                {"name": "retreat", "kind": "reach", "target": "target", "offset": [0.0, 0.0, 0.12]},
                {"name": "hold", "kind": "hold", "steps": 30, "target": "target", "offset": [0.0, 0.0, 0.12]},
            ],
        }
