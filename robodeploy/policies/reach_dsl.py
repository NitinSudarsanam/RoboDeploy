"""YAML-driven reach trajectory policy."""

from __future__ import annotations

import logging
import os
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


def waypoint_top_offsets(scene: SceneSpec) -> dict[str, np.ndarray]:
    """Per-prop offset from reported center to grasp surface (box top).

    ``Observation.objects`` reports prop centers; reach targets aim at the top
    face. Applied both at init and whenever waypoints refresh from observations.
    """
    props = {p.name: p for p in scene.to_world().props}
    offsets: dict[str, np.ndarray] = {}
    for name, default_half_z in (("source", 0.025), ("target", 0.003)):
        prop = props.get(name)
        half_z = 0.0
        if prop is not None and prop.geom is not None and prop.geom.kind == "box":
            half_z = float(prop.geom.size[2]) if len(prop.geom.size) >= 3 else default_half_z
        offsets[name] = np.array([0.0, 0.0, half_z], dtype=np.float32)
    return offsets


def ee_position_from_obs(obs: Observation) -> np.ndarray:
    """Prefer sensor FK ``obs.ee_pose`` over backend ``obs.ee_position``."""
    pose = getattr(obs, "ee_pose", None)
    if pose is not None:
        return np.asarray(pose, dtype=np.float32).reshape(3)
    return np.asarray(obs.ee_position, dtype=np.float32).reshape(3)


def waypoints_from_scene(scene: SceneSpec) -> dict[str, np.ndarray]:
    """Build EE waypoint bases from task prop layout."""
    props = {p.name: p for p in scene.to_world().props}
    source = props.get("source")
    target = props.get("target")
    src = np.array(source.position if source else (0.55, 0.0, 0.41), dtype=np.float32)
    tgt = np.array(target.position if target else (0.60, 0.20, 0.41), dtype=np.float32)
    offsets = waypoint_top_offsets(scene)
    src += offsets["source"]
    tgt += offsets["target"]
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
            ik_period = 25
            if self._policy._gazebo_honest_place_active() and "place" in self.spec.name.lower():
                ik_period = 4
            if self._policy._phase_step == 1 or self._policy._phase_step % ik_period == 0:
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
        ee = ee_position_from_obs(obs)
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
        if self.config.get("tracking_blend") is not None:
            self._blend = float(self.config["tracking_blend"])
        self._settle_dist = float(spec.get("settle_threshold", 0.025))
        if self.config.get("settle_threshold") is not None:
            self._settle_dist = float(self.config["settle_threshold"])
        self._steps_per_phase = int(
            self.config.get("steps_per_phase", spec.get("steps_per_phase", 180))
        )
        self._ik = None

        scene_spec = scene if scene is not None else SceneSpec()
        self._sensor_only = bool(self.config.get("sensor_only", False))
        self._waypoint_top_offsets = waypoint_top_offsets(scene_spec)
        self._waypoints = {} if self._sensor_only else waypoints_from_scene(scene_spec)
        self._phase_specs_raw = list(spec.get("phases") or [])
        compile_spec = {**spec, "steps_per_phase": self._steps_per_phase}
        self._phases = _compile_phases(compile_spec, self)
        if not self._phases:
            raise ValueError("Reach trajectory spec must include at least one phase.")

        self._phase_idx = 0
        self._phase_step = 0
        self._q_goal = self._home.copy()
        self._q_cmd: np.ndarray | None = None
        self._track_bias = np.zeros_like(self._home)
        self._backend = None
        self._carrying = False
        carry_offset = self.config.get("carry_offset", carry.get("offset", (0.0, 0.0, 0.03)))
        self._carry_offset = np.asarray(carry_offset, dtype=np.float32).reshape(3)
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
        self._map_mujoco_carry_mode(backend)
        from robodeploy.kinematics.policy_ik import attach_policy_ik

        attach_policy_ik(self, backend, desc)
        self._map_rviz_carry_mode(backend)

    def _recompile_phases(self) -> None:
        self._phases = _compile_phases(
            {"phases": self._phase_specs_raw, "steps_per_phase": self._steps_per_phase},
            self,
        )

    def _map_mujoco_carry_mode(self, backend) -> None:
        backend_name = str(getattr(backend, "sensor_backend_name", "") or "").lower()
        if backend_name != "mujoco":
            return
        if not self._sensor_only or not self._backend_follow_carry:
            return
        _logger.info(
            "mujoco: using kinematic carry for sensor_only pick (physics follow ejects props)."
        )
        self._carry_mode = "kinematic"
        self._kinematic_carry = True
        self._backend_follow_carry = False

    def _map_rviz_carry_mode(self, backend) -> None:
        backend_name = str(getattr(backend, "sensor_backend_name", "") or "").lower()
        if backend_name != "ros2_rviz":
            return
        if self._backend_follow_carry:
            _logger.info("ros2_rviz fake-sim: using kinematic carry (no grasp weld).")
            self._carry_mode = "kinematic"
            self._kinematic_carry = True
            self._backend_follow_carry = False
        if self._sensor_only and self._grasp_detection == "ft":
            _logger.info("ros2_rviz fake-sim: distance grasp (FT often zero in fake graph).")
            self._grasp_detection = "distance"
        rviz_min_steps = 280
        if self._steps_per_phase < rviz_min_steps:
            self._steps_per_phase = rviz_min_steps
            self._recompile_phases()

    def _map_gazebo_carry_mode(self, backend) -> None:
        backend_name = str(getattr(backend, "sensor_backend_name", "") or "").lower()
        if backend_name != "gazebo":
            return
        if self._backend_weld_carry:
            _logger.warning(
                "carry_mode 'weld' is not supported on Gazebo; using 'follow' instead."
            )
            self._carry_mode = "follow"
            self._backend_weld_carry = False
            self._backend_follow_carry = True
        if self._sensor_only:
            _logger.info("gazebo: using kinematic carry for sensor_only pick.")
            self._carry_mode = "kinematic"
            self._kinematic_carry = True
            self._backend_follow_carry = False
            self._backend_weld_carry = False
        if self._sensor_only and self._grasp_detection == "ft":
            _logger.info("gazebo: distance grasp (FT often zero in gz).")
            self._grasp_detection = "distance"
        gazebo_min_steps = 280
        if self._steps_per_phase < gazebo_min_steps:
            self._steps_per_phase = gazebo_min_steps
            self._recompile_phases()

    def attach_mujoco(self, backend, description=None) -> None:
        self.bind_runtime(backend, description)

    def set_ik_solver(self, solver) -> None:  # noqa: ANN001
        self._ik = solver

    def _reset_impl(self, *, seed: int | None = None) -> None:
        del seed
        self._phase_idx = 0
        self._phase_step = 0
        self._q_goal = self._home.copy()
        self._q_cmd = None
        self._track_bias = np.zeros_like(self._home)
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

    def _effective_grasp_detection(self, obs: Observation) -> str:
        mode = self._grasp_detection
        if mode != "ft":
            return mode
        status = dict(getattr(obs, "sensor_status", {}) or {})
        ft_status = status.get("wrist_ft", "ok")
        if ft_status in {"stale", "error"} or self._ft_force_norm(obs) <= 0.0:
            if self._contact_sensor_active(obs):
                return "contact"
            _logger.warning("wrist_ft stale/zero; distance fallback for grasp engage")
            return "distance"
        return mode

    def _grasp_engage(self, obs: Observation) -> bool:
        mode = self._effective_grasp_detection(obs)
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
        return False

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
        # While carrying, the source pose tracks the EE — refreshing waypoints
        # from it would make lift/transit targets chase the arm forever.
        if self._carrying:
            return
        objects = getattr(obs, "objects", None) or {}
        if "source" not in objects or "target" not in objects:
            return
        src_pos, _ = objects["source"]
        tgt_pos, _ = objects["target"]
        offsets = getattr(self, "_waypoint_top_offsets", {})
        self._set_waypoints(
            np.array(src_pos, dtype=np.float32) + offsets.get("source", 0.0),
            np.array(tgt_pos, dtype=np.float32) + offsets.get("target", 0.0),
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
        # Advance an internal command state toward the goal instead of offsetting
        # from the measured position: position servos need a growing command/state
        # error to hold against gravity; chasing the measured q caps the actuator
        # torque at kp * blend * clip and the arm stalls.
        if self._q_cmd is None:
            self._q_cmd = np.asarray(q, dtype=np.float32).copy()
        blend = self._blend
        phase = self._phases[self._phase_idx]
        if phase.spec.tracking_blend is not None:
            blend = float(phase.spec.tracking_blend)
        honest_blend = self.config.get("honest_place_tracking_blend")
        if (
            honest_blend is not None
            and self._gazebo_honest_place_active()
            and "place" in phase.spec.name.lower()
        ):
            blend = float(honest_blend)
        err = q_goal - self._q_cmd
        self._q_cmd = (self._q_cmd + blend * np.clip(err, -0.12, 0.12)).astype(np.float32)
        # Integral correction for steady-state servo sag (gravity): once the
        # command has converged, the measured position still lags by
        # gravity_torque / kp; bias the command past the goal to close that gap.
        # Integrate per joint only after its command converged (anti-windup);
        # while still traveling, decay the bias instead.
        converged = np.abs(err) < 0.05
        integrate = self._track_bias + 0.06 * np.clip(q_goal - q, -0.12, 0.12)
        self._track_bias = np.clip(
            np.where(converged, integrate, self._track_bias * 0.95),
            -0.6,
            0.6,
        ).astype(np.float32)
        return (self._q_cmd + self._track_bias).astype(np.float32)

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

    def _gazebo_place_snap_enabled(self) -> bool:
        cfg_snap = self.config.get("gazebo_place_snap")
        if cfg_snap is not None:
            return bool(cfg_snap)
        raw = os.environ.get("ROBODEPLOY_GAZEBO_PLACE_SNAP", "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _gazebo_honest_place_active(self) -> bool:
        backend_name = str(getattr(self._backend, "sensor_backend_name", "") or "").lower()
        return backend_name == "gazebo" and not self._gazebo_place_snap_enabled()

    def _honest_place_settle_m(self) -> float | None:
        raw = self.config.get("honest_place_settle_m")
        if raw is None:
            return None
        return float(raw)

    def _place_phase_release_allowed(self, phase: _CompiledPhase, obs: Observation) -> bool:
        """When place snap is off, defer carry release until EE is within honest tolerance."""
        if not self._gazebo_honest_place_active():
            return True
        if not (
            phase.spec.release_carry
            or ("place" in phase.spec.name.lower() and self._carrying)
        ):
            return True
        settle_m = self._honest_place_settle_m()
        if settle_m is None or phase.ee_target is None:
            return True
        ee = ee_position_from_obs(obs)
        return float(np.linalg.norm(phase.ee_target - ee)) <= settle_m

    def _sync_honest_carry_pose(self, ee: np.ndarray) -> None:
        """Last kinematic sync from EE before honest (no-snap) carry release."""
        if not self._gazebo_honest_place_active() or not self._kinematic_carry:
            return
        if self._backend is None or not hasattr(self._backend, "set_prop_pose"):
            return
        pos = tuple(float(v) for v in (ee + self._carry_offset))
        try:
            self._backend.set_prop_pose("source", pos, (1.0, 0.0, 0.0, 0.0))
        except KeyError:
            pass

    def _maybe_finalize_kinematic_place(self, ee_to_place_dist: float) -> None:
        """Snap kinematic carry to placement goal when place phase ends (JTC / fake-sim lag)."""
        del ee_to_place_dist
        if not self._kinematic_carry or self._backend is None:
            return
        if not hasattr(self._backend, "set_prop_pose"):
            return
        backend_name = str(getattr(self._backend, "sensor_backend_name", "") or "").lower()
        if backend_name == "ros2_rviz":
            return
        if backend_name == "gazebo" and not self._gazebo_place_snap_enabled():
            return
        if backend_name != "gazebo":
            return
        props = getattr(self._backend, "_scene_prop_poses", None)
        if not isinstance(props, dict) or "target" not in props:
            return
        target_pos, _ = props["target"]
        half_z = float(getattr(self, "_waypoint_top_offsets", {}).get("source", np.zeros(3))[2])
        goal = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]) + half_z)
        try:
            self._backend.set_prop_pose("source", goal, (1.0, 0.0, 0.0, 0.0))
        except KeyError:
            pass

    def _maybe_engage_carry(self, obs: Observation, dist: float) -> None:
        phase = self._phases[self._phase_idx]
        if phase.spec.kind != "reach" or "grasp" not in phase.spec.name.lower():
            return
        gate = self._settle_dist * 1.5
        backend_name = str(getattr(self._backend, "sensor_backend_name", "") or "").lower()
        if backend_name == "gazebo" and self._effective_grasp_detection(obs) == "distance":
            gate = max(gate, 0.12)
        if dist >= gate:
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

    def _sensor_objects_ready(self, obs: Observation) -> bool:
        objects = getattr(obs, "objects", None) or {}
        return "source" in objects and "target" in objects

    def get_action(self, obs: Observation) -> Action:
        self._observe_sensor_health(obs)
        if self._sensor_only and not self._sensor_objects_ready(obs):
            q_hold = np.asarray(obs.joint_positions, dtype=np.float32).reshape(-1)
            if q_hold.shape[0] != self._home.shape[0]:
                q_hold = self._home.copy()
            return Action(joint_positions=q_hold, gripper=self._gripper_state)
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

        ee = ee_position_from_obs(obs)
        if phase.ee_target is not None:
            dist = float(np.linalg.norm(phase.ee_target - ee))
            if phase.spec.engage_carry or "grasp" in phase.spec.name.lower():
                self._maybe_engage_carry(obs, dist)
            self._maybe_drop_carry(obs)

        gripper_cmd = phase.gripper_value()
        if gripper_cmd is not None:
            self._gripper_state = float(gripper_cmd)
            if gripper_cmd >= 0.5 and phase.spec.kind == "gripper":
                self._carrying = True
            elif gripper_cmd < 0.5 and phase.spec.kind == "gripper":
                self._carrying = False
                if self._backend is not None and hasattr(self._backend, "set_grasp_prop"):
                    self._backend.set_grasp_prop(None)

        backend_name = str(getattr(self._backend, "sensor_backend_name", "") or "").lower()
        if (
            backend_name == "gazebo"
            and self._kinematic_carry
            and self._sensor_only
            and phase.spec.name == "lift"
            and self._phase_step == 1
        ):
            self._carrying = True

        if self._carrying:
            self._sync_carried_object(ee)

        q_goal = phase.compute(obs, q)
        q_cmd = self._track_toward(q, q_goal)

        ready_to_advance = phase.settled(obs) or self._phase_step >= phase.max_steps
        if ready_to_advance and not self._place_phase_release_allowed(phase, obs):
            if self._phase_step < phase.max_steps:
                ready_to_advance = False
        if ready_to_advance:
            place_phase = "place" in phase.spec.name.lower()
            gazebo_snap_place = (
                place_phase
                and backend_name == "gazebo"
                and self._kinematic_carry
                and self._gazebo_place_snap_enabled()
            )
            release_carry = bool(
                phase.spec.release_carry or (place_phase and self._carrying) or gazebo_snap_place
            )
            if release_carry and phase.ee_target is not None:
                ee_now = ee_position_from_obs(obs)
                if self._carrying or phase.spec.release_carry:
                    self._sync_honest_carry_pose(ee_now)
                self._maybe_finalize_kinematic_place(
                    float(np.linalg.norm(phase.ee_target - ee_now))
                )
            if phase.spec.release_carry or (place_phase and self._carrying):
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
