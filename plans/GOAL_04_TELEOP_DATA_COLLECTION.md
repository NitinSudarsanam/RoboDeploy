# Goal 4 — Teleop + Data Collection

**Priority**: Tier 2. **Effort**: ~50h. **Touches**: imitation learning, demo collection.

## Problem

Imitation learning story = dead without operator input. Today:
- Zero teleop interfaces. No keyboard, spacemouse, gello, VR, gamepad.
- No web GUI.
- ROS2 backend has no Twist/Joy subscriber.
- MuJoCo viewer = passive (no mouse → IK → action translation).
- `examples/so101/*` runs sinusoid policy, not operator input.
- Recording (`DemoSession`) only captures hardcoded policies.

## Current State (Audit)

### Recording infrastructure (works)
- `robodeploy/demo_recording.py:34-59` — `DemoRecorder` collects `DemoFrame` (obs, action, reward, done).
- `robodeploy/demo_recording.py:70-89` — `DemoSession` wraps env, auto-records `step()`.
- Export: `export_demo_jsonl()`, `export_demo_hdf5()`.
- Replay: `iter_replay_actions()`.
- Schema in `_to_jsonable()`: full Observation + Action serialized as JSON.

### Missing
- Zero grep hits for `keyboard`, `joystick`, `gamepad`, `pygame`, `spacemouse`, `vr`, `gello`, `pynput`.
- No FastAPI / WebSocket teleop server.
- No MuJoCo passive viewer mouse-IK.
- No ROS2 `geometry_msgs/Twist` or `sensor_msgs/Joy` subscriber.
- No `robodeploy replay` CLI subcommand.
- No RLDS / LeRobot dataset export.

---

## Deliverables

### D1. TeleopInputDevice Interface — `robodeploy/teleop/base.py` (NEW, ~100 lines)

```python
from abc import ABC, abstractmethod

@dataclass
class TeleopCommand:
    """Device-agnostic operator input."""
    delta_position: np.ndarray | None = None      # [3] m  (EE delta in tool/base frame)
    delta_orientation_rpy: np.ndarray | None = None  # [3] rad  (incremental euler)
    delta_joint_positions: np.ndarray | None = None  # [dof] rad
    gripper_command: float | None = None           # 0=open, 1=close
    button_pressed: dict[str, bool] = field(default_factory=dict)  # named buttons
    record_toggle: bool = False
    reset_episode: bool = False
    e_stop: bool = False

class ITeleopDevice(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def poll(self) -> TeleopCommand | None: ...
    @abstractmethod
    def stop(self) -> None: ...

    @property
    def is_alive(self) -> bool: return True
```

### D2. KeyboardTeleop — `robodeploy/teleop/keyboard.py` (NEW, ~200 lines)

Cross-platform via `pynput` (preferred) with `pygame` fallback.

```python
class KeyboardTeleop(ITeleopDevice):
    """WASD-style EE control + extras.

    Default bindings (configurable):
      W/S: +/- X    A/D: +/- Y    Q/E: +/- Z
      I/K: +/- pitch  J/L: +/- yaw  U/O: +/- roll
      Space: toggle gripper
      R: reset episode
      Tab: toggle recording
      Esc: e-stop
      [/]: decrease/increase step size
    """
    def __init__(self, *, step_position=0.01, step_orientation=0.05, bindings: dict | None = None): ...
    def start(self): ...        # spawn listener thread (pynput.keyboard.Listener)
    def poll(self) -> TeleopCommand: ...
    def stop(self): ...
```

### D3. SpaceMouseTeleop — `robodeploy/teleop/spacemouse.py` (NEW, ~250 lines)

Uses `pyspacemouse` or `pyspnav` (HIDAPI). 6-DOF analog.

```python
class SpaceMouseTeleop(ITeleopDevice):
    """3Dconnexion SpaceMouse / SpaceNavigator (6-DOF)."""
    def __init__(self, *, deadzone=0.05, scale_position=0.002, scale_orientation=0.01,
                 button_map: dict[int, str] | None = None): ...
    def start(self):
        import pyspacemouse
        if not pyspacemouse.open(callback=self._on_state, button_callback=self._on_button):
            raise RuntimeError("SpaceMouse not detected")
    def poll(self) -> TeleopCommand: ...
    def stop(self): pyspacemouse.close()
```

### D4. GamepadTeleop — `robodeploy/teleop/gamepad.py` (NEW, ~250 lines)

Xbox/PS controller via `pygame.joystick` or `evdev` (Linux).

```python
class GamepadTeleop(ITeleopDevice):
    """Two-stick + triggers + buttons.

    Default bindings:
      Left stick X/Y → EE Y/X
      Right stick X/Y → EE yaw/pitch
      LB/RB → EE Z down/up
      LT → close gripper (analog)
      RT → open gripper (analog)
      A → reset episode
      B → e-stop
      X → toggle record
    """
    def __init__(self, *, joystick_index=0, deadzone=0.1, scale_position=0.005, scale_orientation=0.05): ...
```

### D5. MuJoCoMouseIKTeleop — `robodeploy/teleop/mujoco_mouse.py` (NEW, ~300 lines)

Drag EE marker in MuJoCo `passive_viewer`. Mouse delta → 3D world → IK → joint command.

```python
class MuJoCoMouseIKTeleop(ITeleopDevice):
    """Drag the wrist marker in the MuJoCo viewer to teleoperate."""
    def __init__(self, *, backend: MuJoCoBackend, ee_body_name: str = "wrist_link"):
        self._backend = backend
        self._viewer = mujoco.viewer.launch_passive(...)
        self._target_pos = None
        self._target_quat = None
    def start(self): ...      # install mouse callback that updates _target_pos on drag
    def poll(self) -> TeleopCommand:
        if self._target_pos is None: return None
        # Compute delta from current EE pose
        return TeleopCommand(delta_position=..., gripper_command=...)
```

### D6. VRTeleop (OpenXR) — `robodeploy/teleop/vr.py` (NEW, ~300 lines, optional)

Stubbed; require `pyopenxr`. Quest controller pose → EE pose mapping. Out of MVP — note for future.

```python
class VRTeleop(ITeleopDevice):
    """OpenXR-based VR controller teleop. Requires Quest/Index + pyopenxr."""
    def __init__(self, *, controller="right", scale_position=1.0): ...
```

### D7. ROS2 Twist/Joy Bridge — `robodeploy/teleop/ros2_bridge.py` (NEW, ~200 lines)

Subscribes to standard ROS2 topics so external ROS teleop nodes (`teleop_twist_keyboard`, joysticks) feed RoboDeploy.

```python
class Ros2TwistTeleop(ITeleopDevice):
    """Subscribe to /cmd_vel (geometry_msgs/Twist) for EE command."""
    def __init__(self, *, topic="/cmd_vel", scale_position=0.01, scale_orientation=0.05): ...

class Ros2JoyTeleop(ITeleopDevice):
    """Subscribe to /joy (sensor_msgs/Joy) for analog input."""
    def __init__(self, *, topic="/joy", axes_map: dict[str, int] | None = None, button_map: dict[str, int] | None = None): ...
```

### D8. TeleopController — `robodeploy/teleop/controller.py` (NEW, ~250 lines)

Wraps `ITeleopDevice` + IK solver into a `IPolicy` so any env can be teleoperated transparently.

```python
@register_policy("teleop")
class TeleopPolicy(PolicyBase):
    """Adapts an ITeleopDevice to PolicyBase. Computes Action via IK when device gives EE delta."""

    def __init__(self, *, device: ITeleopDevice, action_space: ActionSpace = ActionSpace.JOINT_POS,
                 ik_solver: IKSolver | None = None, default_action: Literal["hold","zero"] = "hold",
                 max_step_position_m: float = 0.05, max_step_orientation_rad: float = 0.1):
        ...
    def bind_runtime(self, backend, description): self._ik = self._ik or build_ik(backend, description)
    def reset(self, obs): self._last_action = None
    def get_action(self, obs: Observation) -> Action:
        cmd = self._device.poll()
        if cmd is None: return self._last_action or self._hold(obs)
        if cmd.e_stop: raise SafetyError("operator e-stop")
        if cmd.delta_position is not None or cmd.delta_orientation_rpy is not None:
            target_pos, target_quat = self._integrate_ee(obs, cmd)
            q = self._ik.solve(target_pos, target_quat, obs.joint_positions)
            action = Action(joint_positions=q, gripper=cmd.gripper_command)
        elif cmd.delta_joint_positions is not None:
            q = obs.joint_positions + cmd.delta_joint_positions
            action = Action(joint_positions=q, gripper=cmd.gripper_command)
        self._last_action = action
        return action
```

### D9. RecordingSession + Hot-Keys — extend `robodeploy/demo_recording.py`

```python
class InteractiveDemoSession(DemoSession):
    """DemoSession that listens to TeleopCommand record_toggle / reset_episode hot-keys."""
    def __init__(self, env, policy, *, output_dir: str, format: Literal["jsonl","hdf5","lerobot"] = "jsonl"):
        ...
    def run(self):
        obs, info = self._env.reset()
        recording = False
        while True:
            action = self._policy.get_action(obs)
            obs, reward, done, info = self._env.step(action)
            cmd = getattr(self._policy._device, "last_command", None)
            if cmd and cmd.record_toggle: recording = not recording; print(f"Recording: {recording}")
            if cmd and cmd.reset_episode: self._save_episode(); obs, info = self._env.reset()
            if recording: self._recorder.add(obs, action, reward, done)
            if done: self._save_episode(); obs, info = self._env.reset()
```

### D10. Dataset Exporters — `robodeploy/dataset_export.py` (EXTEND)

```python
def export_to_lerobot(recorder: DemoRecorder, *, repo_id: str, fps: int = 30, push_to_hub: bool = False): ...
def export_to_rlds(recorder: DemoRecorder, *, output_dir: str): ...
def export_to_robomimic(recorder: DemoRecorder, *, output_path: str): ...
```

### D11. Replay CLI — `robodeploy/cli.py` (EXTEND)

```bash
robodeploy replay demo.jsonl --preset kuka_pick_mujoco
robodeploy replay demo.jsonl --preset kuka_pick_mujoco --speed 0.5 --pause-at-step 100
robodeploy teleop --preset kuka_pick_mujoco --device keyboard --record demos/episode_001.jsonl
robodeploy teleop --preset so101_real --device spacemouse --record so101_demos/
```

### D12. Examples
- `examples/teleop_keyboard_kuka.py` — keyboard teleop on MuJoCo Kuka, record JSONL.
- `examples/teleop_spacemouse_franka.py` — SpaceMouse teleop on MuJoCo Franka.
- `examples/teleop_so101_real.py` — SpaceMouse / gamepad teleop on real SO-101.
- `examples/replay_demo.py` — load + replay a JSONL demo.

### D13. (Optional, stretch) Web Teleop UI — `robodeploy/teleop/web/` (NEW, ~600 lines)

FastAPI + WebSocket server hosting:
- Video stream of `obs.rgb` (mjpeg over HTTP).
- WebSocket sink for browser-side keyboard/gamepad events.
- Browser-side JS captures key/gamepad → posts `TeleopCommand` → server feeds `TeleopPolicy`.

Defer to phase 4.5 if time permits.

---

## Phased Rollout

### Phase 4.1 — Interface + Keyboard (~10h)
- D1 ITeleopDevice + TeleopCommand.
- D2 KeyboardTeleop (pynput primary, pygame fallback).
- D8 TeleopPolicy with IK integration.
- D9 InteractiveDemoSession.
- D12 examples/teleop_keyboard_kuka.py.
- `tests/test_keyboard_teleop.py` (mock pynput).

### Phase 4.2 — SpaceMouse + Gamepad (~12h)
- D3 SpaceMouseTeleop (pyspacemouse).
- D4 GamepadTeleop (pygame.joystick + evdev).
- D12 examples/teleop_spacemouse_franka.py.
- `tests/test_spacemouse_teleop.py`, `tests/test_gamepad_teleop.py` (mocked HID).

### Phase 4.3 — MuJoCo Mouse-IK + ROS2 Bridge (~10h)
- D5 MuJoCoMouseIKTeleop.
- D7 Ros2TwistTeleop + Ros2JoyTeleop.
- `tests/test_mujoco_mouse_teleop.py`, `tests/test_ros2_teleop_bridge.py`.

### Phase 4.4 — Dataset exports + Replay CLI (~10h)
- D10 LeRobot + RLDS + Robomimic exporters.
- D11 `robodeploy replay/teleop` CLI subcommands.
- D12 examples/replay_demo.py.
- `tests/test_lerobot_export.py`, `tests/test_replay_cli.py`.

### Phase 4.5 (stretch) — Web UI (~8h)
- D13 FastAPI + WebSocket server.
- Browser-side JS keyboard + gamepad capture.
- Single-user only; multi-operator deferred.

---

## Acceptance Criteria

- [ ] `robodeploy teleop --device keyboard --preset kuka_pick_mujoco` lets user drive arm via WASD.
- [ ] SpaceMouse 6-DOF input maps to EE delta cleanly (linear scale, deadzone respected).
- [ ] Recorded episodes round-trip: `teleop --record` → `replay` reproduces motion.
- [ ] LeRobot export produces dataset loadable by `lerobot/datasets.LeRobotDataset`.
- [ ] ROS2 `teleop_twist_keyboard` → `/cmd_vel` → `Ros2TwistTeleop` controls sim arm.
- [ ] All teleop devices implement `ITeleopDevice` and pass `tests/test_teleop_contract.py`.
- [ ] Hot-key recording toggle (`Tab` default) + episode reset (`R` default) work in keyboard demo.
- [ ] CLI `robodeploy replay demo.jsonl` plays back at configurable speed with pause/step.

## Dependencies

- `pynput>=1.7` (keyboard) — primary
- `pygame>=2.5` (joystick fallback)
- `pyspacemouse>=1.1` (SpaceMouse, optional)
- `evdev` (Linux gamepad, optional)
- `pyopenxr` (VR, optional, deferred)
- `fastapi`, `uvicorn`, `websockets` (web UI, optional)
- `lerobot`, `tensorflow-datasets`, `rlds` (export adapters)

Add `[project.optional-dependencies] teleop = [pynput, pygame, pyspacemouse]`.

## Risks

- **OS-specific keyboard hooks**: pynput requires accessibility permissions on macOS. Mitigation: doc + pygame fallback.
- **SpaceMouse driver install**: pyspacemouse needs `libspnav` (Linux) or HID drivers. Mitigation: pre-flight CLI check + clear install instructions.
- **IK singularity during teleop**: rapid hand motion past singularity → IK fail. Mitigation: cartesian clamp + DLS damped IK + soft retry to last valid pose.
- **Real-hw safety during teleop**: operator command exceeds joint limits. Mitigation: SafetyFilter clamp (Goal 12) + slew rate.

## Out of Scope

- Bimanual teleop (two operators, two arms). Future.
- Gello driver. Hardware-specific; user contributes.
- Mocap full-body. External pipeline.
- Force/haptic feedback to operator. Future, requires hardware.
