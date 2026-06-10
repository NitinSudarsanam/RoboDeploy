# Goal 6 — Backend Parity (IsaacSim, Gazebo)

**Priority**: Tier 2. **Effort**: ~40h. **Touches**: cross-sim credibility.

## Problem

MuJoCo = production-ready. IsaacSim + Gazebo = partial:

### IsaacSim gaps
- `backends/sim/isaacsim/backend.py:121` raises `NotImplementedError("USD import path is not implemented yet")`.
- `backend.py:601` hardcodes `qfrc = np.zeros(...)` — joint efforts unobservable.
- `backend.py:603-607` hardcodes `ee_pos = zeros`, `ee_quat = (1,0,0,0)` — EE state unobservable.
- `backend.py:351` warns + skips heightfield terrain.
- No `isaacsim_imu.py` sensor.
- Capsule, heightfield, plane geoms unsupported (`backend.py:307-325`).
- Physics tuning = gravity only (`backend.py:179-185`).
- Multi-robot = shim (forces single-robot).
- No live CI (mocked only).

### Gazebo gaps
- `backends/sim/gazebo/scene_builder.py:47-80` — sphere/cylinder/plane only. Mesh + capsule missing.
- No grasp weld API (unlike MuJoCo `attach_grasp_welds`).
- No backend contact query (`has_prop_contact` missing).
- Procedural terrain absent (heightfield works via SDF heightmap only).
- Lighting limited to directional + sphere (no preset library yet).

### ROS2 Real gaps
- `backends/real/ros2/controllers/joint_velocity.py:8` — `NotImplementedError`.
- `backends/real/ros2/controllers/joint_effort.py:8` — `NotImplementedError`.
- `backends/real/ros2/controllers/gripper.py:8` — `NotImplementedError`.
- `backend.py:372-375` — `get_prop_pose()` `NotImplementedError` (needs perception).
- `backend.py:377-383` — `teleport_object()` `NotImplementedError`.

## Current State (Audit)

Feature matrix:

| Capability | MuJoCo | IsaacSim | Gazebo | ROS2 Real |
|---|---|---|---|---|
| Single-agent | ✓ | ✓ | ✓ | ✗ (multi only) |
| Multi-robot | shim | shim | ✓ | ✓ |
| JOINT_POS | ✓ | ✓ | ✓ | ✓ |
| JOINT_TORQUE | ✓ | ✗ | ✗ | ✗ (stub) |
| JOINT_VEL | n/a (config) | ✗ | ✗ | ✗ (stub) |
| Scene props | ✓ | ✓ | ? | Limited |
| Capsule geom | ✓ | ✗ | ✗ | n/a |
| Heightfield | ✓ | ✗ | ✓ | n/a |
| Mesh geom | ✓ | ✓ | ✗ | n/a |
| Grasp welds | ✓ | ✗ | ✗ | ✗ |
| Contact query | ✓ | ✗ | ✗ | ✗ |
| EE state in obs | ✓ | ✗ (zeros) | ✓ | ✓ |
| Joint efforts in obs | ✓ | ✗ (zeros) | ✓ | ✓ |
| Camera | ✓ | ✓ | ✓ | ✓ |
| FT | ✓ | ✓ | ✓ | ✓ |
| IMU | ✓ | ✗ | ✓ | ✗ |
| Physics tuning | full | gravity only | ✗ | n/a |
| Live CI | ✓ | ✗ | ✓ | ✓ |

---

## Deliverables

### A. IsaacSim Parity (~20h)

### D1. EE State from Articulation — fix `backend.py:603-607`

```python
# robodeploy/backends/sim/isaacsim/backend.py
def _build_obs(self) -> Observation:
    art = self._articulation
    ee_link_idx = art.get_link_index(self._ee_link_name)  # discovered from description
    link_world_pose = art.get_link_world_poses(indices=[ee_link_idx])  # Isaac API
    ee_pos = link_world_pose.positions[0].cpu().numpy()      # [3]
    ee_quat = link_world_pose.orientations[0].cpu().numpy()  # [4] wxyz
    ee_lin_vel, ee_ang_vel = art.get_link_velocities(indices=[ee_link_idx])
    ee_lin_vel = ee_lin_vel[0].cpu().numpy()
    ee_ang_vel = ee_ang_vel[0].cpu().numpy()
    ...
```

Discover `ee_link_name` from `RobotDescription.ee_link` config; fallback to last link if unset.

### D2. Joint Efforts from TensorAPI — fix `backend.py:601`

```python
# Isaac Sim exposes computed joint forces via ArticulationView in 4.1+
art_view = self._art_view  # ArticulationView
qfrc = art_view.get_measured_joint_efforts().cpu().numpy().reshape(-1)  # [dof]
```

Fallback to zeros + emit warning if API unavailable.

### D3. IsaacSim IMU — `robodeploy/sensors/imu/sim/isaacsim_imu.py` (NEW, ~150 lines)

```python
class IsaacSimIMUSensor(IMUBase):
    def __init__(self, *, mount_link: str, accelerometer_noise_std=0.0, gyro_noise_std=0.0):
        ...

    def initialize(self, backend: "IsaacSimBackend"):
        # Use isaacsim.sensors.IMUSensor or compute from link state derivatives
        from isaacsim.sensors.physics import IMUSensor
        self._sensor = IMUSensor(
            prim_path=f"/World/Robot/{mount_link}/IMU",
            frequency=200,
            translation=np.zeros(3),
            orientation=np.array([1, 0, 0, 0]),
        )

    def read(self) -> SensorData:
        frame = self._sensor.get_current_frame()
        return SensorData(
            payload={
                "linear_acceleration": frame["lin_acc"],
                "angular_velocity": frame["ang_vel"],
            },
            timestamp=time.time(),
            status=SensorStatus.OK,
        )
```

Register in pair at `robodeploy/sensors/imu/sim/mujoco_imu.py:82-87`:

```python
register_sensor_pair(
    name="wrist_imu",
    sim={
        "mujoco": MuJoCoIMUSensor,
        "isaacsim": IsaacSimIMUSensor,    # NEW
        "gazebo": Ros2ImuSensor,
    },
    real=Ros2ImuSensor,
)
```

### D4. Capsule + Heightfield + Plane in IsaacSim — extend `backend.py:307-325`

```python
def _add_prop_geom(self, name: str, geom: UnifiedGeom, pose: Pose3D):
    if geom.kind == "capsule":
        # USD Capsule prim
        prim = UsdGeom.Capsule.Define(stage, f"/World/{name}")
        prim.GetRadiusAttr().Set(geom.size[0])
        prim.GetHeightAttr().Set(geom.size[1])
    elif geom.kind == "heightfield":
        # USD UsdGeom.Mesh built from heightfield grid
        ...
    elif geom.kind == "plane":
        prim = UsdGeom.Plane.Define(stage, f"/World/{name}")
        prim.GetWidthAttr().Set(geom.size[0])
        prim.GetLengthAttr().Set(geom.size[1])
```

### D5. IsaacSim USD Import — fix `backend.py:121`

```python
# Replace NotImplementedError with USD import path:
if asset_path.suffix == ".usd" or asset_path.suffix == ".usda":
    add_reference_to_stage(usd_path=str(asset_path), prim_path="/World/Robot")
elif asset_path.suffix == ".urdf":
    # Existing URDF importer path
    self._urdf_to_usd_and_add(asset_path)
```

Use `omni.kit.commands.execute("URDFParseAndImportFile", ...)` for URDF → USD conversion.

### D6. IsaacSim Physics Tuning — extend `backend.py:179-185`

```python
def set_physics_params(self, **kwargs):
    if "gravity" in kwargs: self._world.set_gravity(kwargs["gravity"])
    if "friction" in kwargs:
        for prop_path in self._prop_paths:
            physx_prim = self._stage.GetPrimAtPath(prop_path)
            PhysxSchema.PhysxMaterialAPI.Apply(physx_prim).GetDynamicFrictionAttr().Set(kwargs["friction"])
    if "restitution" in kwargs: ...
    if "damping" in kwargs: ...
```

### D7. IsaacSim Multi-Robot — fix `backend.py:79-96`

Drop shim. Implement `initialize_multi(robots, scene, shared_sensors)` properly:
- Create separate `/World/Robot_<id>` prim subtrees.
- Track per-robot ArticulationView.
- `step_multi(actions)` dispatches per-robot.
- `get_obs_multi()` returns list ordered by robot_id.

### D8. IsaacSim Live CI — extend `.github/workflows/test.yml`

Add `isaacsim-smoke` job using `nvcr.io/nvidia/isaac-sim:4.1.0` container. Runs `tests/test_isaacsim_smoke.py` with `--allow-root --headless`. Mark `continue-on-error: true` initially.

---

### B. Gazebo Parity (~10h)

### D9. Mesh Geom in Gazebo SDF — extend `backends/sim/gazebo/scene_builder.py:47-80`

```python
def _build_geom_sdf(self, geom: UnifiedGeom, variants: dict[str, str]) -> str:
    if geom.kind == "mesh":
        mesh_uri = variants.get("urdf") or geom.mesh_path  # urdf typically wraps mesh
        return f'''
        <geometry>
          <mesh>
            <uri>{mesh_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>'''
```

### D10. Capsule via Compound in Gazebo

SDF lacks capsule primitive. Approximate with cylinder + 2 hemispheres:

```python
def _build_capsule_compound_sdf(self, name, radius, length, pose) -> str:
    """Two spheres + cylinder, all rigidly fixed."""
    return f'''
    <model name="{name}">
      <link name="body">
        <collision name="cyl"><geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry></collision>
        <collision name="cap_top"><pose>0 0 {length/2} 0 0 0</pose><geometry><sphere><radius>{radius}</radius></sphere></geometry></collision>
        <collision name="cap_bot"><pose>0 0 {-length/2} 0 0 0</pose><geometry><sphere><radius>{radius}</radius></sphere></geometry></collision>
        ... (visual mirrors)
      </link>
    </model>'''
```

### D11. Gazebo Contact Query — `robodeploy/backends/sim/gazebo/contact.py` (NEW)

Use Gazebo `contacts` topic via gz-transport:

```python
class GazeboContactMonitor:
    def __init__(self, gz_transport_node):
        self._node = gz_transport_node
        self._contacts = []
        self._node.subscribe("contacts", self._on_contacts)

    def _on_contacts(self, msg): self._contacts = list(msg.contact)

    def has_contact(self, body_a: str, body_b: str | None = None) -> bool: ...

# Expose via backend:
class ROS2GazeboBackend:
    def has_prop_contact(self, prop_name: str) -> bool:
        return self._contact_monitor.has_contact(prop_name, body_b=self._ee_link)
```

### D12. Gazebo Grasp Weld Equivalent

Gazebo has no MJCF-style equality. Two approaches:
1. **Fixed joint runtime attach**: spawn temporary `gazebo::physics::Joint` between gripper + prop. Requires gz-sim plugin.
2. **Sim-side workaround**: kinematic follow mode (re-teleport prop to gripper pose each step) when grasp is asserted.

Pick (2) for parity; document (1) as future work.

```python
class ROS2GazeboBackend:
    def set_grasp_prop(self, prop_name: str | None, *, mode: str = "follow"):
        if mode != "follow":
            raise NotImplementedError("Gazebo backend only supports grasp mode='follow'")
        self._grasped_prop = prop_name
    def _post_step(self):
        if self._grasped_prop:
            ee_pose = self._get_ee_pose()
            self.set_prop_pose(self._grasped_prop, ee_pose.position, ee_pose.orientation)
```

### D13. Procedural Terrain in Gazebo

Add Perlin / ridge generators that emit SDF heightmap PNG + heightmap entity:

```python
class ProceduralTerrainGenerator:
    @staticmethod
    def perlin(size_m=(4,4), resolution=128, octaves=4, persistence=0.5) -> np.ndarray: ...
    @staticmethod
    def to_sdf_heightmap(heightfield: np.ndarray, out_dir: Path) -> str: ...
```

Same generators reused by MuJoCo backend (existing heightfield).

---

### C. ROS2 Real Parity (~10h)

### D14. JointVelocity Controller — fix `controllers/joint_velocity.py:8`

```python
class JointVelocityController(ControllerBase):
    def __init__(self, *, joint_names, command_topic: str = "/joint_velocity_controller/command", command_hz: float = 0.0):
        super().__init__(joint_names=joint_names, command_hz=command_hz)
        self._publisher = self._node.create_publisher(Float64MultiArray, command_topic, 10)
    def send_action(self, action: Action) -> None:
        if action.joint_velocities is None: raise ActionSpaceMismatch("expected joint_velocities")
        msg = Float64MultiArray(data=action.joint_velocities.tolist())
        self._publisher.publish(msg)
```

### D15. JointEffort Controller — fix `controllers/joint_effort.py:8`

Same shape as D14, publishing to `/effort_controllers/command`.

### D16. Gripper Controller — fix `controllers/gripper.py:8`

```python
class GripperController(ControllerBase):
    """Generic gripper: maps action.gripper [0,1] to action_msgs/GripperCommand or std_msgs/Float64."""
    def __init__(self, *, command_topic: str, command_type: Literal["gripper_command","float"] = "gripper_command"): ...
    def send_action(self, action: Action): ...
```

### D17. Perception-Driven get_prop_pose — `backend.py:372-375`

Don't raise. Optional perception source:

```python
def get_prop_pose(self, prop_name: str) -> tuple[np.ndarray, np.ndarray]:
    perception = getattr(self, "_perception_source", None)
    if perception is None:
        raise NotImplementedError(
            "ROS2RealBackend cannot infer pose for prop without a perception source. "
            "Inject via config: perception_source = 'vision' | 'mocap' | 'tf'."
        )
    return perception.get_pose(prop_name)
```

Plug in `TFPerceptionSource` (existing) + new `ColorBlobPerceptionSource` (uses Goal 3 D5).

### D18. Backend Capability Markers — `robodeploy/backends/capabilities.py`

Expose capability protocols for runtime feature negotiation:

```python
class SupportsGraspWeld(Protocol):
    def attach_grasp_welds(self, prop_names: list[str]) -> None: ...
    def set_grasp_prop(self, prop_name: str | None, *, mode: str) -> None: ...

class SupportsContactQuery(Protocol):
    def has_prop_contact(self, prop_name: str, *, other_body: str | None = None) -> bool: ...

class SupportsProceduralTerrain(Protocol):
    def set_terrain(self, kind: Literal["flat","heightfield","procedural"], **kwargs): ...

class SupportsPhysicsTuning(Protocol):
    def set_physics_params(self, **kwargs): ...
```

Policies + tasks check via `isinstance(backend, SupportsGraspWeld)` rather than backend-string sniffing.

### D19. Cross-Backend Parity Tests — `tests/test_backend_parity.py` (NEW)

Same `SceneIR` (from Goal 1) loaded into MuJoCo, IsaacSim, Gazebo; assertions:
- Prop count matches.
- Prop poses within 1mm tolerance.
- Step count for fixed-time rollout matches.
- Sensor names produced by `get_obs_multi()` identical.
- Capability protocols correctly exposed (MuJoCo implements SupportsGraspWeld; Gazebo doesn't, etc.).

---

## Phased Rollout

### Phase 6.1 — IsaacSim observation (~6h)
- D1 EE state from articulation.
- D2 joint efforts via TensorAPI.
- `tests/test_isaacsim_obs.py` (mocked Isaac module if no GPU CI).

### Phase 6.2 — IsaacSim sensors + geom (~8h)
- D3 IsaacSim IMU + register pair.
- D4 Capsule + heightfield + plane support.
- D5 USD direct import.

### Phase 6.3 — IsaacSim multi-robot + physics (~6h)
- D6 physics tuning expansion.
- D7 multi-robot.

### Phase 6.4 — Gazebo parity (~10h)
- D9 mesh, D10 capsule compound, D11 contact query, D12 grasp follow, D13 procedural terrain.

### Phase 6.5 — ROS2 Real controllers (~6h)
- D14, D15, D16 (velocity, effort, gripper).
- D17 perception-driven prop pose.

### Phase 6.6 — Capabilities + tests + CI (~4h)
- D18 capability protocols.
- D19 parity tests.
- D8 IsaacSim Docker CI job.

---

## Acceptance Criteria

- [ ] IsaacSim `obs.ee_position` non-zero on standard reach episode.
- [ ] IsaacSim `obs.joint_torques` non-zero when arm moves under gravity.
- [ ] `wrist_imu` sensor resolves to `IsaacSimIMUSensor` when backend=isaacsim.
- [ ] Capsule prop renders + collides in IsaacSim (visual + physics).
- [ ] `.usd` asset loads without `NotImplementedError`.
- [ ] IsaacSim multi-robot reach example runs 2 robots concurrently.
- [x] IsaacSim Docker CI job runs at least 1 smoke test (`test.yml` `isaacsim-smoke`; mocked import path, `continue-on-error: true`).
- [x] Gazebo mesh prop loads from URDF mesh URI.
- [x] Gazebo capsule compound renders + collides.
- [x] `Ros2GazeboBackend.has_prop_contact(...)` returns true when ee touches prop.
- [x] Gazebo grasp follow mode tracks prop to gripper pose.
- [x] Gazebo procedural Perlin terrain renders.
- [x] ROS2 `JointVelocityController.send_action(action)` publishes to controller topic (`tests/test_ros2_controllers_parity.py`).
- [x] `Action(joint_velocities=...)` accepted by ROS2 backend with velocity controller (`tests/test_ros2_controllers_parity.py::test_ros2_backend_supported_action_spaces_include_velocity_and_effort`).
- [x] Gripper controller maps gripper=1.0 to close command (`tests/test_ros2_controllers_parity.py::test_gripper_maps_close_command`).
- [x] `SupportsGraspWeld` protocol matches MuJoCo, doesn't match Gazebo, doesn't match IsaacSim (`tests/test_backend_parity.py`).
- [ ] `tests/test_backend_parity.py` SceneIR round-trip ≤ 1mm pose tolerance across MuJoCo/IsaacSim/Gazebo.

## Risks

- **IsaacSim API churn**: Kit / IsaacSim 4.x vs 5.x ArticulationView API changes. Mitigation: pin Isaac version in extras + version-guarded code paths.
- **Capsule compound mass distribution wrong**: 3-body compound inertia ≠ true capsule. Mitigation: override inertia tensor analytically.
- **Gazebo contacts topic format varies** (Garden vs Harmonic). Mitigation: detect version + branch.
- **No GPU in CI**: IsaacSim Docker requires NVIDIA runtime. Mitigation: gate Isaac CI on self-hosted runner OR mocked-only tests; mark `continue-on-error: true`.
- **Grasp follow vs weld physics divergence**: kinematic teleport breaks momentum. Mitigation: document as Gazebo limitation; recommend MuJoCo for contact-rich tasks.

## Out of Scope

- IsaacSim domain randomization (Isaac's own DR API). Use RoboDeploy's DR.
- IsaacSim ros2_bridge live sensor publish. Use sim sensors directly.
- Gazebo Classic (gazebo-classic, deprecated). Only gz-sim (Garden+).
- Webots / PyBullet backends. New backend pattern documented; community contribution.
