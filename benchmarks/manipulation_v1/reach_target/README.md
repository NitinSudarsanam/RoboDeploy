# reach_target (tier 1)

Joint-space reach with deterministic scripted baseline on the dummy backend.

- **Success**: primary joint within `success_tol` of `target_q`
- **Budget**: 300 steps
- **Canonical preset**: `preset_dummy.yaml` (CI / no sim), `preset_mujoco.yaml` (Kuka sinusoid smoke)
