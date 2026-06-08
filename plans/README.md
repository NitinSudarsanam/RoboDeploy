# RoboDeploy Strategic Plans

Twelve goal plans. Each ~one focused topic. Source: `BROAD_GOALS.md`.

| # | Goal | File | Effort | Tier |
|---|---|---|---|---|
| 1 | Cut representation boilerplate | [GOAL_01_REPRESENTATION_BOILERPLATE.md](GOAL_01_REPRESENTATION_BOILERPLATE.md) | ~100h | 1 |
| 2 | Build training loop | [GOAL_02_TRAINING_LOOP.md](GOAL_02_TRAINING_LOOP.md) | ~80h | 1 |
| 3 | Sensor → policy/task integration | [GOAL_03_SENSOR_INTEGRATION.md](GOAL_03_SENSOR_INTEGRATION.md) | ~40h | 1 |
| 4 | Teleop + data collection | [GOAL_04_TELEOP_DATA_COLLECTION.md](GOAL_04_TELEOP_DATA_COLLECTION.md) | ~50h | 2 |
| 5 | Sim2Real pipeline | [GOAL_05_SIM2REAL.md](GOAL_05_SIM2REAL.md) | ~60h | 2 |
| 6 | Backend parity (Isaac/Gazebo) | [GOAL_06_BACKEND_PARITY.md](GOAL_06_BACKEND_PARITY.md) | ~40h | 2 |
| 7 | Docs + scaffolder CLI | [GOAL_07_DOCS_SCAFFOLDER.md](GOAL_07_DOCS_SCAFFOLDER.md) | ~30h | 3 |
| 8 | Multi-robot + distribution | [GOAL_08_MULTIROBOT_DISTRIBUTION.md](GOAL_08_MULTIROBOT_DISTRIBUTION.md) | ~40h | 3 |
| 9 | Learned policy integration | [GOAL_09_LEARNED_POLICY.md](GOAL_09_LEARNED_POLICY.md) | ~25h | 3 |
| 10 | Observability + replay | [GOAL_10_OBSERVABILITY_REPLAY.md](GOAL_10_OBSERVABILITY_REPLAY.md) | ~30h | 3 |
| 11 | Benchmarks + eval harness | [GOAL_11_BENCHMARKS_EVAL.md](GOAL_11_BENCHMARKS_EVAL.md) | ~50h | 3 |
| 12 | Real-hw safety + recovery | [GOAL_12_SAFETY.md](GOAL_12_SAFETY.md) | ~30h | 3 |

Total ~575h.

## Dependency Graph (Build Order)

```
Goal 1 (Representation)
  ├─→ Goal 3 (Sensor integration)  ─→ Goal 4 (Teleop)
  ├─→ Goal 7 (Docs + scaffolder)
  ├─→ Goal 11 (Benchmarks — uses templates)
  └─→ Goal 6 (Backend parity — uses Scene IR)

Goal 2 (Training loop)
  ├─→ Goal 4 (Teleop — needs dataset adapters)
  ├─→ Goal 9 (Learned policy — needs TrainablePolicyBase)
  ├─→ Goal 11 (Benchmarks — runs trained policies)
  └─→ Goal 5 (Sim2Real — DR sweep uses VecEnv)

Goal 9 (Learned policy)
  └─→ Goal 12 (Safety — depends on SafetyFilter D2)

Goal 6 (Backend parity)
  └─→ Goal 12 (Safety — CollisionGuard needs SupportsContactQuery)

Goal 10 (Observability)
  ├─→ Goal 11 (Benchmarks — needs determinism, RunManifest)
  └─→ Goal 5 (Sim2Real — transfer metrics consume manifests)

Goal 8 (Distribution)
  └─→ depends on Goals 1, 2, 3 being release-ready
```

## Suggested Phasing

**Phase A (foundation, ~220h)**: Goals 1 + 3 + 7. Cuts user friction, unlocks sensor-aware policies.

**Phase B (learning, ~155h)**: Goals 2 + 4 + 9. Repo becomes a learning platform.

**Phase C (cross-sim + real, ~100h)**: Goals 6 + 5. Backend parity + sim2real validation.

**Phase D (operations, ~110h)**: Goals 10 + 12. Reproducibility + safety.

**Phase E (ecosystem, ~90h)**: Goals 11 + 8. Benchmarks + PyPI/Docker/plugins.

## Audit Source Documents

- `SENSOR_INTEGRATION_TODO.md`
- `REPRESENTATION_UPGRADE_PLAN.md`
- `BROAD_GOALS.md`
- `ARCHITECTURE.md`
- `CONTRACTS.md`
