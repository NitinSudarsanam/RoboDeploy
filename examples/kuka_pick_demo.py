"""Kuka pick demo (structure-only).

Migrated from `robodeploy.demos` into `examples/` during the backend architecture
migration. The old `MujocoEngine` and `robots/` package are removed; the new
architecture uses `RobotDescription` + `BackendBase` backends instead.

This placeholder will become a `RoboEnv.make(robot="kuka", backend="mujoco", task="pick_place")`
example once `MuJoCoBackend` is implemented.
"""


def main() -> None:
    raise NotImplementedError(
        "MuJoCoBackend is currently a stub. Implement it, then use RoboEnv.make(...) here."
    )


if __name__ == "__main__":
    main()

