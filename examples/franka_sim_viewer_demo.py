"""Franka Panda oscillation demo (structure-only).

Migrated out of `robodeploy.demos` into `examples/` during the architecture
migration. The legacy oscillation task was removed; the new architecture
expects tasks under `robodeploy.tasks.*` inheriting `TaskBase` and policies
under `robodeploy.policies.*` inheriting `PolicyBase`.

This file is a placeholder showing where the updated demo will live.
"""


def main() -> None:
    raise NotImplementedError(
        "Replace the legacy oscillation demo with a TaskBase/PolicyBase based example."
    )


if __name__ == "__main__":
    main()

