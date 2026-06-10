"""Headless Isaac Sim smoke test for self-hosted GPU runners.

Run with Isaac Sim Kit Python (not system python):
  /path/to/isaac-sim/python.sh scripts/isaacsim_headless_smoke.py

See docs/BACKEND_SETUP.md#isaac-sim-self-hosted-ci.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from isaacsim.simulation_app import SimulationApp  # type: ignore[import-not-found]
    except ImportError:
        print(
            "isaacsim.simulation_app not found — run with Isaac Sim Kit python "
            "(see docs/BACKEND_SETUP.md#isaac-sim-self-hosted-ci)",
            file=sys.stderr,
        )
        return 1

    simulation_app = SimulationApp(
        {"headless": True},
        experience="isaacsim.exp.base.python.kit",
    )
    try:
        from isaacsim.core.api.world import World  # type: ignore[import-not-found]

        world = World(stage_units_in_meters=1.0)
        world.reset()
        world.step(render=False)
        print("isaacsim_headless_smoke: OK (reset + 1 physics step)")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
