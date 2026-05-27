from __future__ import annotations

import unittest

from robodeploy.tasks.manipulation.pick_place import PickPlaceTask
from robodeploy.tasks.randomization import DomainRandomizer, DomainRandomizerConfig, ObjectRandomConfig, RandomLevel


class _FakeBackend:
    is_real = False

    def __init__(self) -> None:
        self.teleports: list[tuple[str, tuple[float, float, float]]] = []

    def teleport_object(self, name: str, position):  # noqa: ANN001
        self.teleports.append((name, tuple(float(v) for v in position)))


class DomainRandomizationTests(unittest.TestCase):
    def test_pick_place_applies_pose_jitter_when_enabled(self):
        backend = _FakeBackend()
        task = PickPlaceTask(
            config={
                "randomize_objects": True,
                "pose_jitter_m": 0.1,
                "random_seed": 0,
            }
        )
        task.reset_fn(backend)
        self.assertTrue(backend.teleports)
        self.assertEqual(backend.teleports[0][0], "source")

    def test_domain_randomizer_skips_none_level(self):
        backend = _FakeBackend()
        DomainRandomizer(DomainRandomizerConfig(level=RandomLevel.NONE)).randomize(backend)
        self.assertEqual(backend.teleports, [])

    def test_domain_randomizer_teleports_configured_objects(self):
        backend = _FakeBackend()
        dr = DomainRandomizer(
            DomainRandomizerConfig(
                level=RandomLevel.LIGHT,
                seed=1,
                objects=[
                    ObjectRandomConfig(
                        object_name="cube",
                        position_center=(0.5, 0.0, 0.1),
                        position_range=(0.02, 0.02, 0.0),
                    )
                ],
            )
        )
        dr.randomize(backend)
        self.assertEqual(len(backend.teleports), 1)


if __name__ == "__main__":
    unittest.main()
