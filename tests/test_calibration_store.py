from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robodeploy.calibration.kinematic.linear import JointLinearMap, LinearKinematicCalibration
from robodeploy.calibration.store import CalibrationStore
from robodeploy.description.so101.calibration import JointCalibration, SO101Calibration


class CalibrationStoreTests(unittest.TestCase):
    def test_round_trip_kinematic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(root=Path(tmp))
            payload = {
                "format": "robodeploy-linear-kinematic-v1",
                "joints": [
                    {"name": "j0", "zero": 100.0, "scale": 650.0, "soft_min": -1.0, "soft_max": 1.0}
                ],
            }
            path = store.save("kinematic", payload, robot_id="so101")
            self.assertTrue(path.is_file())
            loaded = store.load("kinematic", robot_id="so101")
            self.assertEqual(loaded["joints"][0]["name"], "j0")
            entries = store.list_all()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["robot_id"], "so101")

    def test_round_trip_extrinsic_and_system_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CalibrationStore(root=Path(tmp))
            store.save(
                "extrinsic_wrist",
                {"position": [0.0, 0.0, 0.5], "orientation": [1.0, 0.0, 0.0, 0.0]},
                robot_id="franka",
            )
            store.save(
                "system_id",
                {"friction": {"joint_0": {"coulomb_Nm": 0.1}}, "payload_mass_kg": 0.2},
                robot_id="franka",
            )
            ext = store.load("extrinsic_wrist", robot_id="franka")
            sid = store.load("system_id", robot_id="franka")
            self.assertIn("position", ext)
            self.assertIn("payload_mass_kg", sid)

    def test_so101_linear_maps_round_trip(self):
        joints = tuple(
            JointCalibration(
                name=str(i),
                motor_id=i,
                zero_ticks=2048,
                ticks_per_rad=651.9,
                soft_min_rad=-1.0,
                soft_max_rad=1.0,
            )
            for i in range(1, 7)
        )
        cal = SO101Calibration(joints=joints)
        maps = cal.to_linear_kinematic_maps()
        restored = SO101Calibration.from_linear_maps(maps)
        self.assertEqual(len(restored.joints), 6)
        q = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        ticks = cal.to_ticks(q)
        q2 = restored.to_radians(ticks)
        self.assertAlmostEqual(float(q2[0]), 0.1, places=3)

    def test_linear_kinematic_fit(self):
        pairs = [
            ([100.0, 200.0], [0.0, 0.0]),
            ([165.19, 265.19], [0.1, 0.1]),
        ]
        cal = LinearKinematicCalibration([]).fit(pairs)
        can = cal.to_canonical([165.19, 265.19])
        self.assertAlmostEqual(float(can[0]), 0.1, places=2)


if __name__ == "__main__":
    unittest.main()
