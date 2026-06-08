from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SceneValidatorTests(unittest.TestCase):
    def test_duplicate_prop_is_error(self):
        from robodeploy.core.scene_ir import SceneIR, UnifiedGeom, UnifiedPropSpec
        from robodeploy.core.scene_validator import SceneValidator

        geom = UnifiedGeom(kind="box", size=(0.1, 0.1, 0.1))
        ir = SceneIR(
            props=(
                UnifiedPropSpec(name="dup", geometry=geom),
                UnifiedPropSpec(name="dup", geometry=geom),
            )
        )
        report = SceneValidator().validate(ir, "mujoco")
        self.assertFalse(report.ok)
        self.assertTrue(any("Duplicate" in i.message for i in report.issues))

    def test_capsule_on_gazebo_is_warning(self):
        from robodeploy.core.scene_ir import SceneIR, UnifiedGeom, UnifiedPropSpec
        from robodeploy.core.scene_validator import SceneValidator

        ir = SceneIR(props=(UnifiedPropSpec(name="cap", geometry=UnifiedGeom(kind="capsule", size=(0.01, 0.1))),))
        report = SceneValidator().validate(ir, "gazebo")
        self.assertTrue(report.ok)
        self.assertTrue(any(i.level == "warning" for i in report.issues))
