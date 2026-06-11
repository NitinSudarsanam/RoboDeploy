"""Tutorial 02 (your first task) code block stays runnable + interface docstring audit."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class Tutorial02TaskTests(unittest.TestCase):
    def test_tutorial_code_block_compiles_and_builds_scene(self):
        doc = (REPO_ROOT / "docs/tutorials/02_your_first_task.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", doc, flags=re.DOTALL)
        self.assertTrue(blocks, "tutorial must contain a python code block")
        code = blocks[0]
        self.assertLessEqual(
            len([ln for ln in code.splitlines() if ln.strip()]), 30,
            "tutorial task must stay under 30 lines",
        )
        # Unique registry name so the test never collides with example tasks.
        code = code.replace('"my_kitchen_pick"', '"my_kitchen_pick_tutorial02_test"')
        namespace: dict = {}
        exec(compile(code, "tutorial_02_block", "exec"), namespace)
        task = namespace["MyKitchenPickTask"]()
        spec = task.scene_spec()
        prop_names = [p.name for p in spec.world.props]
        self.assertIn(task.source_name, prop_names)
        self.assertIn(task.target_name, prop_names)


class InterfaceDocstringTests(unittest.TestCase):
    def test_every_public_interface_class_has_docstring(self):
        missing = []
        for path in sorted((REPO_ROOT / "robodeploy/core/interfaces").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    if not ast.get_docstring(node):
                        missing.append(f"{path.name}:{node.name}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
