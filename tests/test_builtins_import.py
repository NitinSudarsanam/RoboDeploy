from __future__ import annotations

import unittest

from robodeploy.builtins import failed_builtin_imports, import_builtins


class BuiltinsImportTests(unittest.TestCase):
    def test_import_builtins_does_not_raise(self):
        import_builtins()

    def test_failed_builtin_imports_returns_list(self):
        failures = failed_builtin_imports()
        self.assertIsInstance(failures, list)


if __name__ == "__main__":
    unittest.main()
