from __future__ import annotations

import unittest

from robodeploy.builtins import import_builtins


class BuiltinsImportTests(unittest.TestCase):
    def test_import_builtins_does_not_raise(self):
        import_builtins()


if __name__ == "__main__":
    unittest.main()
