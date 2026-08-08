"""Tests for the unified perception API entry point."""

from __future__ import annotations

import unittest

if __package__:
    from . import main
else:
    import main


class PerceptionGatewayTest(unittest.TestCase):
    def test_routes_are_registered(self) -> None:
        paths = main.app.openapi()["paths"]
        self.assertIn("/perception/pick/locate", paths)
        self.assertIn("/perception/pick/locate/debug", paths)
        self.assertIn("/perception/pick/check", paths)
        self.assertIn("/perception/place/check", paths)
        self.assertIn("/perception/parse", paths)


if __name__ == "__main__":
    unittest.main()
