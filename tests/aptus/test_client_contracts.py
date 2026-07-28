from __future__ import annotations

import unittest

from tools.check_client_contracts import verify_client_contracts


class ClientContractTests(unittest.TestCase):
    def test_swift_desktop_contract_matches_checked_openapi(self) -> None:
        self.assertEqual(verify_client_contracts(), [])


if __name__ == "__main__":
    unittest.main()
