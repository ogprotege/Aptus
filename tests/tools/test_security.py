import tempfile
import unittest
from pathlib import Path

from tools.aptus_audit.security import scan_secrets


class SecretScanTests(unittest.TestCase):
    def test_scan_reports_high_confidence_secret_without_exposing_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret = "AKIAABCDEFGHIJKLMNOP"
            (root / "config.py").write_text(
                f'AWS_ACCESS_KEY_ID = "{secret}"\n',
                encoding="utf-8",
            )

            findings = scan_secrets(root)

            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["path"], "config.py")
            self.assertEqual(findings[0]["line"], 1)
            self.assertEqual(findings[0]["rule_id"], "aws-access-key-id")
            self.assertNotIn(secret, findings[0]["masked_preview"])

    def test_scan_reports_huggingface_user_token_without_exposing_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret = "hf_" + ("A" * 34)
            (root / "train.py").write_text(
                f'HF_TOKEN = "{secret}"\n',
                encoding="utf-8",
            )

            findings = scan_secrets(root)

            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["path"], "train.py")
            self.assertEqual(findings[0]["line"], 1)
            self.assertEqual(findings[0]["rule_id"], "huggingface-token")
            self.assertEqual(findings[0]["severity"], "high")
            self.assertNotIn(secret, findings[0]["masked_preview"])

    def test_scan_ignores_placeholders_and_non_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env.example").write_text(
                "OPENAI_API_KEY=your_openai_api_key_here\n",
                encoding="utf-8",
            )
            (root / "notes.py").write_text(
                "hf_dataset=False\n"
                "from transformers import logging as hf_logging\n"
                'HF_TOKEN = "your_hf_token_here"\n',
                encoding="utf-8",
            )
            (root / "image.bin").write_bytes(b"\x00AKIAABCDEFGHIJKLMNOP")

            self.assertEqual(scan_secrets(root), [])


if __name__ == "__main__":
    unittest.main()
