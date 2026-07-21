import json
import tempfile
import unittest
from pathlib import Path

from tools.aptus_audit.generate import generate_evidence_bundle


class EvidenceGenerationTests(unittest.TestCase):
    def test_generate_evidence_bundle_writes_consistent_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "legacy"
            output = root / "audit"
            source.mkdir()
            (source / "main.py").write_text(
                "from .missing import value\n",
                encoding="utf-8",
            )
            (source / "copy.py").write_text(
                "from .missing import value\n",
                encoding="utf-8",
            )

            summary = generate_evidence_bundle(source, output)

            expected_files = {
                "baseline-manifest.json",
                "inventory.jsonl",
                "duplicate-clusters.json",
                "version-families.json",
                "reference-map.json",
                "secret-scan.json",
                "classification.jsonl",
                "classification-summary.json",
            }
            self.assertTrue(
                expected_files.issubset(
                    {path.name for path in output.iterdir() if path.is_file()}
                )
            )
            self.assertEqual(summary["artifact_count"], 2)
            decisions = [
                json.loads(line)
                for line in (output / "classification.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(decisions), 2)
            self.assertEqual(
                sum(summary["dispositions"].values()),
                summary["artifact_count"],
            )
            self.assertEqual(
                json.loads((output / "classification-summary.json").read_text()),
                summary,
            )

    def test_generate_evidence_bundle_rejects_missing_source_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "audit"
            output.mkdir()
            marker = output / "inventory.jsonl"
            marker.write_text('{"preserve":"old"}\n', encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                generate_evidence_bundle(root / "missing", output)

            self.assertEqual(marker.read_text(), '{"preserve":"old"}\n')


if __name__ == "__main__":
    unittest.main()
