import json
import hashlib
import unittest
from pathlib import Path

from tools.aptus_audit.inventory import inventory_tree


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = REPOSITORY_ROOT / "docs/audits/aptus-legacy"
LEGACY_ROOT = REPOSITORY_ROOT / "HyperTune"


class AuditOutputIntegrityTests(unittest.TestCase):
    def test_stored_inventory_matches_live_source_or_preserved_manifest(self) -> None:
        stored = [
            json.loads(line)
            for line in (AUDIT_ROOT / "inventory.jsonl").read_text().splitlines()
        ]
        stored_identity = [
            (item["path"], item["sha256"], item["size_bytes"], item["kind"])
            for item in stored
        ]
        if LEGACY_ROOT.is_dir():
            current = inventory_tree(LEGACY_ROOT)
            current_identity = [
                (item["path"], item["sha256"], item["size_bytes"], item["kind"])
                for item in current
            ]
            self.assertEqual(current_identity, stored_identity)
        else:
            baseline = json.loads((AUDIT_ROOT / "baseline-manifest.json").read_text())
            digest = hashlib.sha256()
            for item in stored:
                digest.update(item["path"].encode("utf-8"))
                digest.update(b"\0")
                digest.update((item["sha256"] or "").encode("ascii"))
                digest.update(b"\0")
                digest.update(str(item["size_bytes"]).encode("ascii"))
                digest.update(b"\n")
            self.assertEqual(baseline["file_count"], len(stored))
            self.assertEqual(baseline["manifest_sha256"], digest.hexdigest())

    def test_classification_covers_every_artifact_once(self) -> None:
        inventory_paths = {
            json.loads(line)["path"]
            for line in (AUDIT_ROOT / "inventory.jsonl").read_text().splitlines()
        }
        decisions = [
            json.loads(line)
            for line in (AUDIT_ROOT / "classification.jsonl").read_text().splitlines()
        ]
        decision_paths = [item["path"] for item in decisions]

        self.assertEqual(len(decision_paths), len(set(decision_paths)))
        self.assertEqual(set(decision_paths), inventory_paths)
        self.assertTrue(
            all(
                item["disposition"] in {"KEEP", "ADAPT", "ARCHIVE", "DISCARD"}
                for item in decisions
            )
        )

    def test_classification_summary_matches_decision_ledger(self) -> None:
        decisions = [
            json.loads(line)
            for line in (AUDIT_ROOT / "classification.jsonl").read_text().splitlines()
        ]
        summary = json.loads((AUDIT_ROOT / "classification-summary.json").read_text())
        disposition_counts = {
            disposition: sum(item["disposition"] == disposition for item in decisions)
            for disposition in {"KEEP", "ADAPT", "ARCHIVE", "DISCARD"}
            if any(item["disposition"] == disposition for item in decisions)
        }

        self.assertEqual(summary["artifact_count"], len(decisions))
        self.assertEqual(summary["dispositions"], disposition_counts)
        self.assertEqual(
            summary["adapt_candidates"],
            [item["path"] for item in decisions if item["disposition"] == "ADAPT"],
        )
        baseline = json.loads((AUDIT_ROOT / "baseline-manifest.json").read_text())
        self.assertEqual(
            summary["baseline_manifest_sha256"],
            baseline["manifest_sha256"],
        )

    def test_generated_bundle_manifest_matches_current_evidence(self) -> None:
        manifest = json.loads(
            (AUDIT_ROOT / "generated-bundle-manifest.json").read_text()
        )

        for filename, expected_hash in manifest["files"].items():
            with self.subTest(filename=filename):
                actual_hash = hashlib.sha256(
                    (AUDIT_ROOT / filename).read_bytes()
                ).hexdigest()
                self.assertEqual(actual_hash, expected_hash)

    def test_extraction_ledger_accounts_for_every_adapt_candidate(self) -> None:
        summary = json.loads((AUDIT_ROOT / "classification-summary.json").read_text())
        ledger = (AUDIT_ROOT / "extraction-ledger.md").read_text(encoding="utf-8")

        missing = [
            path for path in summary["adapt_candidates"] if f"`{path}`" not in ledger
        ]
        self.assertEqual(missing, [])

    def test_required_human_reports_exist_and_are_substantive(self) -> None:
        required = [
            "executive-summary.md",
            "hidden-gems.md",
            "failure-and-risk-register.md",
            "static-typescript.md",
            "static-python.md",
            "provenance-report.md",
            "sandbox-summary.md",
            "architecture-options.md",
        ]

        for filename in required:
            with self.subTest(filename=filename):
                report = AUDIT_ROOT / filename
                self.assertTrue(report.is_file())
                self.assertGreater(len(report.read_text(encoding="utf-8")), 500)


if __name__ == "__main__":
    unittest.main()
