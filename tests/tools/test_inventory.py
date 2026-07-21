import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.aptus_audit.cli import generate_inventory_bundle
from tools.aptus_audit.inventory import (
    build_duplicate_clusters,
    build_version_families,
    inventory_tree,
    write_json,
    write_jsonl,
)


class InventoryTreeTests(unittest.TestCase):
    def test_inventory_tree_hashes_and_sorts_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "nested").mkdir()
            (root / "z.txt").write_text("same", encoding="utf-8")
            (root / "nested" / "a.txt").write_text("same", encoding="utf-8")
            (root / "empty.py").write_bytes(b"")

            records = inventory_tree(root)

            self.assertEqual(
                [record["path"] for record in records],
                ["empty.py", "nested/a.txt", "z.txt"],
            )
            self.assertEqual(records[0]["size_bytes"], 0)
            self.assertEqual(records[1]["sha256"], records[2]["sha256"])
            self.assertEqual(records[0]["kind"], "empty")
            self.assertEqual(records[1]["kind"], "text")

    @unittest.skipIf(os.name == "nt", "symlink behavior differs on Windows")
    def test_inventory_tree_records_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.txt").write_text("target", encoding="utf-8")
            (root / "link.txt").symlink_to("target.txt")

            records = inventory_tree(root)
            link = next(record for record in records if record["path"] == "link.txt")

            self.assertEqual(link["kind"], "symlink")
            self.assertEqual(link["symlink_target"], "target.txt")
            self.assertIsNone(link["sha256"])

    def test_duplicate_clusters_include_only_repeated_hashes(self) -> None:
        records = [
            {"path": "a.py", "sha256": "abc", "size_bytes": 10, "kind": "text"},
            {"path": "b.py", "sha256": "abc", "size_bytes": 10, "kind": "text"},
            {"path": "c.py", "sha256": "def", "size_bytes": 20, "kind": "text"},
            {"path": "link", "sha256": None, "size_bytes": 0, "kind": "symlink"},
        ]

        clusters = build_duplicate_clusters(records)

        self.assertEqual(
            clusters,
            [
                {
                    "cluster_id": "sha256:abc",
                    "sha256": "abc",
                    "size_bytes": 10,
                    "paths": ["a.py", "b.py"],
                }
            ],
        )

    def test_version_families_group_finder_and_v2_names(self) -> None:
        records = [
            {"path": "tests/test_optimizer.py"},
            {"path": "tests/test_optimizer 2.py"},
            {"path": "src/formulas/rank.ts"},
            {"path": "src/formulas/rank_v2.ts"},
            {"path": "unrelated.py"},
        ]

        families = build_version_families(records)

        self.assertEqual(
            families,
            [
                {
                    "family_id": "normalized:src/formulas/rank.ts",
                    "normalized_path": "src/formulas/rank.ts",
                    "paths": [
                        "src/formulas/rank.ts",
                        "src/formulas/rank_v2.ts",
                    ],
                },
                {
                    "family_id": "normalized:tests/test_optimizer.py",
                    "normalized_path": "tests/test_optimizer.py",
                    "paths": [
                        "tests/test_optimizer 2.py",
                        "tests/test_optimizer.py",
                    ],
                },
            ],
        )

    def test_writers_emit_stable_machine_readable_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jsonl_path = root / "inventory.jsonl"
            json_path = root / "duplicates.json"
            rows = [{"path": "a"}, {"path": "b"}]

            write_jsonl(jsonl_path, rows)
            write_json(json_path, {"clusters": rows})

            self.assertEqual(
                [json.loads(line) for line in jsonl_path.read_text().splitlines()],
                rows,
            )
            self.assertEqual(json.loads(json_path.read_text()), {"clusters": rows})

    def test_generate_inventory_bundle_writes_summary_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "legacy"
            output = root / "audit"
            source.mkdir()
            (source / "one.txt").write_text("same", encoding="utf-8")
            (source / "two.txt").write_text("same", encoding="utf-8")

            summary = generate_inventory_bundle(source, output)

            self.assertEqual(summary["file_count"], 2)
            self.assertEqual(summary["duplicate_cluster_count"], 1)
            self.assertEqual(summary["duplicate_file_count"], 2)
            self.assertEqual(summary["source_root"], str(source.resolve()))
            self.assertEqual(
                len((output / "inventory.jsonl").read_text().splitlines()),
                2,
            )
            self.assertEqual(
                json.loads((output / "duplicate-clusters.json").read_text())[
                    "clusters"
                ][0]["paths"],
                ["one.txt", "two.txt"],
            )
            self.assertEqual(
                json.loads((output / "baseline-manifest.json").read_text()),
                summary,
            )

    def test_generate_inventory_bundle_rejects_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "audit"
            output.mkdir()
            marker = output / "baseline-manifest.json"
            marker.write_text('{"preserve": true}\n', encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                generate_inventory_bundle(root / "missing", output)

            self.assertEqual(marker.read_text(), '{"preserve": true}\n')

    def test_generate_inventory_bundle_rejects_output_inside_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "legacy"
            source.mkdir()

            with self.assertRaises(ValueError):
                generate_inventory_bundle(source, source / "audit-output")

            self.assertFalse((source / "audit-output").exists())


if __name__ == "__main__":
    unittest.main()
