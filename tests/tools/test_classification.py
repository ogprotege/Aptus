import unittest

from tools.aptus_audit.classification import classify_artifacts


class ClassificationTests(unittest.TestCase):
    def test_classifies_duplicate_copy_and_preserves_canonical_artifact(self) -> None:
        records = [
            {
                "path": "tests/test_optimizer.py",
                "kind": "text",
                "extension": ".py",
            },
            {
                "path": "tests/test_optimizer 2.py",
                "kind": "text",
                "extension": ".py",
            },
        ]
        clusters = [
            {
                "cluster_id": "sha256:abc",
                "paths": ["tests/test_optimizer 2.py", "tests/test_optimizer.py"],
            }
        ]

        results = classify_artifacts(records, clusters, {})
        by_path = {result["path"]: result for result in results}

        self.assertEqual(by_path["tests/test_optimizer 2.py"]["disposition"], "DISCARD")
        self.assertEqual(
            by_path["tests/test_optimizer 2.py"]["canonical_path"],
            "tests/test_optimizer.py",
        )
        self.assertNotEqual(
            by_path["tests/test_optimizer.py"]["disposition"],
            "DISCARD",
        )

    def test_hard_rules_cover_junk_empty_and_vendored_reference(self) -> None:
        records = [
            {"path": ".DS_Store", "kind": "binary", "extension": ""},
            {
                "path": "src/methods/premium/vaporware.ts",
                "kind": "empty",
                "extension": ".ts",
            },
            {
                "path": "PyReft-Repo/pyreft/interventions.py",
                "kind": "text",
                "extension": ".py",
            },
        ]

        results = {
            result["path"]: result for result in classify_artifacts(records, [], {})
        }

        self.assertEqual(results[".DS_Store"]["disposition"], "DISCARD")
        self.assertEqual(
            results["src/methods/premium/vaporware.ts"]["disposition"],
            "DISCARD",
        )
        self.assertEqual(
            results["PyReft-Repo/pyreft/interventions.py"]["disposition"],
            "ARCHIVE",
        )
        self.assertIn(
            "DO_NOT_SHIP_PROVENANCE",
            results["PyReft-Repo/pyreft/interventions.py"]["warnings"],
        )

    def test_known_domain_asset_is_adapted_not_discarded(self) -> None:
        records = [
            {
                "path": "src/python/resource_scanner.py",
                "kind": "text",
                "extension": ".py",
            }
        ]

        result = classify_artifacts(records, [], {})[0]

        self.assertEqual(result["disposition"], "ADAPT")
        self.assertGreaterEqual(result["knowledge_value"], 4)
        self.assertIn("EVIDENCE:KNOWN_DOMAIN_ASSET", result["evidence_ids"])

    def test_duplicate_canonicalization_preserves_explicit_domain_asset(self) -> None:
        records = [
            {
                "path": "src/hypertuner/evaluation/lora_evaluator.py",
                "kind": "text",
                "extension": ".py",
            },
            {
                "path": "src/hypertuner/training/lora_evaluator.py",
                "kind": "text",
                "extension": ".py",
            },
        ]
        clusters = [
            {
                "cluster_id": "sha256:evaluator",
                "paths": [
                    "src/hypertuner/evaluation/lora_evaluator.py",
                    "src/hypertuner/training/lora_evaluator.py",
                ],
            }
        ]

        by_path = {
            result["path"]: result
            for result in classify_artifacts(records, clusters, {})
        }

        canonical = by_path["src/hypertuner/evaluation/lora_evaluator.py"]
        duplicate = by_path["src/hypertuner/training/lora_evaluator.py"]
        self.assertEqual(canonical["disposition"], "ADAPT")
        self.assertEqual(
            canonical["canonical_path"],
            "src/hypertuner/evaluation/lora_evaluator.py",
        )
        self.assertEqual(duplicate["disposition"], "DISCARD")
        self.assertEqual(
            duplicate["canonical_path"],
            "src/hypertuner/evaluation/lora_evaluator.py",
        )

    def test_returns_one_complete_decision_per_input_record(self) -> None:
        records = [
            {"path": "README.md", "kind": "text", "extension": ".md"},
            {"path": "server.js", "kind": "text", "extension": ".js"},
        ]

        results = classify_artifacts(records, [], {})

        self.assertEqual(
            [result["path"] for result in results], ["README.md", "server.js"]
        )
        for result in results:
            self.assertIn(
                result["disposition"], {"KEEP", "ADAPT", "ARCHIVE", "DISCARD"}
            )
            self.assertIn(result["operational_viability"], range(6))
            self.assertIn(result["knowledge_value"], range(6))
            self.assertTrue(result["rationale"])
            self.assertTrue(result["confidence"])


if __name__ == "__main__":
    unittest.main()
