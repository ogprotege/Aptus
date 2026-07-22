import unittest

from aptus.domain import Distribution, Method
from aptus.evidence import EVIDENCE_REGISTRY
from aptus.methods import (
    METHOD_REGISTRY,
    MethodLifecycle,
    descriptor_for_compiler,
    method_descriptors,
)


class MethodRegistryTests(unittest.TestCase):
    def test_selectable_registry_exactly_matches_executable_enum(self) -> None:
        selectable = {
            item.method_id for item in method_descriptors() if item.selectable
        }
        self.assertEqual(selectable, {item.value for item in Method})
        self.assertTrue(
            all(
                item.lifecycle == MethodLifecycle.GATED_EXECUTABLE
                and item.compiler_id
                and item.export_kind
                for item in method_descriptors()
                if item.selectable
            )
        )

    def test_researched_methods_do_not_become_selectable_by_presence(self) -> None:
        expected = {
            "dora": MethodLifecycle.EXPERIMENTAL,
            "bitfit": MethodLifecycle.EXPERIMENTAL,
            "adalora": MethodLifecycle.EXPERIMENTAL,
            "loreft": MethodLifecycle.RESEARCH_ONLY,
            "aflora": MethodLifecycle.RESEARCH_ONLY,
            "bilora": MethodLifecycle.RESEARCH_ONLY,
            "sharelora": MethodLifecycle.EXPERIMENTAL,
        }
        for method_id, lifecycle in expected.items():
            with self.subTest(method=method_id):
                descriptor = METHOD_REGISTRY[method_id]
                self.assertFalse(descriptor.selectable)
                self.assertEqual(descriptor.lifecycle, lifecycle)
                self.assertIsNone(descriptor.compiler_id)
                self.assertIsNone(descriptor.export_kind)
                self.assertTrue(descriptor.blocker)
                self.assertTrue(descriptor.pilot_requirement)

    def test_every_descriptor_evidence_reference_resolves(self) -> None:
        for descriptor in method_descriptors():
            with self.subTest(method=descriptor.method_id):
                self.assertTrue(descriptor.evidence_ids)
                self.assertTrue(
                    set(descriptor.evidence_ids).issubset(EVIDENCE_REGISTRY)
                )

    def test_registry_aliases_and_ids_are_globally_unique(self) -> None:
        identities: list[str] = list(METHOD_REGISTRY)
        for descriptor in method_descriptors():
            identities.extend(descriptor.aliases)
        self.assertEqual(len(identities), len(set(identities)))

    def test_every_selectable_compiler_and_distribution_resolves(self) -> None:
        canonical_distributions = {item.value for item in Distribution}
        for descriptor in method_descriptors():
            if not descriptor.selectable:
                continue
            with self.subTest(method=descriptor.method_id):
                self.assertEqual(
                    descriptor_for_compiler(str(descriptor.compiler_id)), descriptor
                )
                self.assertTrue(
                    set(descriptor.supported_distributions).issubset(
                        canonical_distributions
                    )
                )


if __name__ == "__main__":
    unittest.main()
