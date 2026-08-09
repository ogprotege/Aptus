from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


class BundleProgramPackagingTests(unittest.TestCase):
    def test_parent_export_verification_runtime_is_pinned(self) -> None:
        project = tomllib.loads(
            (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        )

        dependencies = project["project"]["dependencies"]
        self.assertIn("numpy==2.3.5", dependencies)
        self.assertIn("safetensors==0.8.0", dependencies)

    def test_wheel_declares_every_bundle_program_resource_directory(self) -> None:
        project = tomllib.loads(
            (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
        )
        package_data = set(project["tool"]["setuptools"]["package-data"]["aptus"])
        self.assertTrue(
            {
                "_bundle_programs/cuda/*.py",
                "_bundle_programs/mlx/*.py",
            }
            <= package_data
        )

    def test_frozen_sidecar_collects_bundle_programs_as_data(self) -> None:
        specification = (
            REPOSITORY / "desktop" / "macos" / "AptusBackend.spec"
        ).read_text(encoding="utf-8")
        self.assertIn('APTUS_SOURCE / "aptus" / "_bundle_programs"', specification)
        self.assertIn('bundle_program_root.rglob("*.py")', specification)
        self.assertIn('"aptus/_bundle_programs/"', specification)
        self.assertIn('("plan_contract.py", "runtime_lease.py")', specification)

    def test_release_build_rechecks_clean_checkout_before_publishing(self) -> None:
        script = (REPOSITORY / "desktop" / "macos" / "build.sh").read_text(
            encoding="utf-8"
        )
        first_check = script.index("Aptus release evidence requires a clean checkout")
        final_check = script.index("Aptus release gates changed tracked files")
        checksum_publication = script.index('shasum -a 256 "Aptus.app.zip"')
        self.assertLess(first_check, final_check)
        self.assertLess(final_check, checksum_publication)
        self.assertGreaterEqual(script.count('REQUIRE_CLEAN_CHECKOUT" == "1'), 2)

    def test_dmg_verification_detaches_the_whole_device_with_bounded_retries(
        self,
    ) -> None:
        script = (REPOSITORY / "desktop" / "macos" / "build.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("hdiutil attach -plist -readonly", script)
        self.assertIn("system-entities.$entity_index.dev-entry", script)
        self.assertIn('"$DMG_ENTITY_DEVICE" == /dev/disk<->', script)
        self.assertIn("system-entities.$entity_index.content-hint", script)
        self.assertIn('"$DMG_ENTITY_HINT" == "GUID_partition_scheme"', script)
        self.assertIn("returned multiple backing devices", script)
        self.assertIn("for attempt in {1..5}", script)
        self.assertIn('hdiutil detach "$DMG_DEVICE"', script)
        self.assertIn('hdiutil detach -force "$DMG_DEVICE"', script)
        self.assertNotIn('hdiutil detach "$DMG_MOUNT"', script)


if __name__ == "__main__":
    unittest.main()
