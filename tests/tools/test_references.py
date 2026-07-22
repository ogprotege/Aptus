import tempfile
import unittest
from pathlib import Path

from tools.aptus_audit.references import analyze_references


class ReferenceAnalysisTests(unittest.TestCase):
    def test_analyzes_python_parse_status_and_relative_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pkg").mkdir()
            (root / "pkg" / "main.py").write_text(
                "from .helper import value\nfrom .missing import nope\n",
                encoding="utf-8",
            )
            (root / "pkg" / "helper.py").write_text("value = 1\n", encoding="utf-8")
            (root / "broken.py").write_text("def broken(:\n", encoding="utf-8")

            report = analyze_references(root)
            files = {item["path"]: item for item in report["files"]}

            self.assertEqual(files["pkg/main.py"]["parse_status"], "passed")
            self.assertEqual(files["broken.py"]["parse_status"], "failed")
            imports = {
                item["specifier"]: item for item in files["pkg/main.py"]["imports"]
            }
            self.assertEqual(imports[".helper"]["status"], "resolved")
            self.assertEqual(imports[".helper"]["resolved_path"], "pkg/helper.py")
            self.assertEqual(imports[".missing"]["status"], "missing")

    def test_resolves_typescript_relative_import_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.ts").write_text(
                "import { value } from './present';\n"
                "const missing = require('./missing');\n",
                encoding="utf-8",
            )
            (root / "present.ts").write_text(
                "export const value = 1;\n", encoding="utf-8"
            )

            report = analyze_references(root)
            main = next(item for item in report["files"] if item["path"] == "main.ts")
            imports = {item["specifier"]: item for item in main["imports"]}

            self.assertEqual(imports["./present"]["status"], "resolved")
            self.assertEqual(imports["./present"]["resolved_path"], "present.ts")
            self.assertEqual(imports["./missing"]["status"], "missing")
            self.assertEqual(report["summary"]["missing_relative_imports"], 1)


if __name__ == "__main__":
    unittest.main()
