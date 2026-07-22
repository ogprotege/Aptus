from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


SCRIPT_EXTENSIONS = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
JAVASCRIPT_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
IMPORT_PATTERNS = (
    re.compile(r"(?:import|export)\s+(?:[^'\"\n]*?\s+from\s+)?['\"]([^'\"]+)['\"]"),
    re.compile(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)"),
    re.compile(r"\bimport\(\s*['\"]([^'\"]+)['\"]\s*\)"),
)


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _resolve_javascript_import(
    source: Path, specifier: str, root: Path
) -> tuple[str, str | None]:
    if not specifier.startswith("."):
        return "external", None

    base = (source.parent / specifier).resolve()
    candidates = [base]
    for extension in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json"):
        candidates.append(Path(f"{base}{extension}"))
    for extension in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        candidates.append(base / f"index{extension}")

    for candidate in candidates:
        if candidate.is_file() and candidate.is_relative_to(root):
            return "resolved", _relative_path(candidate, root)
    return "missing", None


def _resolve_python_import(
    source: Path,
    module: str | None,
    level: int,
    root: Path,
) -> tuple[str, str | None]:
    if level == 0:
        return "external", None

    base = source.parent
    for _ in range(max(0, level - 1)):
        base = base.parent
    if module:
        base = base.joinpath(*module.split("."))

    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.is_relative_to(root):
            return "resolved", _relative_path(candidate, root)
    return "missing", None


def _python_record(path: Path, root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative_path(path, root),
        "language": "python",
        "parse_status": "passed",
        "parse_error": None,
        "imports": [],
    }
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        record["parse_status"] = "failed"
        record["parse_error"] = {
            "type": type(error).__name__,
            "line": getattr(error, "lineno", None),
            "message": str(error),
        }
        return record

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "specifier": alias.name,
                        "kind": "import",
                        "line": node.lineno,
                        "status": "external",
                        "resolved_path": None,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            specifier = "." * node.level + (node.module or "")
            status, resolved_path = _resolve_python_import(
                path, node.module, node.level, root
            )
            imports.append(
                {
                    "specifier": specifier,
                    "kind": "from",
                    "line": node.lineno,
                    "status": status,
                    "resolved_path": resolved_path,
                }
            )
    record["imports"] = sorted(
        imports, key=lambda item: (item["line"], item["specifier"])
    )
    return record


def _javascript_record(path: Path, root: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        return {
            "path": _relative_path(path, root),
            "language": "javascript-typescript",
            "parse_status": "failed",
            "parse_error": {
                "type": type(error).__name__,
                "line": None,
                "message": str(error),
            },
            "imports": [],
        }

    imports = []
    seen = set()
    for pattern in IMPORT_PATTERNS:
        for match in pattern.finditer(text):
            specifier = match.group(1)
            key = (specifier, text.count("\n", 0, match.start()) + 1)
            if key in seen:
                continue
            seen.add(key)
            status, resolved_path = _resolve_javascript_import(path, specifier, root)
            imports.append(
                {
                    "specifier": specifier,
                    "kind": "module",
                    "line": key[1],
                    "status": status,
                    "resolved_path": resolved_path,
                }
            )

    return {
        "path": _relative_path(path, root),
        "language": "javascript-typescript",
        "parse_status": "not_checked",
        "parse_error": None,
        "imports": sorted(imports, key=lambda item: (item["line"], item["specifier"])),
    }


def analyze_references(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_file() or path.suffix.lower() not in SCRIPT_EXTENSIONS:
            continue
        if path.suffix.lower() == ".py":
            files.append(_python_record(path, root))
        else:
            files.append(_javascript_record(path, root))

    imports = [item for file in files for item in file["imports"]]
    summary = {
        "script_file_count": len(files),
        "python_file_count": sum(file["language"] == "python" for file in files),
        "javascript_typescript_file_count": sum(
            file["language"] == "javascript-typescript" for file in files
        ),
        "python_parse_failures": sum(
            file["language"] == "python" and file["parse_status"] == "failed"
            for file in files
        ),
        "relative_import_count": sum(
            item["specifier"].startswith(".") for item in imports
        ),
        "missing_relative_imports": sum(
            item["status"] == "missing" for item in imports
        ),
    }
    return {"summary": summary, "files": files}
