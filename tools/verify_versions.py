from __future__ import annotations

import json
import plistlib
import sys
import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aptus import __version__  # noqa: E402


def verify_versions() -> list[str]:
    failures: list[str] = []
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    if pyproject["project"].get("dynamic") != ["version"]:
        failures.append("pyproject.toml must declare version as dynamic")
    dynamic = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {})
    if dynamic.get("version", {}).get("attr") != "aptus.__version__":
        failures.append("pyproject.toml must source version from aptus.__version__")

    web_package = json.loads(
        (REPOSITORY_ROOT / "web/package.json").read_text(encoding="utf-8")
    )
    web_lock = json.loads(
        (REPOSITORY_ROOT / "web/package-lock.json").read_text(encoding="utf-8")
    )
    info = plistlib.loads(
        (REPOSITORY_ROOT / "desktop/macos/Resources/Info.plist").read_bytes()
    )
    openapi = json.loads(
        (REPOSITORY_ROOT / "docs/reference/openapi.v1.json").read_text(encoding="utf-8")
    )
    observed = {
        "web/package.json": web_package.get("version"),
        "web/package-lock.json": web_lock.get("version"),
        "web/package-lock.json root package": web_lock.get("packages", {})
        .get("", {})
        .get("version"),
        "desktop Info.plist": info.get("CFBundleShortVersionString"),
        "OpenAPI info": openapi.get("info", {}).get("version"),
    }
    for source, value in observed.items():
        if value != __version__:
            failures.append(
                f"{source} reports {value!r}; expected canonical {__version__!r}"
            )

    python_sources = sorted((SOURCE_ROOT / "aptus").glob("*.py"))
    for path in python_sources:
        if path.name == "__init__.py":
            continue
        if f'"{__version__}"' in path.read_text(encoding="utf-8"):
            failures.append(
                f"{path.relative_to(REPOSITORY_ROOT)} hardcodes the product version"
            )
    return failures


def main() -> int:
    failures = verify_versions()
    if failures:
        for failure in failures:
            print(f"version check: {failure}", file=sys.stderr)
        return 1
    print(f"Aptus version surfaces agree on {__version__}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
