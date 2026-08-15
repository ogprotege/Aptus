from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aptus.api import create_app  # noqa: E402
from aptus.api_contracts import API_CONTRACT_VERSION  # noqa: E402


def render_openapi() -> str:
    with tempfile.TemporaryDirectory(prefix="aptus-openapi-") as temporary:
        schema = create_app(
            state_dir=Path(temporary) / "state",
            allow_unauthenticated=True,
        ).openapi()
    schema["info"]["x-aptus-contract-version"] = API_CONTRACT_VERSION
    return json.dumps(schema, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify Aptus's checked OpenAPI contract."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "reference" / "openapi.v1.json",
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = render_openapi()
    if arguments.check:
        if not arguments.output.is_file():
            print(f"Missing OpenAPI artifact: {arguments.output}", file=sys.stderr)
            return 1
        if arguments.output.read_text(encoding="utf-8") != rendered:
            print(
                f"OpenAPI artifact is stale: regenerate {arguments.output}",
                file=sys.stderr,
            )
            return 1
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
