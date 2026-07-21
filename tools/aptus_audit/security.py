from __future__ import annotations

import re
from pathlib import Path
from typing import Any


SECRET_PATTERNS = (
    (
        "aws-access-key-id",
        "high",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    (
        "github-token",
        "high",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b"),
    ),
    (
        "openai-style-key",
        "high",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,255}\b"),
    ),
    (
        "private-key-header",
        "critical",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)


def _masked(value: str) -> str:
    if len(value) <= 8:
        return "[REDACTED]"
    return f"{value[:4]}…{value[-4:]}"


def scan_secrets(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    findings: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue

        content = path.read_bytes()
        if b"\x00" in content[:8192]:
            continue

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule_id, severity, pattern in SECRET_PATTERNS:
                for match in pattern.finditer(line):
                    masked_line = (
                        line[: match.start()]
                        + _masked(match.group(0))
                        + line[match.end() :]
                    )
                    findings.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "line": line_number,
                            "rule_id": rule_id,
                            "severity": severity,
                            "masked_preview": masked_line[:160],
                        }
                    )

    return findings
