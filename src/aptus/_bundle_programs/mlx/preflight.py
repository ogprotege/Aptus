#!/usr/bin/env python3
"""Fail-closed MLX-LM dependency and uninterrupted-run preflight."""

from __future__ import annotations

import importlib.metadata
import platform


def main() -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX-LM requires Apple silicon macOS.")
    expected = {"mlx": "0.31.2", "mlx-lm": "0.31.3"}
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Expected {package}=={version}.")
    print("MLX-LM dependencies and Apple silicon platform are present.")
    print("Pilot and full runs are uninterrupted from scratch; crash-resume remains unsupported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
