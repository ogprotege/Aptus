from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import sys
from pathlib import Path
from typing import Sequence

from . import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptus-desktop",
        description="Run the authenticated loopback service for Aptus for Mac.",
    )
    parser.add_argument(
        "--state-dir",
        required=True,
        type=Path,
        help="Private application state directory.",
    )
    parser.add_argument(
        "--ready-file",
        required=True,
        type=Path,
        help="Session-specific JSON endpoint file consumed by the native host.",
    )
    return parser


def _private_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved.chmod(0o700)
    actual_mode = stat.S_IMODE(resolved.stat().st_mode)
    if actual_mode != 0o700:
        raise PermissionError(f"Aptus desktop directory must use mode 0700: {resolved}")
    return resolved


def _write_ready_file(path: Path, *, port: int) -> None:
    parent = _private_directory(path.expanduser().resolve().parent)
    target = parent / path.name
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = {
        "host": "127.0.0.1",
        "port": port,
        "version": __version__,
    }
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        actual_mode = stat.S_IMODE(target.stat().st_mode)
        if actual_mode != 0o600:
            target.unlink(missing_ok=True)
            raise PermissionError(
                f"Aptus desktop readiness file must use mode 0600: {target}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def _serve(state_dir: Path, ready_file: Path, session_token: str) -> int:
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - packaging contract
        raise RuntimeError(
            "The Aptus desktop runtime requires the server extra."
        ) from error

    from .api import create_app

    state_root = _private_directory(state_dir)
    ready_target = ready_file.expanduser().resolve()
    ready_target.unlink(missing_ok=True)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind(("127.0.0.1", 0))
        server_socket.listen(2048)
        port = int(server_socket.getsockname()[1])
        _write_ready_file(ready_target, port=port)
        config = uvicorn.Config(
            create_app(
                state_dir=state_root,
                allowed_hosts=("127.0.0.1",),
                session_token=session_token,
                execution_enabled=False,
            ),
            host="127.0.0.1",
            port=port,
            access_log=False,
        )
        uvicorn.Server(config).run(sockets=[server_socket])
    finally:
        ready_target.unlink(missing_ok=True)
        server_socket.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    token = os.environ.get("APTUS_DESKTOP_SESSION_TOKEN", "")
    if len(token) < 32:
        print(
            "Aptus desktop error: APTUS_DESKTOP_SESSION_TOKEN must contain at least 32 characters.",
            file=sys.stderr,
        )
        return 2
    try:
        return _serve(arguments.state_dir, arguments.ready_file, token)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Aptus desktop error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
