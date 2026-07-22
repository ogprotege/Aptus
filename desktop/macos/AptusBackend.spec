# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


REPOSITORY_ROOT = Path(SPECPATH).resolve().parents[1]
APTUS_SOURCE = REPOSITORY_ROOT / "src"
requested_codesign_identity = os.environ.get("APTUS_CODESIGN_IDENTITY")
PYINSTALLER_CODESIGN_IDENTITY = (
    requested_codesign_identity
    if requested_codesign_identity and requested_codesign_identity != "-"
    else None
)

hiddenimports = sorted(
    set(
        collect_submodules("aptus")
        + collect_submodules("fastapi")
        + collect_submodules("pydantic")
        + collect_submodules("starlette")
        + collect_submodules("uvicorn")
    )
)
datas = collect_data_files("aptus", includes=["_web/**"])

analysis = Analysis(
    [str(Path(SPECPATH) / "backend_entry.py")],
    pathex=[str(APTUS_SOURCE)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="aptus-desktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=PYINSTALLER_CODESIGN_IDENTITY,
    entitlements_file=None,
)
