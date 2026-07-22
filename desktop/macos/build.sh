#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPOSITORY_ROOT="${SCRIPT_DIR:h:h}"
BUILD_ROOT="${SCRIPT_DIR}/build"
DIST_ROOT="${SCRIPT_DIR}/dist"
PROJECT_FILE="${SCRIPT_DIR}/Aptus.xcodeproj"
TEST_DERIVED_DATA="${BUILD_ROOT}/TestDerivedData"
APP_DERIVED_DATA="${BUILD_ROOT}/AppDerivedData"
CONFIGURATION="Release"
BUILD_BACKEND=1
CREATE_DMG=1
RUN_TESTS=1
RUN_WEB_GATE=1

for argument in "$@"; do
  case "$argument" in
    --skip-backend) BUILD_BACKEND=0 ;;
    --skip-dmg) CREATE_DMG=0 ;;
    --skip-tests) RUN_TESTS=0 ;;
    --skip-web) RUN_WEB_GATE=0 ;;
    --debug) CONFIGURATION="Debug" ;;
    *)
      print -u2 "Unknown Aptus build option: $argument"
      exit 2
      ;;
  esac
done

if (( ! BUILD_BACKEND )) && [[ "$CONFIGURATION" != "Debug" ]]; then
  print -u2 "Release builds cannot use --skip-backend. Add --debug for a repository-backed development app."
  exit 2
fi
if (( ! RUN_TESTS )) && [[ "$CONFIGURATION" != "Debug" ]]; then
  print -u2 "Release builds cannot use --skip-tests. Add --debug for an unchecked development build."
  exit 2
fi
if (( ! RUN_WEB_GATE )) && [[ "$CONFIGURATION" != "Debug" ]]; then
  print -u2 "Release builds cannot use --skip-web. Add --debug to reuse development workbench assets."
  exit 2
fi
if [[ -n "${APTUS_PYINSTALLER_PYTHON:-}" && "$CONFIGURATION" != "Debug" ]]; then
  print -u2 "Release builds cannot use APTUS_PYINSTALLER_PYTHON. The hash-locked Python 3.12 environment is mandatory."
  exit 2
fi

for required_tool in xcodegen xcodebuild xcrun ditto codesign plutil; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    print -u2 "Aptus desktop build requires $required_tool."
    exit 2
  fi
done
if (( CREATE_DMG )) && ! command -v hdiutil >/dev/null 2>&1; then
  print -u2 "Aptus disk-image packaging requires hdiutil."
  exit 2
fi

rm -rf "$BUILD_ROOT" "$DIST_ROOT" "$PROJECT_FILE"
mkdir -p "$BUILD_ROOT" "$DIST_ROOT"

if (( RUN_TESTS )); then
  if ! command -v uv >/dev/null 2>&1; then
    print -u2 "Aptus product tests require uv."
    exit 2
  fi
  (
    cd "$REPOSITORY_ROOT"
    PYTHONPATH="$REPOSITORY_ROOT/src:$REPOSITORY_ROOT" \
      uv run --isolated --python 3.12 --locked --extra server --extra test \
      python -m unittest discover -s tests -t .
  )
fi

if (( RUN_WEB_GATE )); then
  for required_tool in node npm; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
      print -u2 "Aptus workbench build requires $required_tool."
      exit 2
    fi
  done
  (
    cd "$REPOSITORY_ROOT/web"
    npm ci
    npm test
    npm run typecheck
    npm run build
  )
fi

BACKEND_BINARY=""
if (( BUILD_BACKEND )); then
  if [[ -n "${APTUS_PYINSTALLER_PYTHON:-}" ]]; then
    PYINSTALLER_PYTHON="$APTUS_PYINSTALLER_PYTHON"
  else
    if ! command -v uv >/dev/null 2>&1; then
      print -u2 "Aptus packaging requires uv or an APTUS_PYINSTALLER_PYTHON override."
      exit 2
    fi
    BACKEND_ENV="$BUILD_ROOT/backend-venv"
    uv venv --python 3.12 "$BACKEND_ENV"
    uv pip install \
      --python "$BACKEND_ENV/bin/python" \
      --require-hashes \
      --only-binary :all: \
      --requirement "$SCRIPT_DIR/requirements-build.lock"
    uv pip install \
      --python "$BACKEND_ENV/bin/python" \
      --no-deps \
      --no-build-isolation \
      "$REPOSITORY_ROOT"
    PYINSTALLER_PYTHON="$BACKEND_ENV/bin/python"
  fi
  if [[ ! -x "$PYINSTALLER_PYTHON" ]]; then
    print -u2 "APTUS_PYINSTALLER_PYTHON must name an executable Python 3.12 environment."
    exit 2
  fi
  if ! "$PYINSTALLER_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    print -u2 "Aptus desktop packaging requires Python 3.12."
    exit 2
  fi
  if ! "$PYINSTALLER_PYTHON" -c 'import PyInstaller' >/dev/null 2>&1; then
    print -u2 "PyInstaller is missing from $PYINSTALLER_PYTHON. Install Aptus with [server,desktop-build]."
    exit 2
  fi
  "$PYINSTALLER_PYTHON" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$BUILD_ROOT/backend-dist" \
    --workpath "$BUILD_ROOT/backend-work" \
    "$SCRIPT_DIR/AptusBackend.spec"
  BACKEND_BINARY="$BUILD_ROOT/backend-dist/aptus-desktop"
fi

xcodegen generate \
  --spec "$SCRIPT_DIR/project.yml" \
  --project-root "$SCRIPT_DIR"

if (( RUN_TESTS )); then
  mkdir -p "$TEST_DERIVED_DATA/Profiles"
  LLVM_PROFILE_FILE="$TEST_DERIVED_DATA/Profiles/aptus-%p.profraw" \
    xcodebuild \
    -project "$PROJECT_FILE" \
    -scheme Aptus \
    -configuration Debug \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath "$TEST_DERIVED_DATA" \
    ARCHS=arm64 \
    ONLY_ACTIVE_ARCH=YES \
    CODE_SIGNING_ALLOWED=NO \
    test
fi

xcodebuild \
  -project "$PROJECT_FILE" \
  -scheme Aptus \
  -configuration "$CONFIGURATION" \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$APP_DERIVED_DATA" \
  ARCHS=arm64 \
  ONLY_ACTIVE_ARCH=YES \
  CODE_SIGNING_ALLOWED=NO \
  build

BUILT_APP="$APP_DERIVED_DATA/Build/Products/$CONFIGURATION/Aptus.app"
OUTPUT_APP="$DIST_ROOT/Aptus.app"
ditto "$BUILT_APP" "$OUTPUT_APP"

mkdir -p "$OUTPUT_APP/Contents/Resources/backend"
if (( BUILD_BACKEND )); then
  ditto "$BACKEND_BINARY" "$OUTPUT_APP/Contents/Resources/backend/aptus-desktop"
  chmod 0755 "$OUTPUT_APP/Contents/Resources/backend/aptus-desktop"
fi

xcrun swift \
  "$SCRIPT_DIR/scripts/render_icon.swift" \
  "$SCRIPT_DIR/Resources/AptusMark.svg" \
  "$OUTPUT_APP/Contents/Resources/AppIcon.icns"

SIGNING_IDENTITY="${APTUS_CODESIGN_IDENTITY:--}"
if [[ -x "$OUTPUT_APP/Contents/Resources/backend/aptus-desktop" ]]; then
  if [[ "$SIGNING_IDENTITY" == "-" ]]; then
    codesign --force --sign - "$OUTPUT_APP/Contents/Resources/backend/aptus-desktop"
  else
    codesign --force --options runtime --timestamp --sign "$SIGNING_IDENTITY" "$OUTPUT_APP/Contents/Resources/backend/aptus-desktop"
  fi
fi
if [[ "$SIGNING_IDENTITY" == "-" ]]; then
  codesign --force --sign - --entitlements "$SCRIPT_DIR/Resources/Aptus.entitlements" "$OUTPUT_APP"
else
  codesign --force --options runtime --timestamp --sign "$SIGNING_IDENTITY" \
    --entitlements "$SCRIPT_DIR/Resources/Aptus.entitlements" "$OUTPUT_APP"
fi
codesign --verify --deep --strict --verbose=2 "$OUTPUT_APP"

if (( BUILD_BACKEND )); then
  PROBE_ROOT="$BUILD_ROOT/launch-probe"
  PROBE_FILE="$PROBE_ROOT/result.json"
  PROBE_LOG="$PROBE_ROOT/app.log"
  mkdir -p "$PROBE_ROOT"
  APTUS_DESKTOP_LAUNCH_PROBE_FILE="$PROBE_FILE" \
  APTUS_DESKTOP_LAUNCH_PROBE_ROOT="$PROBE_ROOT/runtime" \
  LLVM_PROFILE_FILE="$PROBE_ROOT/aptus-%p.profraw" \
    "$OUTPUT_APP/Contents/MacOS/Aptus" >"$PROBE_LOG" 2>&1 &
  APP_PID=$!
  for attempt in {1..200}; do
    if [[ -s "$PROBE_FILE" ]] || ! kill -0 "$APP_PID" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  for attempt in {1..50}; do
    if ! kill -0 "$APP_PID" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$APP_PID" 2>/dev/null; then
    kill -TERM "$APP_PID" 2>/dev/null || true
  fi
  wait "$APP_PID" 2>/dev/null || true
  if [[ ! -s "$PROBE_FILE" ]]; then
    print -u2 "Packaged Aptus did not report a backend-ready visible window."
    tail -n 80 "$PROBE_LOG" >&2 || true
    exit 1
  fi
  if [[ "$(plutil -extract backendReady raw -o - "$PROBE_FILE")" != "true" \
        || "$(plutil -extract windowVisible raw -o - "$PROBE_FILE")" != "true" \
        || "$(plutil -extract workbenchLoaded raw -o - "$PROBE_FILE")" != "true" \
        || "$(plutil -extract reactReady raw -o - "$PROBE_FILE")" != "true" \
        || "$(plutil -extract workbenchMarker raw -o - "$PROBE_FILE")" != "aptus-workbench-v1" \
        || "$(plutil -extract host raw -o - "$PROBE_FILE")" != "127.0.0.1" ]]; then
    print -u2 "Packaged Aptus launch probe returned an invalid result."
    plutil -p "$PROBE_FILE" >&2 || true
    exit 1
  fi
  if [[ -e "$PROBE_ROOT/runtime/caches/Aptus/sessions/packaged-launch" ]]; then
    print -u2 "Packaged Aptus left stale desktop-session metadata after shutdown."
    exit 1
  fi
fi

if (( CREATE_DMG )); then
  DMG_STAGE="$BUILD_ROOT/dmg-stage"
  DMG_MOUNT="$BUILD_ROOT/dmg-verify"
  DMG_PATH="$DIST_ROOT/Aptus-macOS-arm64.dmg"
  mkdir -p "$DMG_STAGE" "$DMG_MOUNT"
  ditto "$OUTPUT_APP" "$DMG_STAGE/Aptus.app"
  ln -s /Applications "$DMG_STAGE/Applications"
  hdiutil create \
    -volname "Aptus" \
    -srcfolder "$DMG_STAGE" \
    -ov \
    -format UDZO \
    "$DMG_PATH"
  hdiutil attach -readonly -nobrowse -mountpoint "$DMG_MOUNT" "$DMG_PATH" >/dev/null
  DMG_LAYOUT_VALID=1
  [[ -d "$DMG_MOUNT/Aptus.app" ]] || DMG_LAYOUT_VALID=0
  [[ -L "$DMG_MOUNT/Applications" ]] || DMG_LAYOUT_VALID=0
  [[ "$(readlink "$DMG_MOUNT/Applications")" == "/Applications" ]] || DMG_LAYOUT_VALID=0
  hdiutil detach "$DMG_MOUNT" >/dev/null
  if (( ! DMG_LAYOUT_VALID )); then
    print -u2 "Aptus disk image is missing the app or Applications shortcut."
    exit 1
  fi
fi

print "Aptus app: $OUTPUT_APP"
if (( CREATE_DMG )); then
  print "Aptus disk image: $DIST_ROOT/Aptus-macOS-arm64.dmg"
fi
