#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPOSITORY_ROOT="${SCRIPT_DIR:h}"
BUILD_SCRIPT="${REPOSITORY_ROOT}/desktop/macos/build.sh"
DIST_ROOT="${REPOSITORY_ROOT}/desktop/macos/dist"
ITERATIONS="${1:-10}"

if [[ "$ITERATIONS" != <-> ]] || (( ITERATIONS < 1 )); then
  print -u2 "Usage: tools/repeat_desktop_release_gate.zsh [positive-iteration-count]"
  exit 2
fi
if [[ -n "$(git -C "$REPOSITORY_ROOT" status --porcelain)" ]]; then
  print -u2 "The repeated release gate requires a clean checkout."
  exit 2
fi

EVIDENCE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/aptus-desktop-release-gate.XXXXXX")"
SUMMARY_PATH="$EVIDENCE_ROOT/RELEASE-GATE.tsv"
print 'iteration\tstarted_utc\tfinished_utc\tduration_seconds\tapp_zip_sha256\tdmg_sha256' > "$SUMMARY_PATH"

for (( iteration = 1; iteration <= ITERATIONS; iteration++ )); do
  LOG_PATH="$EVIDENCE_ROOT/build-${iteration}.log"
  STARTED_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  STARTED_SECONDS="$(date '+%s')"
  print "Aptus full desktop release gate ${iteration}/${ITERATIONS}"
  APTUS_REQUIRE_CLEAN_CHECKOUT=1 "$BUILD_SCRIPT" > "$LOG_PATH" 2>&1
  FINISHED_SECONDS="$(date '+%s')"
  FINISHED_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  APP_ZIP="$DIST_ROOT/Aptus.app.zip"
  DMG="$DIST_ROOT/Aptus-macOS-arm64.dmg"
  [[ -f "$APP_ZIP" && -f "$DMG" && -d "$DIST_ROOT/Aptus.app" ]]
  codesign --verify --deep --strict "$DIST_ROOT/Aptus.app"
  hdiutil verify "$DMG" >/dev/null
  APP_ZIP_SHA="$(shasum -a 256 "$APP_ZIP" | awk '{print $1}')"
  DMG_SHA="$(shasum -a 256 "$DMG" | awk '{print $1}')"
  print "${iteration}\t${STARTED_UTC}\t${FINISHED_UTC}\t$(( FINISHED_SECONDS - STARTED_SECONDS ))\t${APP_ZIP_SHA}\t${DMG_SHA}" >> "$SUMMARY_PATH"
done

LOG_ARCHIVE="$DIST_ROOT/release-gate-logs.zip"
FINAL_SUMMARY="$DIST_ROOT/RELEASE-GATE.tsv"
ditto -c -k --sequesterRsrc "$EVIDENCE_ROOT" "$LOG_ARCHIVE"
cp "$SUMMARY_PATH" "$FINAL_SUMMARY"
(
  cd "$DIST_ROOT"
  shasum -a 256 "RELEASE-GATE.tsv" "release-gate-logs.zip" >> SHA256SUMS
)

print "Aptus repeated release gate passed ${ITERATIONS}/${ITERATIONS}."
print "Aptus release evidence: $FINAL_SUMMARY"
print "Aptus release logs: $LOG_ARCHIVE"
