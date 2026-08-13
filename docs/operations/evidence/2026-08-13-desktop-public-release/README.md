# Desktop public Mac distribution, 2026-08-13

> **Status:** Passed — one clean Developer ID signed, notarized, stapled arm64 app and DMG  
> **Evidence class:** Exact-commit public packaging identity  
> **Source commit:** `edc6cfdec48daeb17af8cae7dbb9fde0d8112a81`  
> **Source tree:** `c183799cf90cc8234b401647c42f60f8a63ec96b`  
> **Last reviewed:** 2026-08-13  
> **Review by:** Before any other commit, host, or architecture is called a public Mac download

## Result

One `desktop/macos/build.sh` run from a clean checkout of
`edc6cfdec48daeb17af8cae7dbb9fde0d8112a81` produced `Aptus.app`,
`Aptus.app.zip`, and `Aptus-macOS-arm64.dmg`. Both the app and the DMG were
signed with Developer ID Application, submitted to Apple notarization,
**Accepted**, stapled, stapler-validated, and Gatekeeper-assessed as
`accepted` / `Notarized Developer ID`.

This packet binds that one artifact identity. It does not republish the
binaries.

## Artifact identity

| Field | Value |
| --- | --- |
| Application ID | `com.aptus.desktop` |
| Version / build | `0.2.0` / `1` |
| Architecture | arm64 |
| Signing identity | `Developer ID Application: WILSON WILKERSON WARREN (4KBWH9KYSD)` |
| Team ID | `4KBWH9KYSD` |
| App timestamp | 2026-08-13 12:43:06 local |
| DMG timestamp | 2026-08-13 12:43:37 local |
| `Aptus.app.zip` SHA-256 | `41afcb0cce4fca7a32374fe849b678c8723cb6584cf9e3140cc16bb1bf0a08e5` |
| `Aptus.app.zip` bytes | 16,651,747 |
| `Aptus-macOS-arm64.dmg` SHA-256 | `bf28fe7416b24d1d000c39a83af317b1062fdf74c32fe499341bca9d3ea13834` |
| `Aptus-macOS-arm64.dmg` bytes | 17,003,512 |
| App notary submission | `ad36bb3a-5e1f-420f-ae1f-f2be1d84960a` (Accepted) |
| DMG notary submission | `d3794efb-e44b-4d8d-886d-1e5c32d3f0df` (Accepted) |
| Dist `COMMIT` | `edc6cfdec48daeb17af8cae7dbb9fde0d8112a81` (no dirty marker) |

Independent verification on the build host after the script exited 0:

```text
spctl -a -vv Aptus.app
  accepted
  source=Notarized Developer ID
  origin=Developer ID Application: WILSON WILKERSON WARREN (4KBWH9KYSD)

spctl -a -t open --context context:primary-signature -vv Aptus-macOS-arm64.dmg
  accepted
  source=Notarized Developer ID
  origin=Developer ID Application: WILSON WILKERSON WARREN (4KBWH9KYSD)
```

`xcrun stapler validate` succeeded for both artifacts.

## Bound build

| Field | Value |
| --- | --- |
| Command | `APTUS_REQUIRE_CLEAN_CHECKOUT=1 APTUS_REQUIRE_NOTARIZATION=1 APTUS_CODESIGN_IDENTITY='Developer ID Application: WILSON WILKERSON WARREN (4KBWH9KYSD)' APTUS_NOTARY_PROFILE=aptus-notary desktop/macos/build.sh` |
| Exit | 0 |
| Host | Apple M5 Pro, arm64, 64 GiB, macOS 26.6.1 (25G76) |
| Python tests | 957 ran, 0 failed, 51.871 s |
| Web tests | 21 files / 134 tests passed |
| Native tests | 92 executed, 0 failed, `TEST SUCCEEDED` |
| Build-log SHA-256 | `1e513f34100dceb61500edfd7e4dec5be99810aca335c827956e43b56c5fc73e` |
| Build-log bytes | 328,798 |

The transcript is retained outside Git as
`aptus-work/m6-notarized-build-2.log`. It is not a CI URL.

`edc6cfd` is `db59ed9` (M5 / PR #90) plus the packaged workbench refresh
required because M5 Compare UI had not been copied into `src/aptus/_web/`.
A first notarization of dirty-tree `db59ed9` was Accepted
(`0bc64dc8-e2be-4dd1-9e22-a7720ea1a130` app,
`5ac002ae-1ff7-4257-826f-c2aa58485cd8` DMG) and is **not** this public
identity.

## Claim boundary

**Supports only:** this exact source commit and tree, these two SHA-256
digests, these two notary IDs, Developer ID team `4KBWH9KYSD`, and Gatekeeper
assessment of those local arm64 artifacts on the recorded host.

**Does not support:** any other commit; Intel or universal Macs; App Store
distribution; CI ad-hoc artifacts; the 2026-07-27 ten-run ad-hoc stability
gate; a published download URL; ten notarized repetitions; model quality;
training-runtime acceptance; or Aptus 0.2 product release.

The [2026-07-27 desktop engineering record](../2026-07-27-desktop-release/README.md)
remains the historical 10-of-10 ad-hoc stability packet. It is not this
notarized identity.

## Files

- [`identity.json`](identity.json) — machine-readable rollup  
- [`COMMIT`](COMMIT) — artifact source commit  
- [`SHA256SUMS`](SHA256SUMS) — digests of committed packet files and of the
  unpublished ZIP/DMG

Binaries stay outside Git under ignored `desktop/macos/dist/`.
