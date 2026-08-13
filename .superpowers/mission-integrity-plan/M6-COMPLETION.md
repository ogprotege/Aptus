# M6 — COMPLETION NOTE

> **Phase status:** **COMPLETE** locally (pending PR merge)

| Field | Value |
| --- | --- |
| Phase | M6 |
| Title | Public Mac distribution integrity |
| Started (UTC) | 2026-08-12 |
| Completed (UTC) | 2026-08-13 |
| Start commit | `db59ed9` |
| End commit / tree state | `edc6cfdec48daeb17af8cae7dbb9fde0d8112a81` / `c183799cf90cc8234b401647c42f60f8a63ec96b` |
| Owner sign-off | pending |

## Mission check

- Mission statement still accurate? yes  
- Invariants I1–I12 intact? yes  
- Claim-language violations introduced? none intended; claims bound to one commit and two SHA-256s  

## Tasks completed

| Task | Review |
| --- | --- |
| Developer ID Application identity | `WILSON WILKERSON WARREN (4KBWH9KYSD)` |
| notarytool profile `aptus-notary` | validated |
| Packaged workbench refresh | `edc6cfd` |
| Clean notarized build | exit 0; Gatekeeper accepted |
| Evidence + claim updates | `docs/operations/evidence/2026-08-13-desktop-public-release/` |

## Artifacts produced

- `desktop/macos/dist/Aptus.app.zip` SHA-256 `41afcb0cce4fca7a32374fe849b678c8723cb6584cf9e3140cc16bb1bf0a08e5`
- `desktop/macos/dist/Aptus-macOS-arm64.dmg` SHA-256 `bf28fe7416b24d1d000c39a83af317b1062fdf74c32fe499341bca9d3ea13834`
- App notary `ad36bb3a-5e1f-420f-ae1f-f2be1d84960a` Accepted
- DMG notary `d3794efb-e44b-4d8d-886d-1e5c32d3f0df` Accepted

## Explicit non-claims

- Not Aptus 0.2 product release  
- Not a published download URL  
- Not Intel / universal  
- Not a 10× notarized stability gate  
- First dirty-tree notarization of `db59ed9` is not this identity  

## Deliberately not done

- `tools/repeat_desktop_release_gate.zsh` with notarization (expensive; historical ad-hoc 10× stands)  
- GitHub Release upload  

## Next phase allowed?

- **Yes**, only after owner chooses one M7 axis.  
- **First action:** freeze that identity table; do not start M7 in this packet.  
