# Desktop engineering acceptance, 2026-07-27

> **Status:** Passed local engineering acceptance | **Authority:** Immutable evidence record | **Applies to:** Aptus 0.2 Mac packaging at the tested commit | **Audience:** Release reviewers and maintainers | **Last reviewed:** 2026-07-27 | **Review by:** After any desktop, packaging, dependency, test-gate, or signing change

> **Result:** 10 of 10 consecutive clean release builds passed
> **Tested commit:** `1038ecdd13103418ef1135e1ced634c10370a961`
> **Host:** Apple M5 Pro, 64 GiB unified memory, macOS 26.6 (25G72)
> **Scope:** Local build stability and ad-hoc-signed review packages. This is not public release approval.

## Acceptance result

The repeated release gate ran `desktop/macos/build.sh` ten times from a clean
checkout. Every iteration rebuilt all product layers, ran all required tests,
created a fresh arm64 application and disk image, verified the application
signature, verified the DMG, and recorded the final artifact hashes.

Each iteration passed:

- 327 Python tests;
- generated OpenAPI freshness, client-contract, and version-parity checks;
- 61 React tests, including accessibility and responsive-layout checks;
- TypeScript checking and the production web build;
- 78 native tests, including backend shutdown containment and packaging
  contracts;
- hash-locked Python 3.12 sidecar creation and PyInstaller packaging;
- native Release compilation for arm64 with a macOS 15 deployment floor;
- packaged launch, authenticated backend readiness, React readiness, and clean
  session shutdown;
- strict nested and outer code-signature verification; and
- DMG creation and verification.

Across ten repetitions, that is 4,660 repeated test executions. This count
describes repeated stability coverage, not 4,660 unique tests.

## Timing and size telemetry

The ten builds took 581 seconds in total. The mean was 58.1 seconds, the median
was 58 seconds, and the range was 55 to 63 seconds.

| Iteration | Duration | Application ZIP SHA-256 | DMG SHA-256 |
| ---: | ---: | --- | --- |
| 1 | 55 s | `e8062f3c0511b0e9e6babac12f52e515c74ecac1ca32504920bbe36e62ad9b38` | `a0e25c03009a56159115cd42ca424a5c830bc3a5dfbc08b47b8a9eb8d9d05863` |
| 2 | 58 s | `43014fcb983efedc2ad536d087eea951c0a518f0fcc480ffb96d87363ad82b0e` | `c4aebb64fdd859ab7155cb3b65cf4a939c46ff898e960c30c0490a59dad28dba` |
| 3 | 59 s | `c66cf30c4c08d72e1e2551cc816aca186be4bd98e8c5fa8634b6b61aab066405` | `444339a8325259c9946ca17a5e67b2656e715f35a48a8348f9d31e10f631fb1f` |
| 4 | 63 s | `c6bf3624f0155632d541d2b98c8822d8ccc206a90c02c3c565dcfe895e1dd563` | `2e3eb892687febbcc785b6a43a040ea2272835b897ff1836a73f4ffe115d8d4d` |
| 5 | 58 s | `28553443d3eca06665fc75ea9a042d7222bf5f014f952ba32ace442977c83991` | `e185ca9e29349e4151e17d9e5687c40ef5bb1f7b0de419fd28403ed77f76904e` |
| 6 | 59 s | `cd691862a9acba426635971840267910ecebdcf25c074ad5afdcf3dbf31a7fb2` | `afacab9635dca1ae46e386e241048eeccfa3267ccf0f128ba78e4674b5f68993` |
| 7 | 55 s | `3232f78e9037cbe0bdb551f9943111b35eef371333068029921cc2e96497c873` | `a652c29d6e6e4b1d2c193e7201571ba7270c8bac961367fd50509e1b1b6d5e47` |
| 8 | 57 s | `e1738cdef48c80f222369528081c98fe8d509e2daf83172bfecf19f17e2290aa` | `c55ad221cd40c43a417dbf2af546b4223a38393c40966b91fff039a2a6d3279c` |
| 9 | 56 s | `a06a76c4cfa21e7f584de8f86d08eb3b08256e105da40416d1f3a226a835fdae` | `a7caee6bab64e6949e75b9b889ba4165c1a7304a52c9ddf54361c14cee4b6849` |
| 10 | 61 s | `69d57014a56c4809a44b59e41779d19c8c0c0acebbcf92bce76e2e7548772db7` | `91001ac3febe92bf4925d440c796bd2f627158a4050a2ff0f5de78fed24eaaa7` |

The final uncompressed `Aptus.app` directory occupied 17,480 KiB. Its ZIP was
16,423,499 bytes. The DMG was 16,733,602 bytes. These are host-specific
acceptance measurements. They do not predict another machine's build time,
runtime throughput, training speed, or model quality.

## Final artifact identity

| Artifact | SHA-256 |
| --- | --- |
| `Aptus.app.zip` | `69d57014a56c4809a44b59e41779d19c8c0c0acebbcf92bce76e2e7548772db7` |
| `Aptus-macOS-arm64.dmg` | `91001ac3febe92bf4925d440c796bd2f627158a4050a2ff0f5de78fed24eaaa7` |
| `RELEASE-GATE.tsv` | `7c408bdf124aeeaaccbc1150e7e9004a54d50a3493772c47c18c2fa9c42108a8` |
| `release-gate-logs.zip` | `95487efaaca5a9b38e38ca82f5dca5b28558ad862a495937002a4f66657592ca` |

The application identity was `com.aptus.desktop`, version `0.2.0`, build `1`.
The application and embedded backend were arm64. The strict
`codesign --verify --deep --strict` check passed, and the signature was ad-hoc
with no Team ID. `hdiutil verify` passed for the DMG.

## Source and PR boundary

This packet binds the first repeated gate to
`1038ecdd13103418ef1135e1ced634c10370a961`. The branch was later rebased onto a
newer `main`, so these hashes must not be attributed to the final pull-request
head. A new local rerun can bind its own ignored artifacts and PR report, but it
does not alter this historical packet.

The pull-request workflow independently builds GitHub's synthetic merge commit
on an arm64 macOS 26 runner. It records `GITHUB_SHA` in `COMMIT`, verifies the
app, ZIP, DMG, and checksums, and uploads them as a 30-day GitHub Actions
artifact. Push workflows bind the pushed commit. Pull-request workflows bind
the exact merge candidate tested by GitHub, not the branch head.

Local binaries and full logs live under ignored `desktop/macos/dist/` and are
not committed to Git. This record retains their identities and bounded results
without adding review binaries to repository history.

## Dependency-audit boundary

`npm audit --omit=dev` reported zero production dependency advisories. The full
development audit reported four high-severity transitive advisories through the
OpenAPI generation toolchain: `@redocly/openapi-core`, `js-yaml`, `minimatch`,
and `brace-expansion`. The generator processes the trusted checked-in OpenAPI
document, but that does not erase the advisories. They remain tracked release
engineering debt.

## Open gates

- The packages are not Developer ID signed.
- The app and DMG are not notarized or stapled.
- Gatekeeper public-distribution assessment has not passed.
- No qualifying CUDA target-host pilot or full run exists.
- The tested desktop commit predates the final branch head. The current local
  rerun and exact merge-candidate CI must pass before PR handoff.

Aptus 0.2 therefore remains an unreleased engineering preview.

## Reproduce

Run the stability gate only from a clean checkout:

```bash
tools/repeat_desktop_release_gate.zsh 10
```

The script writes `RELEASE-GATE.tsv` and `release-gate-logs.zip` beside the app,
ZIP, DMG, `SHA256SUMS`, and `COMMIT` under `desktop/macos/dist/`.

## Related documentation

- [MLX-LM target-host acceptance](../2026-07-27-mlx-lm-acceptance/README.md)
- [Release gates](../../release-gates.md)
- [Release evidence template](../../release-evidence-template.md)
- [Current capabilities](../../../product/current-capabilities.md)
