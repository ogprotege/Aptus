# Aptus for Mac Code and Architecture Review

**Last updated:** 2026-07-22
**Scope:** AppKit host, WebKit boundary, Python sidecar, React integration, packaging, and macOS validation
**Review posture:** Current source after all review-driven hardening

## Executive summary

Aptus now has a coherent macOS desktop application. AppKit owns the application and process lifecycle. WebKit presents the existing React workbench. The bundled Python sidecar remains authoritative for planning, compilation, validation, and job rules. The native bridge is limited to readiness, file selection, and Finder actions.

No open Critical code defect or Important architecture defect remains in the reviewed source. The shutdown state machine is bounded and identity-aware. It also follows captured child subtrees after the original process exits. React now proves authenticated bootstrap and a committed UI marker. The Release build now enforces isolated Python 3.12 tests, current React assets, native tests, and the hash-locked sidecar environment.

Two blockers remain for public distribution. Developer ID signing and notarization have not been performed. The declared macOS 13 support floor has not been tested on macOS 13. Neither blocker prevents local use of the definitive ad-hoc-signed application.

## Critical findings

### RB-1 | Critical for public distribution | Developer ID and notarization are unverified

**Evidence:**

- `desktop/macos/build.sh:181-195` signs the nested helper and outer application with either an ad-hoc identity or a supplied Developer ID identity.
- `desktop/macos/AptusBackend.spec:11-16,44-59` passes that Developer ID into PyInstaller for embedded binaries.
- `desktop/macos/README.md:47-56` correctly identifies the default signature as local-only.
- The last local artifact passed strict code-signature verification, but it reported `Signature=adhoc` and no Team ID. `spctl --assess` rejected it.

**Impact:** The application is usable for local evaluation. It is not ready for ordinary public installation. Gatekeeper behavior, notarization, hardened-runtime compatibility, and quarantine launch remain unproven.

**Required before public release:** Build with a real Developer ID Application identity. Notarize and staple the application or DMG. Run `spctl --assess`. Install from the DMG under quarantine and launch it on a clean machine. PyInstaller must receive the same identity because embedded one-file binaries cannot be repaired only by post-processing. See the [official PyInstaller macOS signing guidance](https://pyinstaller.org/en/v6.17.0/feature-notes.html#macos-binary-code-signing).

### RB-2 | Critical for the stated support floor | macOS 13 is configured but not runtime-tested

**Evidence:**

- `desktop/macos/project.yml:5-12` sets a macOS 13 deployment target.
- `desktop/macos/README.md:16-22` claims macOS 13 or newer on Apple Silicon.
- This review ran on macOS 26.5.2 arm64 with Xcode 26.6. No macOS 13 host or VM was available.

**Impact:** The Mach-O deployment floor is correct. Runtime compatibility at that floor is not established. WebKit behavior, cookies, panels, PyInstaller extraction, and application shutdown may still differ.

**Required before public release:** Install the current DMG on a clean macOS 13 Apple Silicon machine. Repeat on a current supported macOS version. Exercise startup, all five workbench stages, native panels, Finder reveal, failure and Retry, quit, and persisted-state relaunch.

## Important findings

No open Important source finding remains after the final hardening pass.

## Minor findings

### MIN-1 | Make the application version single-source

Version `0.2.0` is repeated in `pyproject.toml:7`, `src/aptus/__init__.py:3`, several API responses in `src/aptus/api.py`, `web/package.json:4`, and `desktop/macos/Resources/Info.plist:21-24`.

The default package probe catches a Python and app version mismatch, so divergence does not pass silently. Maintenance still requires several coordinated edits. Generate the API, web, and bundle versions from one release source. Keep a packaging assertion for the final Python and application versions.

## Findings fixed during review

| Original finding | Resolution and current evidence |
|---|---|
| IMP-1: shutdown could wait forever or leave descendants | `BackendController.swift:253-348` now uses bounded graceful and forced deadlines. `BackendProcessTree.swift:20-90` tracks exact PID and start-time identities, rescans every captured live subtree, freezes descendants, and then sends `SIGKILL`. `AptusApplication.swift:3-56` returns `.terminateLater` until shutdown completes. `BackendControllerIntegrationTests.swift:180-299` covers TERM-resistant stop, restart ordering, cooperative-root exit, and late nested forks. |
| IMP-2: a finished HTML navigation could pass the launch probe | `WebViewController.swift:36-56,96-107,156-168` requires both same-origin document completion and the React signal. `DesktopBridge.swift:57-70,124-133,193-204` validates a versioned marker. `App.tsx:280-350,740-745` emits the marker only after authenticated bootstrap succeeds. `build.sh:197-241` requires document, React, marker, window, backend, and teardown evidence. |
| IMP-3: the desktop build omitted the Python product gate | `build.sh:36-47,63-73` makes isolated locked Python 3.12 tests mandatory for Release. Release also rejects `--skip-tests`, `--skip-web`, and `APTUS_PYINSTALLER_PYTHON`. `PackagingContractTests.swift:16-27` preserves those rules. |
| Hash-lock enforcement lacked regression coverage | `build.sh:101-112` requires hashes, accepts wheels only, and installs the local project without dependency resolution. `PackagingContractTests.swift:29-65` checks every locked requirement stanza for a SHA-256 hash and checks the required installer flags. |
| External popup requests did not require a user gesture | `WebViewController.swift:31-33,126-189` applies one `.linkActivated` HTTP(S) policy to ordinary and new-window navigation. `BackendModelsTests.swift:60-86` covers allowed and rejected cases. |
| Backend logs could grow without a retention bound | `ApplicationPaths.swift:105-180` rotates a 2 MiB tail, retains two private archives, and repairs oversized existing archives. `ApplicationPathsTests.swift:102-183` covers rotation, count, permissions, no-op behavior, and archive repair. |
| Native bridge promises could remain pending forever | `DesktopBridge.swift:16-75` applies 30-second request and five-minute modal timeouts. `DesktopBridge.swift:84-133,155-305` returns typed errors for correlatable malformed or unsupported requests. `DesktopBridgeTests.swift:5-133` preserves the contract. |
| Native application bootstrap did not retain its delegate | `main.swift:3-7` creates, assigns, and retains the delegate before entering the AppKit run loop. |
| Retry could retain a dead backend process | Backend lifecycle mutations now converge on the main queue. Termination clears the old process before Retry. Integration tests cover unexpected exit and fresh restart. |
| Desktop execution limits existed only in React | `src/aptus/desktop.py:93-99` disables execution in the sidecar. `src/aptus/api.py` rejects runtime validation and every job submission with `desktop_execution_disabled`. |
| Desktop API access lacked a dedicated authenticated boundary | The native host creates a random token. The sidecar binds only `127.0.0.1`. Every desktop route requires the private session cookie and hardened response headers. |
| Production could honor ambient executable overrides | `BackendModels.swift:97-143` compiles overrides and repository fallbacks only under `DEBUG`. Release resolves the bundled sidecar. |
| The desktop cookie persisted beyond the process | `WebViewController.swift:4-16,76-123` uses an HttpOnly, SameSite Strict session cookie inside a non-persistent WebKit store. |
| Release builds could omit or resolve an untracked sidecar | `build.sh:32-47` rejects Release bypasses. `requirements-build.lock` supplies exact hashed wheels. Developer ID reaches PyInstaller at build time. |
| Application paths relied on best-effort permissions | `ApplicationPaths.swift` verifies directory and file types plus mode `0700` and `0600`. Tests repair permissive paths and reject a non-regular log. |
| The DMG lacked install affordance and layout verification | `build.sh:244-262` stages `Aptus.app` plus an `/Applications` link, mounts the image read-only, and verifies both entries. |

## Architecture assessment

The selected structure should remain:

```text
AppKit lifecycle and native panels
              |
              v
WKWebView with exact-origin bridge
              |
              v
Authenticated ephemeral loopback
              |
              v
FastAPI and the existing Aptus core
```

This split avoids a second domain model in Swift. React owns the five-stage workflow and presentation state. Python owns facts, feasibility, compilation, validation, jobs, leases, and evidence. AppKit owns launch, failure recovery, process shutdown, local paths, file panels, Finder actions, and navigation policy.

The bridge remains narrow and exact-origin. Browser-only Aptus continues to work because the React client feature-detects the complete native object. The macOS execution boundary also exists in the server, so a UI bypass cannot launch CUDA work locally.

The application is not sandboxed (`desktop/macos/Resources/Aptus.entitlements:1-5`). That is acceptable for a direct Developer ID tool that must read selected datasets and write bundles. The trusted-local-user model must remain explicit. The desktop API must not become a remote or multi-user service without a different authorization and filesystem-isolation design.

## Verification performed

- Isolated locked Python 3.12.13 gate: **205 tests passed**.
- React gate: **41 tests passed**, followed by a passing TypeScript check.
- Native macOS gate: **35 tests passed**. This includes late-fork descendant cleanup, TERM-resistant stop and restart, archive repair, bridge timeouts and typed errors, React readiness, navigation policy, and packaging contracts.
- Release bypass checks: `--skip-tests`, `--skip-web`, and `APTUS_PYINSTALLER_PYTHON` each failed before build cleanup with exit code 2.
- `zsh -n desktop/macos/build.sh` passed.
- No late-fork or TERM-resistant fixture process remained after the native suite.
- The definitive full build completed after the final shutdown, log, and Release-gate changes.
- Its packaged probe reported `backendReady=true`, `windowVisible=true`, `workbenchLoaded=true`, `reactReady=true`, marker `aptus-workbench-v1`, host `127.0.0.1`, and version `0.2.0`.
- `Aptus.app/Contents/MacOS/Aptus` and the bundled `aptus-desktop` helper are arm64 Mach-O executables.
- `codesign --verify --deep --strict` passed. The signature is ad hoc, as intended for this local artifact.
- The DMG mounted read-only and contained `Aptus.app` plus an `/Applications` link.
- Final DMG SHA-256: `551a22b9c107092c93c716352564073b21c0ecbedcce8a93e69a7a2696e90245`.
- The probe session directory was removed. No Aptus helper or lifecycle fixture remained alive.

Not verified in this final source pass:

- Developer ID signing, notarization, stapling, or successful Gatekeeper assessment.
- Installation and launch under quarantine on a clean machine.
- Runtime behavior on macOS 13.
- CUDA execution, which is intentionally outside the Mac application and remains governed by repository release gates.

## Next steps

1. Complete Developer ID signing, notarization, quarantine installation, and the clean macOS 13 matrix.
2. Exercise every workbench stage on the installed application and preserve dated evidence.
3. Consolidate release version sources before the next version bump.

The reviewed source is suitable for a local macOS application. Do not describe the artifact as publicly distributable until RB-1 and RB-2 have dated passing evidence.
