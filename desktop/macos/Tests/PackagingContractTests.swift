import Foundation
import XCTest

final class PackagingContractTests: XCTestCase {
    func testProjectUsesMacOS15FallbackWithoutPinningAnOldSDK() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let project = try String(
            contentsOf: directory.appendingPathComponent("project.yml"),
            encoding: .utf8
        )
        XCTAssertTrue(project.contains("macOS: \"15.0\""))
        XCTAssertTrue(project.contains("MACOSX_DEPLOYMENT_TARGET: \"15.0\""))
        XCTAssertFalse(project.contains("13.0"))
        XCTAssertFalse(project.contains("SDKROOT"))
    }

    func testBuildTargetsAppleSiliconAndUsesTheInstalledSDK() throws {
        let script = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("build.sh")
        let contents = try String(contentsOf: script, encoding: .utf8)
        XCTAssertTrue(contents.contains("platform=macOS,arch=arm64"))
        XCTAssertTrue(contents.contains("ARCHS=arm64"))
        XCTAssertFalse(contents.contains("-sdk macosx"))
    }

    func testPyInstallerUsesTheRequestedDeveloperIDForEmbeddedBinaries() throws {
        let spec = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("AptusBackend.spec")
        let contents = try String(contentsOf: spec, encoding: .utf8)
        XCTAssertTrue(contents.contains("os.environ.get(\"APTUS_CODESIGN_IDENTITY\")"))
        XCTAssertTrue(contents.contains("requested_codesign_identity != \"-\""))
        XCTAssertTrue(contents.contains("codesign_identity=PYINSTALLER_CODESIGN_IDENTITY"))
    }

    func testReleaseBuildCannotOmitThePackagedBackend() throws {
        let script = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("build.sh")
        let contents = try String(contentsOf: script, encoding: .utf8)
        XCTAssertTrue(contents.contains("Release builds cannot use --skip-backend."))
        XCTAssertTrue(contents.contains("Release builds cannot use --skip-tests."))
        XCTAssertTrue(contents.contains("Release builds cannot use --skip-web."))
        XCTAssertTrue(contents.contains("Release builds cannot use APTUS_PYINSTALLER_PYTHON."))
        XCTAssertTrue(contents.contains("uv run --isolated --python 3.12 --locked"))
    }

    func testDefaultSidecarBuildUsesTheTrackedPythonLock() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let script = try String(
            contentsOf: directory.appendingPathComponent("build.sh"),
            encoding: .utf8
        )
        let lock = try String(
            contentsOf: directory.appendingPathComponent("requirements-build.lock"),
            encoding: .utf8
        )
        XCTAssertTrue(script.contains("requirements-build.lock"))
        XCTAssertTrue(script.contains("--require-hashes"))
        XCTAssertTrue(script.contains("--only-binary :all:"))
        XCTAssertTrue(script.contains("--no-deps"))
        XCTAssertTrue(script.contains("--no-build-isolation"))
        XCTAssertTrue(lock.contains("pyinstaller=="))
        XCTAssertTrue(lock.contains("--hash=sha256:"))
        XCTAssertFalse(lock.contains("aptus @"))

        let lines = lock.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let requirementIndexes = lines.indices.filter { index in
            let line = lines[index]
            return !line.hasPrefix(" ") && !line.hasPrefix("#") && line.contains("==")
        }
        XCTAssertFalse(requirementIndexes.isEmpty)
        for (position, index) in requirementIndexes.enumerated() {
            let end = position + 1 < requirementIndexes.count
                ? requirementIndexes[position + 1]
                : lines.endIndex
            let stanza = lines[index ..< end].joined(separator: "\n")
            XCTAssertTrue(
                stanza.contains("--hash=sha256:"),
                "Missing a SHA-256 hash for locked requirement: \(lines[index])"
            )
        }
    }

    func testPackagedProbeRequiresTheVersionedReactReadyHandshake() throws {
        let script = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("build.sh")
        let contents = try String(contentsOf: script, encoding: .utf8)
        XCTAssertTrue(contents.contains("plutil -extract reactReady"))
        XCTAssertTrue(contents.contains("plutil -extract workbenchMarker"))
        XCTAssertTrue(contents.contains("aptus-workbench-v1"))
    }

    func testReleaseResolverCompilesAmbientBackendOverridesOut() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/BackendModels.swift")
        let contents = try String(contentsOf: source, encoding: .utf8)
        XCTAssertTrue(contents.contains("#if DEBUG\n        if let explicit = environment[\"APTUS_DESKTOP_BACKEND\"]"))
        XCTAssertTrue(contents.contains("#if DEBUG\n        if let pythonPath = environment[\"APTUS_DESKTOP_PYTHON\"]"))
        XCTAssertTrue(contents.contains("#if DEBUG\n        if let repositoryRoot = bundle.object"))
    }
}
