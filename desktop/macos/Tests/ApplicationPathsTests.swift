import Foundation
import XCTest
@testable import Aptus

final class ApplicationPathsTests: XCTestCase {
    func testPathsStayInsideDedicatedAptusRoots() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let support = temporary.appendingPathComponent("support", isDirectory: true)
        let logs = temporary.appendingPathComponent("logs", isDirectory: true)
        let caches = temporary.appendingPathComponent("caches", isDirectory: true)
        let paths = try ApplicationPaths(
            applicationSupportRoot: support,
            logsRoot: logs,
            cachesRoot: caches,
            sessionIdentifier: "known-session"
        )

        XCTAssertEqual(paths.stateDirectory.path, support.appendingPathComponent("Aptus/state").path)
        XCTAssertEqual(paths.logFile.path, logs.appendingPathComponent("Aptus/backend.log").path)
        XCTAssertEqual(
            paths.readyFile.path,
            caches.appendingPathComponent("Aptus/sessions/known-session/ready.json").path
        )

        try paths.prepare()
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.stateDirectory.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.logFile.path))
    }

    func testRemovingSessionDoesNotRemovePersistentState() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "ephemeral"
        )
        try paths.prepare()
        paths.removeEphemeralSession()
        XCTAssertFalse(FileManager.default.fileExists(atPath: paths.sessionDirectory.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: paths.stateDirectory.path))
    }

    func testPrepareRepairsAndVerifiesPermissiveExistingPaths() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "permissions"
        )
        try paths.prepare()

        let manager = FileManager.default
        let directories = [
            paths.applicationSupport,
            paths.stateDirectory,
            paths.logsDirectory,
            paths.sessionDirectory.deletingLastPathComponent().deletingLastPathComponent(),
            paths.sessionDirectory.deletingLastPathComponent(),
            paths.sessionDirectory,
        ]
        for directory in directories {
            try manager.setAttributes([.posixPermissions: 0o777], ofItemAtPath: directory.path)
        }
        try manager.setAttributes([.posixPermissions: 0o666], ofItemAtPath: paths.logFile.path)

        try paths.prepare()

        for directory in directories {
            let attributes = try manager.attributesOfItem(atPath: directory.path)
            XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o700)
        }
        let logAttributes = try manager.attributesOfItem(atPath: paths.logFile.path)
        XCTAssertEqual((logAttributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
    }

    func testPrepareRejectsNonRegularLogPath() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "wrong-type"
        )
        try FileManager.default.createDirectory(
            at: paths.logFile,
            withIntermediateDirectories: true
        )
        XCTAssertThrowsError(try paths.prepare())
    }

    func testPrepareRotatesOnlyTwoBoundedLogTails() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "rotation"
        )
        try paths.prepare(maximumRetainedLogBytes: 8)

        for contents in ["first-12345678", "second-ABCDEFGH", "third-abcdefgh"] {
            try Data(contents.utf8).write(to: paths.logFile)
            try paths.prepare(maximumRetainedLogBytes: 8)
        }

        let activeAttributes = try FileManager.default.attributesOfItem(atPath: paths.logFile.path)
        XCTAssertEqual((activeAttributes[.size] as? NSNumber)?.uint64Value, 0)
        XCTAssertEqual(try String(contentsOf: paths.rotatedLogFiles[0], encoding: .utf8), "abcdefgh")
        XCTAssertEqual(try String(contentsOf: paths.rotatedLogFiles[1], encoding: .utf8), "ABCDEFGH")
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: paths.logsDirectory.appendingPathComponent("backend.log.3").path
        ))

        for archive in paths.rotatedLogFiles {
            let attributes = try FileManager.default.attributesOfItem(atPath: archive.path)
            XCTAssertLessThanOrEqual((attributes[.size] as? NSNumber)?.uint64Value ?? .max, 8)
            XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        }
    }

    func testPrepareLeavesLogBelowRotationThresholdInPlace() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "no-rotation"
        )
        try paths.prepare(maximumRetainedLogBytes: 8)
        try Data("short".utf8).write(to: paths.logFile)

        try paths.prepare(maximumRetainedLogBytes: 8)

        XCTAssertEqual(try String(contentsOf: paths.logFile, encoding: .utf8), "short")
        XCTAssertTrue(paths.rotatedLogFiles.allSatisfy {
            !FileManager.default.fileExists(atPath: $0.path)
        })
    }

    func testPrepareCapsOversizedExistingArchives() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "archive-cap"
        )
        try paths.prepare(maximumRetainedLogBytes: 8)
        try Data("first-12345678".utf8).write(to: paths.rotatedLogFiles[0])
        try Data("second-ABCDEFGH".utf8).write(to: paths.rotatedLogFiles[1])

        try paths.prepare(maximumRetainedLogBytes: 8)

        XCTAssertEqual(
            try String(contentsOf: paths.rotatedLogFiles[0], encoding: .utf8),
            "12345678"
        )
        XCTAssertEqual(
            try String(contentsOf: paths.rotatedLogFiles[1], encoding: .utf8),
            "ABCDEFGH"
        )
        for archive in paths.rotatedLogFiles {
            let attributes = try FileManager.default.attributesOfItem(atPath: archive.path)
            XCTAssertEqual((attributes[.size] as? NSNumber)?.uint64Value, 8)
            XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        }
    }
}
