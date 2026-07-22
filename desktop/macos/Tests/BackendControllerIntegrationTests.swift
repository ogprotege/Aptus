import Foundation
import Darwin
import XCTest
@testable import Aptus

final class BackendControllerIntegrationTests: XCTestCase {
    func testDevelopmentBackendReachesAuthenticatedReadyState() throws {
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .standardizedFileURL
        let python = repository.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw XCTSkip("The repository Python environment is unavailable.")
        }
        let desktopModule = repository.appendingPathComponent("src/aptus/desktop.py")
        guard FileManager.default.fileExists(atPath: desktopModule.path) else {
            throw XCTSkip("The Aptus desktop Python entrypoint is unavailable.")
        }

        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("aptus-native-integration-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "session"
        )
        let bootstrap = "import runpy,sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('aptus.desktop', run_name='__main__')"
        let executable = BackendExecutable(
            url: python,
            leadingArguments: ["-I", "-c", bootstrap, repository.appendingPathComponent("src").path]
        )
        let controller = BackendController(
            pathsFactory: { paths },
            executableResolver: { executable }
        )
        let ready = expectation(description: "Authenticated desktop backend became ready")
        let stopped = expectation(description: "Desktop backend stopped")
        controller.onStateChange = { state in
            switch state {
            case let .ready(session):
                XCTAssertEqual(session.origin.host, "127.0.0.1")
                XCTAssertGreaterThanOrEqual(Data(base64Encoded: session.token)?.count ?? 0, 32)
                XCTAssertTrue(FileManager.default.fileExists(atPath: paths.readyFile.path))
                ready.fulfill()
                controller.stop()
                XCTAssertFalse(FileManager.default.fileExists(atPath: paths.sessionDirectory.path))
            case .stopped:
                stopped.fulfill()
            case let .failed(message):
                XCTFail(message)
            case .starting, .stopping:
                break
            }
        }
        controller.start()
        wait(for: [ready, stopped], timeout: 20)
    }

    func testUnexpectedBackendExitCanRetryWithAFreshProcess() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("aptus-native-retry-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "session"
        )
        let failing = BackendExecutable(
            url: URL(fileURLWithPath: "/usr/bin/false"),
            leadingArguments: []
        )
        let waiting = BackendExecutable(
            url: URL(fileURLWithPath: "/bin/sleep"),
            leadingArguments: ["30"]
        )
        var resolutionCount = 0
        let controller = BackendController(
            pathsFactory: { paths },
            executableResolver: {
                resolutionCount += 1
                return resolutionCount == 1 ? failing : waiting
            }
        )
        let unexpectedFailure = expectation(description: "Unexpected exit reported a failure")
        let retryStarted = expectation(description: "Retry launched a fresh process")
        let stopped = expectation(description: "Retried process stopped cleanly")
        var startingCount = 0
        var requestedRetry = false
        var retriedProcessStarted = false
        controller.onStateChange = { state in
            switch state {
            case .starting:
                startingCount += 1
                if startingCount == 2 {
                    retriedProcessStarted = true
                    retryStarted.fulfill()
                    controller.stop()
                }
            case .failed:
                guard !requestedRetry else {
                    XCTFail("The fresh retry process exited unexpectedly.")
                    return
                }
                requestedRetry = true
                unexpectedFailure.fulfill()
                controller.restart()
            case .stopped:
                if retriedProcessStarted {
                    stopped.fulfill()
                }
            case .ready, .stopping:
                break
            }
        }

        controller.start()
        wait(for: [unexpectedFailure, retryStarted, stopped], timeout: 10)
        XCTAssertEqual(resolutionCount, 2)
    }

    func testStartupValidationFailureRemainsFailedAfterChildTermination() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("aptus-native-failure-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "session"
        )
        let script = #"printf '{"host":"0.0.0.0","port":1,"version":"test"}' > "$3"; chmod 600 "$3"; sleep 30"#
        let invalidReadiness = BackendExecutable(
            url: URL(fileURLWithPath: "/bin/sh"),
            leadingArguments: ["-c", script]
        )
        let controller = BackendController(
            pathsFactory: { paths },
            executableResolver: { invalidReadiness }
        )
        let failed = expectation(description: "Invalid readiness reported a failure")
        let remainedFailed = expectation(description: "Failure survived child termination")
        let stopped = expectation(description: "Failed controller stopped on request")
        var failureObserved = false
        var cleanupRequested = false
        controller.onStateChange = { state in
            switch state {
            case .failed:
                guard !failureObserved else { return }
                failureObserved = true
                failed.fulfill()
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    guard case .failed = controller.state else {
                        XCTFail("Child termination overwrote the startup failure state.")
                        remainedFailed.fulfill()
                        return
                    }
                    remainedFailed.fulfill()
                    cleanupRequested = true
                    controller.stop()
                }
            case .stopped:
                if cleanupRequested {
                    stopped.fulfill()
                }
            case .starting, .ready, .stopping:
                break
            }
        }

        controller.start()
        wait(for: [failed, remainedFailed, stopped], timeout: 10)
    }

    func testStopForceTerminatesATermResistantProcessTreeBeforeCompleting() throws {
        let fixture = try makeTermResistantFixture(prefix: "aptus-native-force-stop")
        defer { fixture.cleanup() }
        let controller = BackendController(
            pathsFactory: { fixture.paths },
            executableResolver: { fixture.executable },
            shutdownGraceInterval: 0.15,
            shutdownForceInterval: 0.75
        )

        controller.start()
        let pids = try fixture.waitForProcessIdentifiers()
        XCTAssertTrue(isProcessAlive(pids.parent))
        XCTAssertTrue(isProcessAlive(pids.child))

        let stopped = expectation(description: "Forced process-tree shutdown completed")
        let startedAt = Date()
        controller.stop {
            XCTAssertFalse(self.isProcessAlive(pids.parent))
            XCTAssertFalse(self.isProcessAlive(pids.child))
            XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.paths.sessionDirectory.path))
            stopped.fulfill()
        }
        XCTAssertEqual(controller.state, .stopping)

        wait(for: [stopped], timeout: 2)
        XCTAssertGreaterThanOrEqual(Date().timeIntervalSince(startedAt), 0.1)
        XCTAssertEqual(controller.state, .stopped)
    }

    func testRestartWaitsForTermResistantProcessTreeBeforeLaunchingReplacement() throws {
        let fixture = try makeTermResistantFixture(prefix: "aptus-native-force-restart")
        defer { fixture.cleanup() }
        let replacement = BackendExecutable(
            url: URL(fileURLWithPath: "/bin/sleep"),
            leadingArguments: ["30"]
        )
        var resolutionCount = 0
        let controller = BackendController(
            pathsFactory: { fixture.paths },
            executableResolver: {
                resolutionCount += 1
                return resolutionCount == 1 ? fixture.executable : replacement
            },
            shutdownGraceInterval: 0.15,
            shutdownForceInterval: 0.75
        )

        controller.start()
        let pids = try fixture.waitForProcessIdentifiers()
        let replacementStarted = expectation(description: "Replacement started after forced cleanup")
        let replacementStopped = expectation(description: "Replacement stopped")
        // The initial start occurs before the observer is installed.
        var startingCount = 1
        controller.onStateChange = { state in
            switch state {
            case .starting:
                startingCount += 1
                guard startingCount == 2 else { return }
                XCTAssertFalse(self.isProcessAlive(pids.parent))
                XCTAssertFalse(self.isProcessAlive(pids.child))
                replacementStarted.fulfill()
                controller.stop()
            case .stopped:
                if startingCount == 2 {
                    replacementStopped.fulfill()
                }
            case let .failed(message):
                XCTFail(message)
            case .ready, .stopping:
                break
            }
        }

        controller.restart()
        wait(for: [replacementStarted, replacementStopped], timeout: 3)
        XCTAssertEqual(resolutionCount, 2)
    }

    func testCooperativeRootCannotLoseLateForkingChildDescendants() throws {
        let fixture = try makeLateForkingFixture()
        defer { fixture.cleanup() }
        let controller = BackendController(
            pathsFactory: { fixture.paths },
            executableResolver: { fixture.executable },
            shutdownGraceInterval: 0.5,
            shutdownForceInterval: 0.75
        )

        controller.start()
        let initialPIDs = try fixture.waitForInitialProcessIdentifiers()
        guard let rootIdentity = BackendProcessTree.identity(for: initialPIDs.root),
              let childIdentity = BackendProcessTree.identity(for: initialPIDs.child) else {
            return XCTFail("The late-fork fixture processes were not running.")
        }
        var inventory: Set<BackendProcessIdentity> = [rootIdentity, childIdentity]
        let stopped = expectation(description: "Late-fork process tree stopped")
        controller.stop {
            XCTAssertTrue(BackendProcessTree.living(inventory).isEmpty)
            for pid in fixture.allRecordedPIDs() {
                XCTAssertFalse(self.isProcessAlive(pid), "Process \(pid) survived shutdown.")
            }
            stopped.fulfill()
        }

        let latePIDs = try fixture.waitForLateProcessIdentifiers()
        XCTAssertTrue(fixture.waitUntilProcessExits(initialPIDs.root, timeout: 0.25))
        XCTAssertTrue(isProcessAlive(initialPIDs.child))
        inventory.formUnion(BackendProcessTree.expanding([childIdentity]))
        let inventoriedPIDs = Set(inventory.map(\.pid))
        XCTAssertEqual(inventoriedPIDs, Set([
            initialPIDs.root,
            initialPIDs.child,
            latePIDs.lateChild,
            latePIDs.nestedChild,
        ]))

        wait(for: [stopped], timeout: 2)
        XCTAssertEqual(controller.state, .stopped)
    }

    private func makeTermResistantFixture(prefix: String) throws -> TermResistantFixture {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "session"
        )
        let parentPIDFile = temporary.appendingPathComponent("parent.pid")
        let childPIDFile = temporary.appendingPathComponent("child.pid")
        let script = """
        trap '' TERM
        echo $$ > '\(parentPIDFile.path)'
        /bin/sh -c 'trap "" TERM; echo $$ > "\(childPIDFile.path)"; while :; do sleep 1; done' &
        while :; do sleep 1; done
        """
        return TermResistantFixture(
            root: temporary,
            paths: paths,
            executable: BackendExecutable(
                url: URL(fileURLWithPath: "/bin/sh"),
                leadingArguments: ["-c", script]
            ),
            parentPIDFile: parentPIDFile,
            childPIDFile: childPIDFile
        )
    }

    private func makeLateForkingFixture() throws -> LateForkingFixture {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("aptus-native-late-fork-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: temporary,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "session"
        )
        let trigger = temporary.appendingPathComponent("root-stopped")
        let rootPIDFile = temporary.appendingPathComponent("root.pid")
        let childPIDFile = temporary.appendingPathComponent("child.pid")
        let latePIDFile = temporary.appendingPathComponent("late.pid")
        let nestedPIDFile = temporary.appendingPathComponent("nested.pid")
        let childScript = temporary.appendingPathComponent("child.sh")
        let lateScript = temporary.appendingPathComponent("late.sh")
        let nestedScript = temporary.appendingPathComponent("nested.sh")

        try writeExecutableScript(
            """
            #!/bin/sh
            trap '' TERM
            echo $$ > \(shellQuote(nestedPIDFile.path))
            while :; do :; done
            """,
            to: nestedScript
        )
        try writeExecutableScript(
            """
            #!/bin/sh
            trap '' TERM
            echo $$ > \(shellQuote(latePIDFile.path))
            \(shellQuote(nestedScript.path)) &
            while :; do :; done
            """,
            to: lateScript
        )
        try writeExecutableScript(
            """
            #!/bin/sh
            trap '' TERM
            echo $$ > \(shellQuote(childPIDFile.path))
            while [ ! -e \(shellQuote(trigger.path)) ]; do sleep 0.01; done
            sleep 0.05
            \(shellQuote(lateScript.path)) &
            while :; do :; done
            """,
            to: childScript
        )
        let rootScript = """
        trap "touch \(shellQuote(trigger.path)); exit 0" TERM
        echo $$ > \(shellQuote(rootPIDFile.path))
        \(shellQuote(childScript.path)) &
        while :; do :; done
        """
        return LateForkingFixture(
            root: temporary,
            paths: paths,
            executable: BackendExecutable(
                url: URL(fileURLWithPath: "/bin/sh"),
                leadingArguments: ["-c", rootScript]
            ),
            rootPIDFile: rootPIDFile,
            childPIDFile: childPIDFile,
            latePIDFile: latePIDFile,
            nestedPIDFile: nestedPIDFile
        )
    }

    private func writeExecutableScript(_ contents: String, to url: URL) throws {
        try contents.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
    }

    private func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\"'\"'"))'"
    }

    private func isProcessAlive(_ pid: pid_t) -> Bool {
        Darwin.kill(pid, 0) == 0 || errno == EPERM
    }
}

private struct LateForkingFixture {
    let root: URL
    let paths: ApplicationPaths
    let executable: BackendExecutable
    let rootPIDFile: URL
    let childPIDFile: URL
    let latePIDFile: URL
    let nestedPIDFile: URL

    func waitForInitialProcessIdentifiers(
        timeout: TimeInterval = 2
    ) throws -> (root: pid_t, child: pid_t) {
        let values = try waitForPIDs([rootPIDFile, childPIDFile], timeout: timeout)
        return (values[0], values[1])
    }

    func waitForLateProcessIdentifiers(
        timeout: TimeInterval = 0.4
    ) throws -> (lateChild: pid_t, nestedChild: pid_t) {
        let values = try waitForPIDs([latePIDFile, nestedPIDFile], timeout: timeout)
        return (values[0], values[1])
    }

    func allRecordedPIDs() -> [pid_t] {
        [rootPIDFile, childPIDFile, latePIDFile, nestedPIDFile].compactMap(readPID)
    }

    func waitUntilProcessExits(_ pid: pid_t, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if Darwin.kill(pid, 0) != 0, errno != EPERM {
                return true
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        return false
    }

    func cleanup() {
        var identities: Set<BackendProcessIdentity> = []
        for pid in allRecordedPIDs() {
            if let identity = BackendProcessTree.identity(for: pid) {
                identities.formUnion(BackendProcessTree.identities(rootedAt: identity.pid))
            }
        }
        BackendProcessTree.forceTerminate(identities)
        try? FileManager.default.removeItem(at: root)
    }

    private func waitForPIDs(_ files: [URL], timeout: TimeInterval) throws -> [pid_t] {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let values = files.compactMap(readPID)
            if values.count == files.count {
                return values
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        throw FixtureError.processIdentifiersUnavailable
    }

    private func readPID(_ url: URL) -> pid_t? {
        guard let value = try? String(contentsOf: url, encoding: .utf8),
              let pid = pid_t(value.trimmingCharacters(in: .whitespacesAndNewlines)),
              pid > 0 else {
            return nil
        }
        return pid
    }

    private enum FixtureError: Error {
        case processIdentifiersUnavailable
    }
}

private struct TermResistantFixture {
    let root: URL
    let paths: ApplicationPaths
    let executable: BackendExecutable
    let parentPIDFile: URL
    let childPIDFile: URL

    func waitForProcessIdentifiers(timeout: TimeInterval = 2) throws -> (parent: pid_t, child: pid_t) {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let parent = readPID(parentPIDFile), let child = readPID(childPIDFile) {
                return (parent, child)
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        throw FixtureError.processIdentifiersUnavailable
    }

    func cleanup() {
        for url in [parentPIDFile, childPIDFile] {
            if let pid = readPID(url), Darwin.kill(pid, 0) == 0 || errno == EPERM {
                _ = Darwin.kill(pid, SIGKILL)
            }
        }
        try? FileManager.default.removeItem(at: root)
    }

    private func readPID(_ url: URL) -> pid_t? {
        guard let value = try? String(contentsOf: url, encoding: .utf8),
              let pid = pid_t(value.trimmingCharacters(in: .whitespacesAndNewlines)),
              pid > 0 else {
            return nil
        }
        return pid
    }

    private enum FixtureError: Error {
        case processIdentifiersUnavailable
    }
}
