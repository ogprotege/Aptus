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
        let sourcePath = repository.appendingPathComponent("src").path
        let environment = ProcessInfo.processInfo.environment
        let configuredPython: URL?
        if let rawConfiguredPython = environment["APTUS_DESKTOP_TEST_PYTHON"] {
            let candidate = URL(fileURLWithPath: rawConfiguredPython)
            configuredPython = FileManager.default.isExecutableFile(atPath: candidate.path)
                ? candidate
                : nil
        } else {
            configuredPython = nil
        }
        let executable: BackendExecutable
        if let configuredPython {
            executable = BackendExecutable(
                url: configuredPython,
                leadingArguments: ["-I", "-c", bootstrap, sourcePath]
            )
        } else {
            let pathCandidates = environment["PATH"]?
                .split(separator: ":")
                .map { URL(fileURLWithPath: String($0)).appendingPathComponent("uv") } ?? []
            let knownCandidates = [
                FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(".local/bin/uv"),
                URL(fileURLWithPath: "/opt/homebrew/bin/uv"),
                URL(fileURLWithPath: "/usr/local/bin/uv"),
            ]
            guard let uv = (pathCandidates + knownCandidates).first(where: {
                FileManager.default.isExecutableFile(atPath: $0.path)
            }) else {
                throw XCTSkip("uv is unavailable for the isolated development backend.")
            }
            executable = BackendExecutable(
                url: uv,
                leadingArguments: [
                    "run", "--isolated", "--python", "3.12", "--locked", "--extra", "server",
                    "--project", repository.path,
                    "python", "-I", "-c", bootstrap, sourcePath,
                ]
            )
        }
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
            case .stopped:
                XCTAssertFalse(FileManager.default.fileExists(atPath: paths.sessionDirectory.path))
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

    func testUnexpectedRootExitRetainsDescendantOwnershipAcrossOneThousandPolls() throws {
        let fixture = try makeSleepingFixture(prefix: "aptus-native-unexpected-tree")
        defer { fixture.cleanup() }
        let shutdown = InjectedShutdownHarness()
        shutdown.survivorDiscoverable = false
        let controller = BackendController(
            pathsFactory: { fixture.paths },
            executableResolver: { fixture.executable },
            shutdownGraceInterval: 1,
            shutdownForceInterval: 1,
            shutdownEnvironment: shutdown.environment
        )
        let failed = expectation(description: "Unexpected root exit preserved the failed state")
        controller.onStateChange = { state in
            if case .failed = state {
                failed.fulfill()
            }
        }

        controller.start()
        let rootPID = try XCTUnwrap(shutdown.capturedRootPID)

        // The representative child appears after launch. One ownership poll must
        // retain its immutable identity before the root and its live tree vanish.
        shutdown.survivorDiscoverable = true
        shutdown.fire()
        shutdown.rootTreeDiscoverable = false
        XCTAssertEqual(Darwin.kill(rootPID, SIGKILL), 0)
        wait(for: [failed], timeout: 1)

        for _ in 0 ..< 1_000 {
            shutdown.fire()
        }
        guard case .failed = controller.state else {
            return XCTFail("Unexpected-exit cleanup must preserve the root failure state.")
        }
        XCTAssertTrue(shutdown.survivorVisible)
        XCTAssertTrue(shutdown.forceTerminatedIdentities.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.paths.sessionDirectory.path))

        shutdown.forceTerminationRemovesSurvivor = true
        shutdown.advance(seconds: 1.1)
        shutdown.fire()
        shutdown.fire()

        XCTAssertFalse(shutdown.survivorVisible)
        XCTAssertTrue(shutdown.forceTerminatedIdentities.contains(shutdown.survivorObservation.identity))
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.paths.sessionDirectory.path))
        guard case .failed = controller.state else {
            return XCTFail("Successful descendant containment must not erase the root failure.")
        }

        let stopped = expectation(description: "Preserved failure can be dismissed after containment")
        controller.stop { result in
            XCTAssertEqual(result, .success)
            stopped.fulfill()
        }
        wait(for: [stopped], timeout: 1)
        XCTAssertEqual(controller.state, .stopped)
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
        controller.stop { result in
            XCTAssertEqual(result, .success)
            XCTAssertFalse(self.isProcessActive(pids.parent))
            XCTAssertFalse(self.isProcessActive(pids.child))
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
        controller.stop { result in
            XCTAssertEqual(result, .success)
            XCTAssertTrue(BackendProcessTree.living(inventory).isEmpty)
            for pid in fixture.allRecordedPIDs() {
                XCTAssertFalse(self.isProcessActive(pid), "Process \(pid) survived shutdown.")
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

    func testInjectedShutdownTimeoutRetainsOwnershipUntilSuccessfulRetry() throws {
        let fixture = try makeSleepingFixture(prefix: "aptus-native-injected-timeout")
        defer { fixture.cleanup() }
        let shutdown = InjectedShutdownHarness()
        let controller = BackendController(
            pathsFactory: { fixture.paths },
            executableResolver: { fixture.executable },
            shutdownGraceInterval: 1,
            shutdownForceInterval: 1,
            shutdownEnvironment: shutdown.environment
        )
        let failed = expectation(description: "Typed shutdown failure returned")
        var failure: BackendShutdownFailure?

        controller.start()
        controller.stop { result in
            guard case let .failure(value) = result else {
                return XCTFail("An injected survivor must return a failed shutdown result.")
            }
            failure = value
            failed.fulfill()
        }
        shutdown.advance(seconds: 1.1)
        shutdown.fire()
        for _ in 0 ..< 1_000 {
            shutdown.fire()
        }
        XCTAssertEqual(controller.state, .stopping)

        shutdown.advance(seconds: 1.0)
        shutdown.fire()
        wait(for: [failed], timeout: 1)

        let observedFailure = try XCTUnwrap(failure)
        XCTAssertEqual(observedFailure.activeProcesses, [shutdown.survivorObservation])
        XCTAssertFalse(observedFailure.signalAttempts.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.paths.sessionDirectory.path))
        guard case .failed = controller.state else {
            return XCTFail("The controller must expose the failed shutdown state.")
        }

        shutdown.survivorVisible = false
        let stopped = expectation(description: "Retained shutdown ownership cleaned up on retry")
        controller.stop { result in
            XCTAssertEqual(result, .success)
            stopped.fulfill()
        }
        shutdown.fire()
        wait(for: [stopped], timeout: 1)
        XCTAssertEqual(controller.state, .stopped)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.paths.sessionDirectory.path))
    }

    func testRestartDoesNotLaunchReplacementAfterInjectedShutdownTimeout() throws {
        let fixture = try makeSleepingFixture(prefix: "aptus-native-injected-restart")
        defer { fixture.cleanup() }
        let shutdown = InjectedShutdownHarness()
        var resolutionCount = 0
        let controller = BackendController(
            pathsFactory: { fixture.paths },
            executableResolver: {
                resolutionCount += 1
                return fixture.executable
            },
            shutdownGraceInterval: 1,
            shutdownForceInterval: 1,
            shutdownEnvironment: shutdown.environment
        )

        controller.start()
        controller.restart()
        shutdown.advance(seconds: 2.1)
        shutdown.fire()
        RunLoop.current.run(until: Date().addingTimeInterval(0.05))

        XCTAssertEqual(resolutionCount, 1)
        guard case .failed = controller.state else {
            return XCTFail("A failed restart shutdown must remain failed.")
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.paths.sessionDirectory.path))

        shutdown.survivorVisible = false
        let stopped = expectation(description: "Failed restart ownership released after explicit retry")
        controller.stop { result in
            XCTAssertEqual(result, .success)
            stopped.fulfill()
        }
        shutdown.fire()
        wait(for: [stopped], timeout: 1)
        XCTAssertEqual(resolutionCount, 1)
    }

    func testProcessStateAndIdentityDistinguishZombieAndPIDReuse() {
        let original = BackendProcessIdentity(
            pid: 42,
            startSeconds: 100,
            startMicroseconds: 200
        )
        let reusedPID = BackendProcessIdentity(
            pid: 42,
            startSeconds: 101,
            startMicroseconds: 1
        )
        let zombie = BackendProcessObservation(
            identity: original,
            parentPID: 1,
            state: BackendProcessState(rawValue: 5)
        )

        XCTAssertEqual(zombie.state, .zombie)
        XCTAssertFalse(zombie.state.requiresContainment)
        XCTAssertNotEqual(original, reusedPID)
        XCTAssertEqual(Set([original, reusedPID]).count, 2)
    }

    private func makeSleepingFixture(prefix: String) throws -> SleepingFixture {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        let paths = try ApplicationPaths(
            applicationSupportRoot: temporary.appendingPathComponent("support"),
            logsRoot: temporary.appendingPathComponent("logs"),
            cachesRoot: temporary.appendingPathComponent("caches"),
            sessionIdentifier: "session"
        )
        return SleepingFixture(
            root: temporary,
            paths: paths,
            executable: BackendExecutable(
                url: URL(fileURLWithPath: "/bin/sleep"),
                leadingArguments: ["30"]
            )
        )
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

    private func isProcessActive(_ pid: pid_t) -> Bool {
        BackendProcessTree.observation(for: pid)?.state.requiresContainment == true
    }
}

private final class InjectedShutdownHarness {
    private let scheduler = ManualShutdownScheduler()
    private var currentNanoseconds: UInt64 = 1_000_000
    private var rootIdentity: BackendProcessIdentity?
    private let survivorIdentity = BackendProcessIdentity(
        pid: 999_999,
        startSeconds: 123,
        startMicroseconds: 456
    )
    var rootTreeDiscoverable = true
    var survivorDiscoverable = true
    var survivorVisible = true
    var forceTerminationRemovesSurvivor = false
    private(set) var forceTerminatedIdentities: Set<BackendProcessIdentity> = []

    var capturedRootPID: pid_t? {
        rootIdentity?.pid
    }

    var survivorObservation: BackendProcessObservation {
        BackendProcessObservation(
            identity: survivorIdentity,
            parentPID: rootIdentity?.pid ?? 1,
            state: .sleeping
        )
    }

    lazy var environment = BackendShutdownEnvironment(
        now: { [unowned self] in
            DispatchTime(uptimeNanoseconds: self.currentNanoseconds)
        },
        identities: { [unowned self] rootPID in
            let root = BackendProcessIdentity(
                pid: rootPID,
                startSeconds: 100,
                startMicroseconds: 200
            )
            self.rootIdentity = root
            guard self.rootTreeDiscoverable else { return [] }
            return self.survivorDiscoverable
                ? [root, self.survivorIdentity]
                : [root]
        },
        expanding: { $0 },
        activeObservations: { [unowned self] identities in
            guard self.survivorVisible, identities.contains(self.survivorIdentity) else {
                return []
            }
            return [self.survivorObservation]
        },
        suspend: { identities in
            Self.signalAttempts(identities, signal: SIGSTOP)
        },
        forceTerminate: { [unowned self] identities in
            self.forceTerminatedIdentities.formUnion(identities)
            let attempts = Self.signalAttempts(identities, signal: SIGKILL)
            if self.forceTerminationRemovesSurvivor,
               identities.contains(self.survivorIdentity) {
                self.survivorVisible = false
            }
            return attempts
        },
        forceTerminateRoot: { pid in
            BackendSignalAttempt(
                pid: pid,
                expectedIdentity: nil,
                signal: SIGKILL,
                disposition: .delivered
            )
        },
        identity: { [unowned self] pid in
            self.rootIdentity?.pid == pid ? self.rootIdentity : nil
        },
        schedulePolling: { [unowned self] handler in
            self.scheduler.schedule(handler)
        }
    )

    func advance(seconds: TimeInterval) {
        currentNanoseconds += UInt64(seconds * 1_000_000_000)
    }

    func fire() {
        scheduler.fire()
    }

    private static func signalAttempts(
        _ identities: Set<BackendProcessIdentity>,
        signal: Int32
    ) -> [BackendSignalAttempt] {
        identities.map { identity in
            BackendSignalAttempt(
                pid: identity.pid,
                expectedIdentity: identity,
                signal: signal,
                disposition: .delivered
            )
        }
    }
}

private final class ManualShutdownScheduler {
    private var polls: [ManualShutdownPoll] = []

    func schedule(_ handler: @escaping () -> Void) -> BackendShutdownPolling {
        let poll = ManualShutdownPoll(handler: handler)
        polls.append(poll)
        return poll
    }

    func fire() {
        for poll in polls where !poll.cancelled {
            poll.handler()
        }
    }
}

private final class ManualShutdownPoll: BackendShutdownPolling {
    let handler: () -> Void
    private(set) var cancelled = false

    init(handler: @escaping () -> Void) {
        self.handler = handler
    }

    func cancel() {
        cancelled = true
    }
}

private struct SleepingFixture {
    let root: URL
    let paths: ApplicationPaths
    let executable: BackendExecutable

    func cleanup() {
        try? FileManager.default.removeItem(at: root)
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
