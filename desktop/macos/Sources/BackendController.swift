import Darwin
import Foundation

enum BackendShutdownResult: Equatable {
    case success
    case failure(BackendShutdownFailure)
}

struct BackendShutdownFailure: LocalizedError, Equatable {
    let rootPID: pid_t
    let activeProcesses: [BackendProcessObservation]
    let rootProcessRunning: Bool
    let signalAttempts: [BackendSignalAttempt]
    let terminationHandlerObserved: Bool

    var errorDescription: String? {
        var pids = Set(activeProcesses.map { $0.identity.pid })
        if rootProcessRunning {
            pids.insert(rootPID)
        }
        let identifiers = pids.sorted().map(String.init).joined(separator: ", ")
        return "The local planning service did not stop after forced termination (processes: \(identifiers))."
    }
}

protocol BackendShutdownPolling: AnyObject {
    func cancel()
}

private final class DispatchBackendShutdownPolling: BackendShutdownPolling {
    private let timer: DispatchSourceTimer

    init(handler: @escaping () -> Void) {
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(
            deadline: .now(),
            repeating: .milliseconds(25),
            leeway: .milliseconds(5)
        )
        timer.setEventHandler(handler: handler)
        self.timer = timer
        timer.resume()
    }

    func cancel() {
        timer.cancel()
    }
}

struct BackendShutdownEnvironment {
    let now: () -> DispatchTime
    let identities: (pid_t) -> Set<BackendProcessIdentity>
    let expanding: (Set<BackendProcessIdentity>) -> Set<BackendProcessIdentity>
    let activeObservations: (Set<BackendProcessIdentity>) -> [BackendProcessObservation]
    let suspend: (Set<BackendProcessIdentity>) -> [BackendSignalAttempt]
    let forceTerminate: (Set<BackendProcessIdentity>) -> [BackendSignalAttempt]
    let forceTerminateRoot: (pid_t) -> BackendSignalAttempt
    let identity: (pid_t) -> BackendProcessIdentity?
    let schedulePolling: (@escaping () -> Void) -> BackendShutdownPolling

    static let live = BackendShutdownEnvironment(
        now: DispatchTime.now,
        identities: BackendProcessTree.identities,
        expanding: BackendProcessTree.expanding,
        activeObservations: BackendProcessTree.activeObservations,
        suspend: BackendProcessTree.suspend,
        forceTerminate: BackendProcessTree.forceTerminate,
        forceTerminateRoot: BackendProcessTree.forceTerminate,
        identity: BackendProcessTree.identity,
        schedulePolling: { DispatchBackendShutdownPolling(handler: $0) }
    )
}

final class BackendController {
    typealias StateObserver = (BackendState) -> Void

    private struct ShutdownContext {
        var targets: Set<BackendProcessIdentity>
        let rootPID: pid_t
        var gracefulDeadline: DispatchTime
        var forcedDeadline: DispatchTime
        let preserveFailure: Bool
        var forced = false
        var timeoutReported = false
        var terminationHandlerObserved = false
        var signalAttempts: [BackendSignalAttempt] = []
    }

    private(set) var state: BackendState = .stopped {
        didSet { DispatchQueue.main.async { [state, onStateChange] in onStateChange?(state) } }
    }

    var onStateChange: StateObserver?

    private let pathsFactory: () throws -> ApplicationPaths
    private let executableResolver: () -> BackendExecutable?
    private let session: URLSession
    private let expectedVersion: String
    private let shutdownGraceInterval: TimeInterval
    private let shutdownForceInterval: TimeInterval
    private let shutdownEnvironment: BackendShutdownEnvironment
    private var process: Process?
    private var paths: ApplicationPaths?
    private var token: String?
    private var startupTimer: DispatchSourceTimer?
    private var startupDeadline: Date?
    private var healthCheckInFlight = false
    private var logHandle: FileHandle?
    private var intentionalStop = false
    private var stopCompletions: [(BackendShutdownResult) -> Void] = []
    private var ownedProcessIdentities: Set<BackendProcessIdentity> = []
    private var ownershipPolling: BackendShutdownPolling?
    private var shutdownContext: ShutdownContext?
    private var shutdownPolling: BackendShutdownPolling?

    init(
        pathsFactory: @escaping () throws -> ApplicationPaths = { try ApplicationPaths() },
        executableResolver: @escaping () -> BackendExecutable? = { BackendExecutableResolver.resolve() },
        session: URLSession = .shared,
        expectedVersion: String = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "",
        shutdownGraceInterval: TimeInterval = 1.5,
        shutdownForceInterval: TimeInterval = 1.0,
        shutdownEnvironment: BackendShutdownEnvironment = .live
    ) {
        precondition(shutdownGraceInterval > 0)
        precondition(shutdownForceInterval > 0)
        self.pathsFactory = pathsFactory
        self.executableResolver = executableResolver
        self.session = session
        self.expectedVersion = expectedVersion
        self.shutdownGraceInterval = shutdownGraceInterval
        self.shutdownForceInterval = shutdownForceInterval
        self.shutdownEnvironment = shutdownEnvironment
    }

    func start() {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in self?.start() }
            return
        }
        guard process == nil else { return }
        intentionalStop = false
        state = .starting

        do {
            let paths = try pathsFactory()
            try paths.prepare()
            try? FileManager.default.removeItem(at: paths.readyFile)
            guard let executable = executableResolver() else {
                throw BackendError.executableMissing
            }
            let token = try SessionToken.generate()
            let process = Process()
            process.executableURL = executable.url
            process.arguments = executable.leadingArguments + [
                "--state-dir", paths.stateDirectory.path,
                "--ready-file", paths.readyFile.path,
            ]
            var environment = ProcessInfo.processInfo.environment
            environment["APTUS_DESKTOP_SESSION_TOKEN"] = token
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process.environment = environment
            process.currentDirectoryURL = paths.applicationSupport

            let logHandle = try FileHandle(forWritingTo: paths.logFile)
            try logHandle.seekToEnd()
            process.standardOutput = logHandle
            process.standardError = logHandle
            process.terminationHandler = { [weak self] terminated in
                DispatchQueue.main.async {
                    self?.handleTermination(terminated)
                }
            }

            self.paths = paths
            self.token = token
            self.process = process
            self.logHandle = logHandle
            try process.run()
            beginOwnershipTracking(process)
            beginReadinessPolling()
        } catch {
            fail(error)
        }
    }

    func restart() {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in self?.restart() }
            return
        }
        stop { [weak self] result in
            guard result == .success else { return }
            self?.start()
        }
    }

    func stop(completion: ((BackendShutdownResult) -> Void)? = nil) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in self?.stop(completion: completion) }
            return
        }
        startupTimer?.cancel()
        startupTimer = nil
        intentionalStop = true
        if let completion {
            stopCompletions.append(completion)
        }
        if shutdownContext?.timeoutReported == true {
            retryTimedOutShutdown()
            return
        }
        if shutdownContext != nil {
            return
        }
        guard let process else {
            cleanup()
            state = .stopped
            completeStopRequests(with: .success)
            return
        }
        if process.isRunning {
            beginShutdown(process, preserveFailure: false)
        } else {
            captureOwnedProcesses(rootPID: process.processIdentifier)
            if shutdownEnvironment.activeObservations(ownedProcessIdentities).isEmpty {
                cleanup()
                state = .stopped
                completeStopRequests(with: .success)
            } else {
                beginShutdown(
                    process,
                    preserveFailure: false,
                    rootTerminationObserved: true
                )
            }
        }
    }

    private func beginOwnershipTracking(_ process: Process) {
        ownershipPolling?.cancel()
        ownershipPolling = nil
        ownedProcessIdentities = shutdownEnvironment.identities(process.processIdentifier)
        ownershipPolling = shutdownEnvironment.schedulePolling { [weak self] in
            self?.captureOwnedProcesses()
        }
    }

    private func captureOwnedProcesses(rootPID: pid_t? = nil) {
        guard shutdownContext == nil else { return }
        let observedRootPID = rootPID ?? process?.processIdentifier
        if let observedRootPID {
            ownedProcessIdentities.formUnion(
                shutdownEnvironment.identities(observedRootPID)
            )
        }
        ownedProcessIdentities = shutdownEnvironment.expanding(ownedProcessIdentities)
    }

    private func stopOwnershipTracking() {
        ownershipPolling?.cancel()
        ownershipPolling = nil
    }

    private func beginReadinessPolling() {
        startupDeadline = Date().addingTimeInterval(20)
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now(), repeating: .milliseconds(100), leeway: .milliseconds(25))
        timer.setEventHandler { [weak self] in self?.pollReadiness() }
        startupTimer = timer
        timer.resume()
    }

    private func pollReadiness() {
        guard case .starting = state, let paths, let token else { return }
        if let deadline = startupDeadline, Date() >= deadline {
            fail(BackendError.startupTimedOut)
            return
        }
        guard !healthCheckInFlight,
              let data = try? Data(contentsOf: paths.readyFile),
              let readiness = try? JSONDecoder().decode(BackendReadiness.self, from: data) else {
            return
        }
        do {
            let origin = try readiness.validatedOrigin(expectedVersion: expectedVersion)
            checkHealth(origin: origin, token: token, readiness: readiness)
        } catch {
            fail(error)
        }
    }

    private func checkHealth(origin: URL, token: String, readiness: BackendReadiness) {
        healthCheckInFlight = true
        var request = URLRequest(url: origin.appendingPathComponent("api/v1/health"))
        request.timeoutInterval = 2
        request.setValue("aptus_desktop_session=\(token)", forHTTPHeaderField: "Cookie")
        session.dataTask(with: request) { [weak self] _, response, _ in
            DispatchQueue.main.async {
                self?.handleHealthResponse(
                    response,
                    origin: origin,
                    token: token,
                    readiness: readiness
                )
            }
        }.resume()
    }

    private func handleHealthResponse(
        _ response: URLResponse?,
        origin: URL,
        token: String,
        readiness: BackendReadiness
    ) {
        healthCheckInFlight = false
        guard case .starting = state, self.token == token else { return }
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else { return }
        startupTimer?.cancel()
        startupTimer = nil
        guard let paths else { return }
        state = .ready(BackendSession(
            origin: origin,
            token: token,
            version: readiness.version,
            logFile: paths.logFile
        ))
    }

    private func handleTermination(_ terminatedProcess: Process) {
        guard process === terminatedProcess else { return }
        startupTimer?.cancel()
        startupTimer = nil
        if intentionalStop {
            if var shutdown = shutdownContext {
                shutdown.terminationHandlerObserved = true
                shutdownContext = shutdown
                evaluateShutdown()
                return
            }
            cleanup()
            if case .failed = state {
                // Preserve the startup or validation error which triggered termination.
            } else {
                state = .stopped
            }
            completeStopRequests(with: .success)
        } else {
            let message = BackendError.serviceExited(terminatedProcess.terminationStatus).errorDescription
                ?? "The local planning service exited unexpectedly."
            state = .failed(message)
            captureOwnedProcesses(rootPID: terminatedProcess.processIdentifier)
            beginShutdown(
                terminatedProcess,
                preserveFailure: true,
                rootTerminationObserved: true
            )
        }
    }

    private func fail(_ error: Error) {
        startupTimer?.cancel()
        startupTimer = nil
        let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        state = .failed(message)
        if let process, process.isRunning {
            intentionalStop = true
            beginShutdown(process, preserveFailure: true)
        } else if let process {
            captureOwnedProcesses(rootPID: process.processIdentifier)
            if shutdownEnvironment.activeObservations(ownedProcessIdentities).isEmpty {
                cleanup()
            } else {
                beginShutdown(
                    process,
                    preserveFailure: true,
                    rootTerminationObserved: true
                )
            }
        } else {
            cleanup()
        }
    }

    private func beginShutdown(
        _ process: Process,
        preserveFailure: Bool,
        rootTerminationObserved: Bool = false
    ) {
        guard shutdownContext == nil else { return }
        intentionalStop = true
        let now = shutdownEnvironment.now()
        let rootPID = process.processIdentifier
        captureOwnedProcesses(rootPID: rootPID)
        stopOwnershipTracking()
        shutdownContext = ShutdownContext(
            targets: ownedProcessIdentities,
            rootPID: rootPID,
            gracefulDeadline: now + dispatchInterval(shutdownGraceInterval),
            forcedDeadline: now + dispatchInterval(
                shutdownGraceInterval + shutdownForceInterval
            ),
            preserveFailure: preserveFailure,
            terminationHandlerObserved: rootTerminationObserved
        )
        if !preserveFailure {
            state = .stopping
        }
        startShutdownPolling()
        if process.isRunning {
            process.terminate()
        } else {
            evaluateShutdown()
        }
    }

    private func retryTimedOutShutdown() {
        guard var shutdown = shutdownContext, shutdown.timeoutReported else { return }
        let now = shutdownEnvironment.now()
        shutdown.timeoutReported = false
        shutdown.forced = true
        shutdown.gracefulDeadline = now
        shutdown.forcedDeadline = now + dispatchInterval(shutdownForceInterval)
        let active = shutdownEnvironment.activeObservations(shutdown.targets)
        shutdown.signalAttempts.append(contentsOf: shutdownEnvironment.forceTerminate(
            Set(active.map(\.identity))
        ))
        shutdownContext = shutdown
        if !shutdown.preserveFailure {
            state = .stopping
        }
        startShutdownPolling()
        evaluateShutdown()
    }

    private func startShutdownPolling() {
        guard shutdownPolling == nil else { return }
        shutdownPolling = shutdownEnvironment.schedulePolling { [weak self] in
            self?.evaluateShutdown()
        }
    }

    private func evaluateShutdown() {
        guard var shutdown = shutdownContext else { return }

        shutdown.targets = shutdownEnvironment.expanding(shutdown.targets)
        var active = shutdownEnvironment.activeObservations(shutdown.targets)
        if active.isEmpty, process?.isRunning != true {
            finishShutdown(preserveFailure: shutdown.preserveFailure)
            return
        }
        guard !shutdown.timeoutReported else {
            shutdownContext = shutdown
            return
        }

        let now = shutdownEnvironment.now()
        if !shutdown.forced,
           now.uptimeNanoseconds >= shutdown.gracefulDeadline.uptimeNanoseconds {
            shutdown.forced = true
            // Freeze the captured tree before the final snapshot so no process can
            // fork between discovery and forced termination.
            shutdown.signalAttempts.append(contentsOf: shutdownEnvironment.suspend(
                Set(active.map(\.identity))
            ))
            for _ in 0 ..< 8 {
                let discovered = shutdownEnvironment.expanding(shutdown.targets)
                    .subtracting(shutdown.targets)
                guard !discovered.isEmpty else { break }
                shutdown.targets.formUnion(discovered)
                shutdown.signalAttempts.append(contentsOf: shutdownEnvironment.suspend(discovered))
            }
            active = shutdownEnvironment.activeObservations(shutdown.targets)
            shutdown.signalAttempts.append(contentsOf: shutdownEnvironment.forceTerminate(
                Set(active.map(\.identity))
            ))
            if let process, process.isRunning,
               shutdownEnvironment.identity(shutdown.rootPID) == nil {
                shutdown.signalAttempts.append(
                    shutdownEnvironment.forceTerminateRoot(shutdown.rootPID)
                )
            }
            active = shutdownEnvironment.activeObservations(shutdown.targets)
        } else if shutdown.forced, !active.isEmpty {
            shutdown.signalAttempts.append(contentsOf: shutdownEnvironment.forceTerminate(
                Set(active.map(\.identity))
            ))
        }

        if now.uptimeNanoseconds >= shutdown.forcedDeadline.uptimeNanoseconds {
            active = shutdownEnvironment.activeObservations(shutdown.targets)
            if active.isEmpty, process?.isRunning != true {
                finishShutdown(preserveFailure: shutdown.preserveFailure)
            } else {
                reportShutdownTimeout(
                    shutdown: shutdown,
                    activeProcesses: active,
                    rootProcessRunning: process?.isRunning == true
                )
            }
            return
        }
        shutdownContext = shutdown
    }

    private func dispatchInterval(_ interval: TimeInterval) -> DispatchTimeInterval {
        .nanoseconds(Int(interval * 1_000_000_000))
    }

    private func reportShutdownTimeout(
        shutdown original: ShutdownContext,
        activeProcesses: [BackendProcessObservation],
        rootProcessRunning: Bool
    ) {
        var shutdown = original
        shutdown.timeoutReported = true
        shutdownContext = shutdown
        shutdownPolling?.cancel()
        shutdownPolling = nil
        let failure = BackendShutdownFailure(
            rootPID: shutdown.rootPID,
            activeProcesses: activeProcesses,
            rootProcessRunning: rootProcessRunning,
            signalAttempts: shutdown.signalAttempts,
            terminationHandlerObserved: shutdown.terminationHandlerObserved
        )
        recordShutdownFailure(failure)
        if !shutdown.preserveFailure {
            state = .failed(failure.errorDescription ?? "The local planning service did not stop.")
        }
        completeStopRequests(with: .failure(failure))
    }

    private func recordShutdownFailure(_ failure: BackendShutdownFailure) {
        let processes = failure.activeProcesses.map { observation in
            let identity = observation.identity
            return "pid=\(identity.pid),parent=\(observation.parentPID),start=\(identity.startSeconds).\(identity.startMicroseconds),state=\(observation.state)"
        }.joined(separator: ";")
        let attempts = failure.signalAttempts.map { attempt in
            "pid=\(attempt.pid),signal=\(attempt.signal),result=\(attempt.disposition)"
        }.joined(separator: ";")
        let message = "[aptus-shutdown-timeout] root=\(failure.rootPID) rootRunning=\(failure.rootProcessRunning) terminationHandlerObserved=\(failure.terminationHandlerObserved) processes=[\(processes)] signals=[\(attempts)]\n"
        guard let data = message.data(using: .utf8) else { return }
        try? logHandle?.write(contentsOf: data)
    }

    private func finishShutdown(preserveFailure: Bool) {
        shutdownPolling?.cancel()
        shutdownPolling = nil
        shutdownContext = nil
        cleanup()
        if !preserveFailure {
            state = .stopped
        }
        completeStopRequests(with: .success)
    }

    private func cleanup() {
        stopOwnershipTracking()
        shutdownPolling?.cancel()
        shutdownPolling = nil
        shutdownContext = nil
        ownedProcessIdentities.removeAll()
        try? logHandle?.close()
        logHandle = nil
        paths?.removeEphemeralSession()
        paths = nil
        token = nil
        process = nil
        healthCheckInFlight = false
        startupDeadline = nil
    }

    private func completeStopRequests(with result: BackendShutdownResult) {
        let completions = stopCompletions
        stopCompletions.removeAll()
        guard !completions.isEmpty else { return }
        DispatchQueue.main.async {
            completions.forEach { $0(result) }
        }
    }
}
