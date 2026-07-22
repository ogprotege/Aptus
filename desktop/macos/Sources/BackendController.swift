import Darwin
import Foundation

final class BackendController {
    typealias StateObserver = (BackendState) -> Void

    private struct ShutdownContext {
        var targets: Set<BackendProcessIdentity>
        let rootPID: pid_t
        let gracefulDeadline: DispatchTime
        let forcedDeadline: DispatchTime
        let preserveFailure: Bool
        var forced = false
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
    private var process: Process?
    private var paths: ApplicationPaths?
    private var token: String?
    private var startupTimer: DispatchSourceTimer?
    private var startupDeadline: Date?
    private var healthCheckInFlight = false
    private var logHandle: FileHandle?
    private var intentionalStop = false
    private var stopCompletions: [() -> Void] = []
    private var shutdownContext: ShutdownContext?
    private var shutdownTimer: DispatchSourceTimer?

    init(
        pathsFactory: @escaping () throws -> ApplicationPaths = { try ApplicationPaths() },
        executableResolver: @escaping () -> BackendExecutable? = { BackendExecutableResolver.resolve() },
        session: URLSession = .shared,
        expectedVersion: String = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "",
        shutdownGraceInterval: TimeInterval = 1.5,
        shutdownForceInterval: TimeInterval = 1.0
    ) {
        precondition(shutdownGraceInterval > 0)
        precondition(shutdownForceInterval > 0)
        self.pathsFactory = pathsFactory
        self.executableResolver = executableResolver
        self.session = session
        self.expectedVersion = expectedVersion
        self.shutdownGraceInterval = shutdownGraceInterval
        self.shutdownForceInterval = shutdownForceInterval
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
        stop {
            self.start()
        }
    }

    func stop(completion: (() -> Void)? = nil) {
        guard Thread.isMainThread else {
            DispatchQueue.main.async { [weak self] in self?.stop(completion: completion) }
            return
        }
        startupTimer?.cancel()
        startupTimer = nil
        intentionalStop = true
        paths?.removeEphemeralSession()
        if let completion {
            stopCompletions.append(completion)
        }
        if shutdownContext != nil {
            return
        }
        guard let process else {
            cleanup()
            state = .stopped
            completeStopRequests()
            return
        }
        if process.isRunning {
            beginShutdown(process, preserveFailure: false)
        } else {
            cleanup()
            state = .stopped
            completeStopRequests()
        }
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
            if shutdownContext != nil {
                evaluateShutdown()
                return
            }
            cleanup()
            if case .failed = state {
                // Preserve the startup or validation error which triggered termination.
            } else {
                state = .stopped
            }
            completeStopRequests()
        } else {
            let message = BackendError.serviceExited(terminatedProcess.terminationStatus).errorDescription
                ?? "The local planning service exited unexpectedly."
            cleanup()
            state = .failed(message)
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
        } else {
            cleanup()
        }
    }

    private func beginShutdown(_ process: Process, preserveFailure: Bool) {
        guard shutdownContext == nil else { return }
        intentionalStop = true
        let now = DispatchTime.now()
        let rootPID = process.processIdentifier
        shutdownContext = ShutdownContext(
            targets: BackendProcessTree.identities(rootedAt: rootPID),
            rootPID: rootPID,
            gracefulDeadline: now + dispatchInterval(shutdownGraceInterval),
            forcedDeadline: now + dispatchInterval(
                shutdownGraceInterval + shutdownForceInterval
            ),
            preserveFailure: preserveFailure
        )
        if !preserveFailure {
            state = .stopping
        }

        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now(), repeating: .milliseconds(25), leeway: .milliseconds(5))
        timer.setEventHandler { [weak self] in self?.evaluateShutdown() }
        shutdownTimer = timer
        timer.resume()
        process.terminate()
    }

    private func evaluateShutdown() {
        guard var shutdown = shutdownContext else { return }

        shutdown.targets = BackendProcessTree.expanding(shutdown.targets)
        var living = BackendProcessTree.living(shutdown.targets)
        if living.isEmpty, process?.isRunning != true {
            finishShutdown(preserveFailure: shutdown.preserveFailure)
            return
        }

        let now = DispatchTime.now()
        if !shutdown.forced,
           now.uptimeNanoseconds >= shutdown.gracefulDeadline.uptimeNanoseconds {
            shutdown.forced = true
            // Freeze the captured tree before the final snapshot so no process can
            // fork between discovery and forced termination.
            BackendProcessTree.suspend(living)
            for _ in 0 ..< 8 {
                let discovered = BackendProcessTree.expanding(shutdown.targets)
                    .subtracting(shutdown.targets)
                guard !discovered.isEmpty else { break }
                shutdown.targets.formUnion(discovered)
                BackendProcessTree.suspend(discovered)
            }
            living = BackendProcessTree.living(shutdown.targets)
            BackendProcessTree.forceTerminate(living)
            if let process, process.isRunning,
               BackendProcessTree.identity(for: shutdown.rootPID) == nil {
                _ = Darwin.kill(shutdown.rootPID, SIGKILL)
            }
            living = BackendProcessTree.living(shutdown.targets)
        } else if shutdown.forced, !living.isEmpty {
            BackendProcessTree.forceTerminate(living)
        }

        if now.uptimeNanoseconds >= shutdown.forcedDeadline.uptimeNanoseconds {
            living = BackendProcessTree.living(shutdown.targets)
            if living.isEmpty, process?.isRunning != true {
                finishShutdown(preserveFailure: shutdown.preserveFailure)
            } else {
                let pids = living.map(\.pid).sorted()
                finishShutdown(
                    preserveFailure: shutdown.preserveFailure,
                    failure: BackendError.shutdownTimedOut(pids)
                )
            }
            return
        }
        shutdownContext = shutdown
    }

    private func dispatchInterval(_ interval: TimeInterval) -> DispatchTimeInterval {
        .nanoseconds(Int(interval * 1_000_000_000))
    }

    private func finishShutdown(preserveFailure: Bool, failure: Error? = nil) {
        shutdownTimer?.cancel()
        shutdownTimer = nil
        shutdownContext = nil
        cleanup()
        if !preserveFailure {
            if let failure {
                state = .failed(
                    (failure as? LocalizedError)?.errorDescription ?? failure.localizedDescription
                )
            } else {
                state = .stopped
            }
        }
        completeStopRequests()
    }

    private func cleanup() {
        shutdownTimer?.cancel()
        shutdownTimer = nil
        shutdownContext = nil
        try? logHandle?.close()
        logHandle = nil
        paths?.removeEphemeralSession()
        paths = nil
        token = nil
        process = nil
        healthCheckInFlight = false
        startupDeadline = nil
    }

    private func completeStopRequests() {
        let completions = stopCompletions
        stopCompletions.removeAll()
        guard !completions.isEmpty else { return }
        DispatchQueue.main.async {
            completions.forEach { $0() }
        }
    }
}
