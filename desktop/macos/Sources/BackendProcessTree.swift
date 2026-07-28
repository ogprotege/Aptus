import Darwin
import Foundation

struct BackendProcessIdentity: Hashable {
    let pid: pid_t
    let startSeconds: UInt64
    let startMicroseconds: UInt64
}

enum BackendProcessState: Equatable {
    case initializing
    case running
    case sleeping
    case stopped
    case zombie
    case unknown(UInt32)

    init(rawValue: UInt32) {
        switch rawValue {
        case 1: self = .initializing
        case 2: self = .running
        case 3: self = .sleeping
        case 4: self = .stopped
        case 5: self = .zombie
        default: self = .unknown(rawValue)
        }
    }

    var requiresContainment: Bool {
        self != .zombie
    }
}

struct BackendProcessObservation: Equatable {
    let identity: BackendProcessIdentity
    let parentPID: pid_t
    let state: BackendProcessState
}

enum BackendSignalDisposition: Equatable {
    case delivered
    case processExited
    case identityChanged
    case zombie
    case permissionDenied
    case failed(Int32)
}

struct BackendSignalAttempt: Equatable {
    let pid: pid_t
    let expectedIdentity: BackendProcessIdentity?
    let signal: Int32
    let disposition: BackendSignalDisposition
}

enum BackendProcessTree {
    static func observation(for pid: pid_t) -> BackendProcessObservation? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.stride)
        guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else {
            return nil
        }
        return BackendProcessObservation(
            identity: BackendProcessIdentity(
                pid: pid,
                startSeconds: info.pbi_start_tvsec,
                startMicroseconds: info.pbi_start_tvusec
            ),
            parentPID: pid_t(info.pbi_ppid),
            state: BackendProcessState(rawValue: info.pbi_status)
        )
    }

    static func identity(for pid: pid_t) -> BackendProcessIdentity? {
        observation(for: pid)?.identity
    }

    static func identities(rootedAt rootPID: pid_t) -> Set<BackendProcessIdentity> {
        guard let root = observation(for: rootPID) else { return [] }

        var result: Set<BackendProcessIdentity> = [root.identity]
        var frontier: [pid_t] = [rootPID]
        while let parent = frontier.popLast() {
            for childPID in childPIDs(of: parent) {
                guard let process = observation(for: childPID), process.parentPID == parent else {
                    continue
                }
                if result.insert(process.identity).inserted {
                    frontier.append(process.identity.pid)
                }
            }
        }
        return result
    }

    static func expanding(_ tracked: Set<BackendProcessIdentity>) -> Set<BackendProcessIdentity> {
        var result = tracked
        for root in tracked where identity(for: root.pid) == root {
            result.formUnion(identities(rootedAt: root.pid))
        }
        return result
    }

    static func activeObservations(
        _ identities: Set<BackendProcessIdentity>
    ) -> [BackendProcessObservation] {
        identities.compactMap { expected in
            guard let current = observation(for: expected.pid),
                  current.identity == expected,
                  current.state.requiresContainment else {
                return nil
            }
            return current
        }.sorted {
            if $0.identity.pid != $1.identity.pid {
                return $0.identity.pid < $1.identity.pid
            }
            if $0.identity.startSeconds != $1.identity.startSeconds {
                return $0.identity.startSeconds < $1.identity.startSeconds
            }
            return $0.identity.startMicroseconds < $1.identity.startMicroseconds
        }
    }

    static func living(_ identities: Set<BackendProcessIdentity>) -> Set<BackendProcessIdentity> {
        Set(activeObservations(identities).map(\.identity))
    }

    @discardableResult
    static func forceTerminate(
        _ identities: Set<BackendProcessIdentity>
    ) -> [BackendSignalAttempt] {
        signal(identities, with: SIGKILL)
    }

    @discardableResult
    static func forceTerminate(pid: pid_t) -> BackendSignalAttempt {
        signal(pid: pid, expectedIdentity: nil, with: SIGKILL)
    }

    @discardableResult
    static func suspend(_ identities: Set<BackendProcessIdentity>) -> [BackendSignalAttempt] {
        signal(identities, with: SIGSTOP)
    }

    private static func childPIDs(of parentPID: pid_t) -> [pid_t] {
        let estimatedCount = max(Int(proc_listchildpids(parentPID, nil, 0)), 16)
        var pids = [pid_t](repeating: 0, count: estimatedCount + 16)
        let count = Int(proc_listchildpids(
            parentPID,
            &pids,
            Int32(pids.count * MemoryLayout<pid_t>.stride)
        ))
        guard count > 0 else { return [] }
        return Array(pids.prefix(min(count, pids.count)).filter { $0 > 0 })
    }

    private static func signal(
        _ identities: Set<BackendProcessIdentity>,
        with signal: Int32
    ) -> [BackendSignalAttempt] {
        identities.sorted {
            if $0.pid != $1.pid { return $0.pid < $1.pid }
            if $0.startSeconds != $1.startSeconds {
                return $0.startSeconds < $1.startSeconds
            }
            return $0.startMicroseconds < $1.startMicroseconds
        }.map { identity in
            self.signal(pid: identity.pid, expectedIdentity: identity, with: signal)
        }
    }

    private static func signal(
        pid: pid_t,
        expectedIdentity: BackendProcessIdentity?,
        with signal: Int32
    ) -> BackendSignalAttempt {
        if let expectedIdentity {
            guard let current = observation(for: pid) else {
                return BackendSignalAttempt(
                    pid: pid,
                    expectedIdentity: expectedIdentity,
                    signal: signal,
                    disposition: .processExited
                )
            }
            guard current.identity == expectedIdentity else {
                return BackendSignalAttempt(
                    pid: pid,
                    expectedIdentity: expectedIdentity,
                    signal: signal,
                    disposition: .identityChanged
                )
            }
            guard current.state.requiresContainment else {
                return BackendSignalAttempt(
                    pid: pid,
                    expectedIdentity: expectedIdentity,
                    signal: signal,
                    disposition: .zombie
                )
            }
        }

        if Darwin.kill(pid, signal) == 0 {
            return BackendSignalAttempt(
                pid: pid,
                expectedIdentity: expectedIdentity,
                signal: signal,
                disposition: .delivered
            )
        }
        let failure = errno
        let disposition: BackendSignalDisposition
        switch failure {
        case ESRCH:
            disposition = .processExited
        case EPERM:
            disposition = .permissionDenied
        default:
            disposition = .failed(failure)
        }
        return BackendSignalAttempt(
            pid: pid,
            expectedIdentity: expectedIdentity,
            signal: signal,
            disposition: disposition
        )
    }
}
