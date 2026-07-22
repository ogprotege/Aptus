import Darwin
import Foundation

struct BackendProcessIdentity: Hashable {
    let pid: pid_t
    let startSeconds: UInt64
    let startMicroseconds: UInt64
}

enum BackendProcessTree {
    private struct Snapshot {
        let identity: BackendProcessIdentity
        let parentPID: pid_t
    }

    static func identity(for pid: pid_t) -> BackendProcessIdentity? {
        snapshot(for: pid)?.identity
    }

    static func identities(rootedAt rootPID: pid_t) -> Set<BackendProcessIdentity> {
        guard let root = snapshot(for: rootPID) else { return [] }

        var result: Set<BackendProcessIdentity> = [root.identity]
        var frontier: [pid_t] = [rootPID]
        while let parent = frontier.popLast() {
            for childPID in childPIDs(of: parent) {
                guard let process = snapshot(for: childPID), process.parentPID == parent else {
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

    static func living(_ identities: Set<BackendProcessIdentity>) -> Set<BackendProcessIdentity> {
        Set(identities.filter { identity(for: $0.pid) == $0 })
    }

    static func forceTerminate(_ identities: Set<BackendProcessIdentity>) {
        signal(identities, with: SIGKILL)
    }

    static func suspend(_ identities: Set<BackendProcessIdentity>) {
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

    private static func snapshot(for pid: pid_t) -> Snapshot? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.stride)
        guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else {
            return nil
        }
        return Snapshot(
            identity: BackendProcessIdentity(
                pid: pid,
                startSeconds: info.pbi_start_tvsec,
                startMicroseconds: info.pbi_start_tvusec
            ),
            parentPID: pid_t(info.pbi_ppid)
        )
    }

    private static func signal(_ identities: Set<BackendProcessIdentity>, with signal: Int32) {
        for identity in identities where self.identity(for: identity.pid) == identity {
            _ = Darwin.kill(identity.pid, signal)
        }
    }
}
