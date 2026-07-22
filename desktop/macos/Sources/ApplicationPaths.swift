import Foundation

enum ApplicationPathsError: LocalizedError {
    case unexpectedItem(URL, expected: String)
    case insecurePermissions(URL, expected: Int, actual: Int)
    case createFileFailed(URL)

    var errorDescription: String? {
        switch self {
        case let .unexpectedItem(url, expected):
            return "Aptus expected \(expected) at \(url.path)."
        case let .insecurePermissions(url, expected, actual):
            return String(
                format: "Aptus could not secure %@ (expected %03o, found %03o).",
                url.path,
                expected,
                actual
            )
        case let .createFileFailed(url):
            return "Aptus could not create its private log at \(url.path)."
        }
    }
}

struct ApplicationPaths: Equatable {
    static let maximumRetainedLogBytes: UInt64 = 2 * 1_024 * 1_024
    static let retainedLogFileCount = 2

    let applicationSupport: URL
    let stateDirectory: URL
    let logsDirectory: URL
    let logFile: URL
    let sessionDirectory: URL
    let readyFile: URL

    var rotatedLogFiles: [URL] {
        (1 ... Self.retainedLogFileCount).map {
            logsDirectory.appendingPathComponent("backend.log.\($0)", isDirectory: false)
        }
    }

    init(
        applicationSupportRoot: URL? = nil,
        logsRoot: URL? = nil,
        cachesRoot: URL? = nil,
        sessionIdentifier: String = UUID().uuidString.lowercased()
    ) throws {
        let manager = FileManager.default
        let supportBase = try applicationSupportRoot ?? manager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let logsBase = try logsRoot ?? manager.url(
            for: .libraryDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("Logs", isDirectory: true)
        let cachesBase = try cachesRoot ?? manager.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )

        applicationSupport = supportBase.appendingPathComponent("Aptus", isDirectory: true)
        stateDirectory = applicationSupport.appendingPathComponent("state", isDirectory: true)
        logsDirectory = logsBase.appendingPathComponent("Aptus", isDirectory: true)
        logFile = logsDirectory.appendingPathComponent("backend.log", isDirectory: false)
        sessionDirectory = cachesBase
            .appendingPathComponent("Aptus", isDirectory: true)
            .appendingPathComponent("sessions", isDirectory: true)
            .appendingPathComponent(sessionIdentifier, isDirectory: true)
        readyFile = sessionDirectory.appendingPathComponent("ready.json", isDirectory: false)
    }

    func prepare(maximumRetainedLogBytes: UInt64 = Self.maximumRetainedLogBytes) throws {
        let manager = FileManager.default
        let sessionsDirectory = sessionDirectory.deletingLastPathComponent()
        let cacheApplicationDirectory = sessionsDirectory.deletingLastPathComponent()
        for directory in [
            applicationSupport,
            stateDirectory,
            logsDirectory,
            cacheApplicationDirectory,
            sessionsDirectory,
            sessionDirectory,
        ] {
            try manager.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try secure(directory, as: .typeDirectory, permissions: 0o700, manager: manager)
        }
        try prepareLog(maximumRetainedLogBytes: maximumRetainedLogBytes, manager: manager)
    }

    func removeEphemeralSession() {
        try? FileManager.default.removeItem(at: sessionDirectory)
    }

    private func prepareLog(maximumRetainedLogBytes: UInt64, manager: FileManager) throws {
        try createLogIfNeeded(at: logFile, manager: manager)
        try secure(logFile, as: .typeRegular, permissions: 0o600, manager: manager)

        for archive in rotatedLogFiles where manager.fileExists(atPath: archive.path) {
            try secure(archive, as: .typeRegular, permissions: 0o600, manager: manager)
            try capExistingLog(
                at: archive,
                maximumBytes: maximumRetainedLogBytes,
                manager: manager
            )
        }

        let attributes = try manager.attributesOfItem(atPath: logFile.path)
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        guard maximumRetainedLogBytes > 0, size >= maximumRetainedLogBytes else {
            return
        }

        let reader = try FileHandle(forReadingFrom: logFile)
        defer { try? reader.close() }
        try reader.seek(toOffset: size - maximumRetainedLogBytes)
        let retainedTail = try reader.read(upToCount: Int(maximumRetainedLogBytes)) ?? Data()

        let temporaryArchive = logsDirectory.appendingPathComponent(
            ".backend.log.rotate-\(UUID().uuidString)",
            isDirectory: false
        )
        guard manager.createFile(
            atPath: temporaryArchive.path,
            contents: retainedTail,
            attributes: [.posixPermissions: 0o600]
        ) else {
            throw ApplicationPathsError.createFileFailed(temporaryArchive)
        }
        defer { try? manager.removeItem(at: temporaryArchive) }
        try secure(temporaryArchive, as: .typeRegular, permissions: 0o600, manager: manager)

        if let oldest = rotatedLogFiles.last,
           manager.fileExists(atPath: oldest.path) {
            try manager.removeItem(at: oldest)
        }
        if rotatedLogFiles.count > 1 {
            for index in stride(from: rotatedLogFiles.count - 2, through: 0, by: -1) {
                let source = rotatedLogFiles[index]
                guard manager.fileExists(atPath: source.path) else { continue }
                try manager.moveItem(at: source, to: rotatedLogFiles[index + 1])
            }
        }
        try manager.moveItem(at: temporaryArchive, to: rotatedLogFiles[0])
        try manager.removeItem(at: logFile)
        try createLogIfNeeded(at: logFile, manager: manager)
        try secure(logFile, as: .typeRegular, permissions: 0o600, manager: manager)
    }

    private func capExistingLog(
        at url: URL,
        maximumBytes: UInt64,
        manager: FileManager
    ) throws {
        let attributes = try manager.attributesOfItem(atPath: url.path)
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        guard maximumBytes > 0, size > maximumBytes else { return }

        let reader = try FileHandle(forReadingFrom: url)
        try reader.seek(toOffset: size - maximumBytes)
        let retainedTail = try reader.read(upToCount: Int(maximumBytes)) ?? Data()
        try reader.close()

        let writer = try FileHandle(forWritingTo: url)
        try writer.truncate(atOffset: 0)
        try writer.write(contentsOf: retainedTail)
        try writer.synchronize()
        try writer.close()
        try secure(url, as: .typeRegular, permissions: 0o600, manager: manager)
    }

    private func createLogIfNeeded(at url: URL, manager: FileManager) throws {
        guard !manager.fileExists(atPath: url.path) else { return }
        guard manager.createFile(
            atPath: url.path,
            contents: nil,
            attributes: [.posixPermissions: 0o600]
        ) else {
            throw ApplicationPathsError.createFileFailed(url)
        }
    }

    private func secure(
        _ url: URL,
        as expectedType: FileAttributeType,
        permissions expectedPermissions: Int,
        manager: FileManager
    ) throws {
        try manager.setAttributes(
            [.posixPermissions: expectedPermissions],
            ofItemAtPath: url.path
        )
        let attributes = try manager.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == expectedType else {
            let description = expectedType == .typeDirectory ? "a private directory" : "a private regular file"
            throw ApplicationPathsError.unexpectedItem(url, expected: description)
        }
        let actualPermissions = (attributes[.posixPermissions] as? NSNumber)?.intValue ?? -1
        guard actualPermissions == expectedPermissions,
              actualPermissions & 0o077 == 0 else {
            throw ApplicationPathsError.insecurePermissions(
                url,
                expected: expectedPermissions,
                actual: actualPermissions
            )
        }
    }
}
