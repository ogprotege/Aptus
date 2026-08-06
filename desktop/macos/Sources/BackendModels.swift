import Foundation
import Security

struct BackendReadiness: Decodable, Equatable {
    let host: String
    let port: Int
    let version: String

    func validatedOrigin(expectedVersion: String) throws -> URL {
        guard version == expectedVersion else {
            throw BackendError.incompatibleVersion(expected: expectedVersion, actual: version)
        }
        guard host == "127.0.0.1" else {
            throw BackendError.invalidReadiness("The desktop service reported a non-loopback host.")
        }
        guard (1 ... 65_535).contains(port) else {
            throw BackendError.invalidReadiness("The desktop service reported an invalid port.")
        }
        guard let origin = URL(string: "http://127.0.0.1:\(port)") else {
            throw BackendError.invalidReadiness("The desktop service reported an invalid origin.")
        }
        return origin
    }
}

struct BackendHealthResponse: Decodable, Equatable {
    static let expectedStatus = "ok"
    static let expectedAPIContractVersion = "aptus.api.v1"

    let status: String
    let version: String
    let apiContractVersion: String

    private enum CodingKeys: String, CodingKey {
        case status
        case version
        case apiContractVersion = "api_contract_version"
    }

    func validate(expectedVersion: String) throws {
        guard status == Self.expectedStatus else {
            throw BackendError.invalidReadiness(
                "The desktop service returned an invalid health status."
            )
        }
        guard apiContractVersion == Self.expectedAPIContractVersion else {
            throw BackendError.invalidReadiness(
                "The desktop service returned an incompatible API contract."
            )
        }
        guard version == expectedVersion else {
            throw BackendError.incompatibleVersion(
                expected: expectedVersion,
                actual: version
            )
        }
    }
}

struct BackendSession: Equatable {
    let origin: URL
    let token: String
    let version: String
    let logFile: URL
}

enum BackendState: Equatable {
    case stopped
    case starting
    case ready(BackendSession)
    case failed(String)
    case stopping
}

enum BackendError: LocalizedError, Equatable {
    case executableMissing
    case invalidReadiness(String)
    case startupTimedOut
    case serviceExited(Int32)
    case serviceUnavailable
    case launchFailed(String)
    case incompatibleVersion(expected: String, actual: String)
    case shutdownTimedOut([pid_t])

    var errorDescription: String? {
        switch self {
        case .executableMissing:
            return "Aptus could not find its signed local planning service. Rebuild or reinstall Aptus."
        case let .invalidReadiness(message):
            return message
        case .startupTimedOut:
            return "The local planning service did not become ready within 20 seconds."
        case let .serviceExited(status):
            return "The local planning service exited unexpectedly (status \(status))."
        case .serviceUnavailable:
            return "The local planning service did not answer its authenticated health check."
        case let .launchFailed(message):
            return "The local planning service could not start: \(message)"
        case let .incompatibleVersion(expected, actual):
            return "Aptus for Mac \(expected) rejected local service version \(actual). Rebuild the desktop package."
        case let .shutdownTimedOut(pids):
            let identifiers = pids.map(String.init).joined(separator: ", ")
            return "The local planning service did not stop after forced termination (processes: \(identifiers))."
        }
    }
}

enum SessionToken {
    static func generate(byteCount: Int = 32) throws -> String {
        precondition(byteCount >= 32)
        var bytes = [UInt8](repeating: 0, count: byteCount)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        guard status == errSecSuccess else {
            throw BackendError.launchFailed("secure session-token generation failed with status \(status)")
        }
        return Data(bytes).base64EncodedString()
    }
}

struct BackendExecutable: Equatable {
    let url: URL
    let leadingArguments: [String]
}

enum BackendExecutableResolver {
    static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundle: Bundle = .main,
        fileManager: FileManager = .default
    ) -> BackendExecutable? {
        #if DEBUG
        if let explicit = environment["APTUS_DESKTOP_BACKEND"], !explicit.isEmpty {
            let url = URL(fileURLWithPath: explicit).standardizedFileURL
            if fileManager.isExecutableFile(atPath: url.path) {
                return BackendExecutable(url: url, leadingArguments: [])
            }
        }
        #endif

        if let packaged = bundle.url(
            forResource: "aptus-desktop",
            withExtension: nil,
            subdirectory: "backend"
        ), fileManager.isExecutableFile(atPath: packaged.path) {
            return BackendExecutable(url: packaged, leadingArguments: [])
        }

        #if DEBUG
        if let pythonPath = environment["APTUS_DESKTOP_PYTHON"], !pythonPath.isEmpty {
            let python = URL(fileURLWithPath: pythonPath).standardizedFileURL
            if fileManager.isExecutableFile(atPath: python.path) {
                return BackendExecutable(url: python, leadingArguments: ["-I", "-m", "aptus.desktop"])
            }
        }
        #endif

        #if DEBUG
        if let repositoryRoot = bundle.object(forInfoDictionaryKey: "AptusDevelopmentRepositoryRoot") as? String,
           !repositoryRoot.isEmpty {
            let repository = URL(fileURLWithPath: repositoryRoot).standardizedFileURL
            let python = repository
                .appendingPathComponent(".venv/bin/python", isDirectory: false)
                .standardizedFileURL
            if fileManager.isExecutableFile(atPath: python.path) {
                let bootstrap = "import runpy,sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('aptus.desktop', run_name='__main__')"
                return BackendExecutable(
                    url: python,
                    leadingArguments: [
                        "-I",
                        "-c",
                        bootstrap,
                        repository.appendingPathComponent("src", isDirectory: true).path,
                    ]
                )
            }
        }
        #endif
        return nil
    }
}
