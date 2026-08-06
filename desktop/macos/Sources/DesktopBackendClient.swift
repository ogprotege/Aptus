import CoreFoundation
import Foundation

struct RuntimeConfigurationResult: Equatable {
    let runtimeID: String
    let interpreterPath: String
}

enum PersistedMLXRuntimeSelection: Equatable {
    case notConfigured
    case configured(path: String)
    case unavailable(path: String, reason: String)
    case invalid(path: String?, reason: String)
}

struct MLXInterpreterCandidate: Equatable, Identifiable {
    let path: String
    let source: String
    let pythonVersion: String?
    let probePassed: Bool
    let compatible: Bool
    let reason: String
    let packageVersions: [String: String]

    var id: String { path }
}

struct MLXRuntimeInventory: Equatable {
    let selection: PersistedMLXRuntimeSelection
    let candidates: [MLXInterpreterCandidate]
}

protocol RuntimeConfiguring: AnyObject {
    func configureRuntime(
        runtimeID: String,
        interpreterPath: String,
        completion: @escaping (Result<RuntimeConfigurationResult, Error>) -> Void
    )
}

protocol RuntimeInventoryLoading: AnyObject {
    func loadMLXRuntimeInventory(
        completion: @escaping (Result<MLXRuntimeInventory, Error>) -> Void
    )
}

protocol PlatformSnapshotLoading: AnyObject {
    func loadPlatformMemorySnapshot(
        completion: @escaping (Result<PlatformMemorySnapshot, Error>) -> Void
    )
}

enum DesktopBackendClientError: LocalizedError {
    case invalidEndpoint
    case invalidSessionCookie
    case invalidPayload
    case transport(String)
    case rejected(statusCode: Int, message: String)
    case invalidResponse
    case invalidPlatformResponse
    case platformUnsupported(String)
    case invalidRuntimeInventory
    case responseTooLarge

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            "Aptus rejected an invalid local service endpoint."
        case .invalidSessionCookie:
            "Aptus could not authenticate the private local request."
        case .invalidPayload:
            "Aptus could not encode the runtime configuration."
        case let .transport(message):
            "The private local service could not be reached: \(message)"
        case let .rejected(_, message):
            message
        case .invalidResponse:
            "The private local service returned an invalid runtime configuration."
        case .invalidPlatformResponse:
            "The private local service returned an invalid platform snapshot."
        case let .platformUnsupported(message):
            message
        case .invalidRuntimeInventory:
            "The private local service returned an invalid runtime inventory."
        case .responseTooLarge:
            "The private local service returned an oversized response."
        }
    }
}

enum DesktopBackendEndpointPolicy {
    static let healthPath = "/api/v1/health"
    static let runtimeConfigurationPath = "/api/v1/runtimes/configure"
    static let runtimeInventoryPath = "/api/v1/runtimes"
    static let platformPath = "/api/v1/platform"
    private static let allowedPaths = Set([
        healthPath,
        runtimeConfigurationPath,
        runtimeInventoryPath,
        platformPath,
    ])

    static func url(for path: String, origin: URL) -> URL? {
        guard allowedPaths.contains(path),
              origin.scheme == "http",
              origin.host == "127.0.0.1",
              origin.port != nil,
              origin.user == nil,
              origin.password == nil,
              var components = URLComponents(
                  url: origin,
                  resolvingAgainstBaseURL: false
              ) else {
            return nil
        }
        components.path = path
        components.query = nil
        components.fragment = nil
        guard let url = components.url,
              DesktopNavigationPolicy.isSameOrigin(url, as: origin) else {
            return nil
        }
        return url
    }
}

enum DesktopBackendRequestFactory {
    static func runtimeConfiguration(
        session: BackendSession,
        runtimeID: String,
        interpreterPath: String
    ) throws -> URLRequest {
        guard let url = DesktopBackendEndpointPolicy.url(
            for: DesktopBackendEndpointPolicy.runtimeConfigurationPath,
            origin: session.origin
        ) else {
            throw DesktopBackendClientError.invalidEndpoint
        }
        guard let cookie = DesktopSessionCookie.make(session: session) else {
            throw DesktopBackendClientError.invalidSessionCookie
        }
        guard let body = try? JSONSerialization.data(withJSONObject: [
            "runtime_id": runtimeID,
            "interpreter_path": interpreterPath,
        ]) else {
            throw DesktopBackendClientError.invalidPayload
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 20
        request.httpBody = body
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        for (name, value) in HTTPCookie.requestHeaderFields(with: [cookie]) {
            request.setValue(value, forHTTPHeaderField: name)
        }
        return request
    }

    static func platformSnapshot(session: BackendSession) throws -> URLRequest {
        guard let url = DesktopBackendEndpointPolicy.url(
            for: DesktopBackendEndpointPolicy.platformPath,
            origin: session.origin
        ) else {
            throw DesktopBackendClientError.invalidEndpoint
        }
        guard let cookie = DesktopSessionCookie.make(session: session) else {
            throw DesktopBackendClientError.invalidSessionCookie
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 10
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        for (name, value) in HTTPCookie.requestHeaderFields(with: [cookie]) {
            request.setValue(value, forHTTPHeaderField: name)
        }
        return request
    }

    static func runtimeInventory(session: BackendSession) throws -> URLRequest {
        guard let url = DesktopBackendEndpointPolicy.url(
            for: DesktopBackendEndpointPolicy.runtimeInventoryPath,
            origin: session.origin
        ) else {
            throw DesktopBackendClientError.invalidEndpoint
        }
        guard let cookie = DesktopSessionCookie.make(session: session) else {
            throw DesktopBackendClientError.invalidSessionCookie
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        for (name, value) in HTTPCookie.requestHeaderFields(with: [cookie]) {
            request.setValue(value, forHTTPHeaderField: name)
        }
        return request
    }
}

private final class DesktopBackendRedirectDelegate: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

final class DesktopBackendClient: RuntimeConfiguring,
    RuntimeInventoryLoading,
    PlatformSnapshotLoading {
    private static let maximumResponseBytes = 1_048_576

    private let backendSession: BackendSession
    private let redirectDelegate: DesktopBackendRedirectDelegate
    private let urlSession: URLSession

    init(session: BackendSession) {
        backendSession = session
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        let delegate = DesktopBackendRedirectDelegate()
        redirectDelegate = delegate
        urlSession = URLSession(
            configuration: configuration,
            delegate: delegate,
            delegateQueue: nil
        )
    }

    deinit {
        urlSession.invalidateAndCancel()
    }

    func configureRuntime(
        runtimeID: String,
        interpreterPath: String,
        completion: @escaping (Result<RuntimeConfigurationResult, Error>) -> Void
    ) {
        let request: URLRequest
        do {
            request = try DesktopBackendRequestFactory.runtimeConfiguration(
                session: backendSession,
                runtimeID: runtimeID,
                interpreterPath: interpreterPath
            )
        } catch {
            complete(.failure(error), completion: completion)
            return
        }

        urlSession.dataTask(with: request) { data, response, error in
            if let error {
                self.complete(
                    .failure(DesktopBackendClientError.transport(error.localizedDescription)),
                    completion: completion
                )
                return
            }
            guard let response = response as? HTTPURLResponse,
                  let data else {
                self.complete(
                    .failure(DesktopBackendClientError.invalidResponse),
                    completion: completion
                )
                return
            }
            guard data.count <= Self.maximumResponseBytes else {
                self.complete(
                    .failure(DesktopBackendClientError.responseTooLarge),
                    completion: completion
                )
                return
            }
            let payload = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            guard (200 ..< 300).contains(response.statusCode) else {
                self.complete(
                    .failure(DesktopBackendClientError.rejected(
                        statusCode: response.statusCode,
                        message: Self.errorMessage(from: payload)
                    )),
                    completion: completion
                )
                return
            }
            do {
                let result = try Self.runtimeConfigurationResult(
                    from: payload,
                    expectedRuntimeID: runtimeID
                )
                self.complete(.success(result), completion: completion)
            } catch {
                self.complete(.failure(error), completion: completion)
            }
        }.resume()
    }

    func loadMLXRuntimeInventory(
        completion: @escaping (Result<MLXRuntimeInventory, Error>) -> Void
    ) {
        let request: URLRequest
        do {
            request = try DesktopBackendRequestFactory.runtimeInventory(
                session: backendSession
            )
        } catch {
            complete(.failure(error), completion: completion)
            return
        }

        urlSession.dataTask(with: request) { data, response, error in
            if let error {
                self.complete(
                    .failure(DesktopBackendClientError.transport(error.localizedDescription)),
                    completion: completion
                )
                return
            }
            guard let response = response as? HTTPURLResponse,
                  let data else {
                self.complete(
                    .failure(DesktopBackendClientError.invalidRuntimeInventory),
                    completion: completion
                )
                return
            }
            guard data.count <= Self.maximumResponseBytes else {
                self.complete(
                    .failure(DesktopBackendClientError.responseTooLarge),
                    completion: completion
                )
                return
            }
            let payload = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            guard (200 ..< 300).contains(response.statusCode) else {
                self.complete(
                    .failure(DesktopBackendClientError.rejected(
                        statusCode: response.statusCode,
                        message: Self.errorMessage(from: payload)
                    )),
                    completion: completion
                )
                return
            }
            do {
                let inventory = try Self.mlxRuntimeInventory(from: payload)
                self.complete(.success(inventory), completion: completion)
            } catch {
                self.complete(.failure(error), completion: completion)
            }
        }.resume()
    }

    func loadPlatformMemorySnapshot(
        completion: @escaping (Result<PlatformMemorySnapshot, Error>) -> Void
    ) {
        let request: URLRequest
        do {
            request = try DesktopBackendRequestFactory.platformSnapshot(
                session: backendSession
            )
        } catch {
            complete(.failure(error), completion: completion)
            return
        }

        urlSession.dataTask(with: request) { data, response, error in
            if let error {
                self.complete(
                    .failure(DesktopBackendClientError.transport(error.localizedDescription)),
                    completion: completion
                )
                return
            }
            guard let response = response as? HTTPURLResponse,
                  let data else {
                self.complete(
                    .failure(DesktopBackendClientError.invalidPlatformResponse),
                    completion: completion
                )
                return
            }
            guard data.count <= Self.maximumResponseBytes else {
                self.complete(
                    .failure(DesktopBackendClientError.responseTooLarge),
                    completion: completion
                )
                return
            }
            let payload = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            guard (200 ..< 300).contains(response.statusCode) else {
                self.complete(
                    .failure(DesktopBackendClientError.rejected(
                        statusCode: response.statusCode,
                        message: Self.errorMessage(from: payload)
                    )),
                    completion: completion
                )
                return
            }
            do {
                let snapshot = try Self.platformMemorySnapshot(from: payload)
                self.complete(.success(snapshot), completion: completion)
            } catch {
                self.complete(.failure(error), completion: completion)
            }
        }.resume()
    }

    private func complete<Value>(
        _ result: Result<Value, Error>,
        completion: @escaping (Result<Value, Error>) -> Void
    ) {
        DispatchQueue.main.async {
            completion(result)
        }
    }

    static func runtimeConfigurationResult(
        from payload: [String: Any]?,
        expectedRuntimeID: String
    ) throws -> RuntimeConfigurationResult {
        guard payload?["status"] as? String == "ok",
              let responseRuntimeID = payload?["runtime_id"] as? String,
              responseRuntimeID == expectedRuntimeID,
              let responsePath = payload?["interpreter_path"] as? String,
              isCanonicalAbsolutePath(responsePath),
              let interpreter = payload?["interpreter"] as? [String: Any],
              interpreter["path"] as? String == responsePath,
              strictBoolean(payload?["persisted"]) == true else {
            throw DesktopBackendClientError.invalidResponse
        }
        return RuntimeConfigurationResult(
            runtimeID: responseRuntimeID,
            interpreterPath: responsePath
        )
    }

    static func platformMemorySnapshot(
        from payload: [String: Any]?
    ) throws -> PlatformMemorySnapshot {
        guard let payload,
              let status = payload["status"] as? String,
              payload.keys.contains("platform") else {
            throw DesktopBackendClientError.invalidPlatformResponse
        }
        let responseError = try optionalPlatformString(payload["error"])
        switch status {
        case "unsupported":
            guard payload["platform"] is NSNull else {
                throw DesktopBackendClientError.invalidPlatformResponse
            }
            throw DesktopBackendClientError.platformUnsupported(
                responseError.flatMap { $0.isEmpty ? nil : $0 }
                    ?? "Apple platform probing is unsupported on this host."
            )
        case "ok":
            break
        default:
            throw DesktopBackendClientError.invalidPlatformResponse
        }
        guard let platform = payload["platform"] as? [String: Any],
              let installedMemory = unsignedInteger(platform["unified_memory_bytes"]),
              installedMemory > 0 else {
            throw DesktopBackendClientError.invalidPlatformResponse
        }

        let availableMemory: UInt64?
        if platform["available_memory_bytes"] == nil
            || platform["available_memory_bytes"] is NSNull {
            availableMemory = nil
        } else {
            guard let parsed = unsignedInteger(platform["available_memory_bytes"]),
                  parsed <= installedMemory else {
                throw DesktopBackendClientError.invalidPlatformResponse
            }
            availableMemory = parsed
        }

        let freePercent: Int?
        if platform["memory_free_percent"] == nil
            || platform["memory_free_percent"] is NSNull {
            freePercent = nil
        } else {
            guard let parsed = unsignedInteger(platform["memory_free_percent"]),
                  parsed <= 100 else {
                throw DesktopBackendClientError.invalidPlatformResponse
            }
            freePercent = Int(parsed)
        }

        let metalGPUCoreCount: Int?
        if platform["metal_gpu_core_count"] == nil
            || platform["metal_gpu_core_count"] is NSNull {
            metalGPUCoreCount = nil
        } else {
            guard let parsed = unsignedInteger(platform["metal_gpu_core_count"]),
                  parsed > 0,
                  parsed <= UInt64(Int.max) else {
                throw DesktopBackendClientError.invalidPlatformResponse
            }
            metalGPUCoreCount = Int(parsed)
        }

        let observedAt = try optionalPlatformString(platform["observed_at"]).flatMap {
            $0.isEmpty ? nil : $0
        }
        return PlatformMemorySnapshot(
            availableMemoryBytes: availableMemory,
            memoryFreePercent: freePercent,
            metalGPUCoreCount: metalGPUCoreCount,
            pilotPeakBytes: nil,
            observedAt: observedAt
        )
    }

    static func persistedMLXRuntimeSelection(
        from payload: [String: Any]?
    ) throws -> PersistedMLXRuntimeSelection {
        guard payload?["schema_version"] as? String == "aptus.runtime-inventory.v1",
              let selected = payload?["selected"] as? [String: Any] else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }
        let compatiblePaths = try mlxPaths(named: "compatible", from: payload)

        guard let rawSelectedPath = selected["mlx-lm"] else {
            return .notConfigured
        }
        guard let selectedPath = rawSelectedPath as? String else {
            return .invalid(
                path: nil,
                reason: "The persisted MLX-LM interpreter path has an invalid type."
            )
        }
        guard !selectedPath.isEmpty else {
            return .invalid(
                path: nil,
                reason: "The persisted MLX-LM interpreter path is empty."
            )
        }
        guard isCanonicalAbsolutePath(selectedPath) else {
            return .invalid(
                path: selectedPath,
                reason: "The persisted MLX-LM interpreter path is not a canonical absolute path."
            )
        }
        if compatiblePaths.contains(selectedPath) {
            return .configured(path: selectedPath)
        }
        return .unavailable(
            path: selectedPath,
            reason: runtimeUnavailabilityReason(
                for: selectedPath,
                payload: payload
            )
        )
    }

    static func mlxRuntimeInventory(
        from payload: [String: Any]?
    ) throws -> MLXRuntimeInventory {
        guard let configuration = payload?["configuration"] as? [String: Any],
              configuration.allSatisfy({
                  !$0.key.isEmpty
                      && ($0.value as? String).map { !$0.isEmpty } == true
              }),
              let selected = payload?["selected"] as? [String: Any],
              selected.allSatisfy({ !$0.key.isEmpty && $0.value is String }) else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }
        let advertisedAvailablePaths = try mlxPaths(named: "available", from: payload)
        let selection = try persistedMLXRuntimeSelection(from: payload)
        guard let rawInterpreters = payload?["interpreters"] as? [Any] else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }

        var candidates: [MLXInterpreterCandidate] = []
        var seenPaths = Set<String>()
        for rawInterpreter in rawInterpreters {
            guard let interpreter = rawInterpreter as? [String: Any],
                  let path = interpreter["path"] as? String,
                  isCanonicalAbsolutePath(path),
                  seenPaths.insert(path).inserted,
                  let source = interpreter["source"] as? String,
                  !source.isEmpty else {
                throw DesktopBackendClientError.invalidRuntimeInventory
            }

            let pythonVersion = try optionalRuntimeString(
                interpreter["python_version"]
            )
            let interpreterError = try optionalRuntimeString(interpreter["error"])
            let runtimes = interpreter["runtimes"] as? [String: Any]
            let mlx = runtimes?["mlx-lm"] as? [String: Any]
            let packageVersions = try runtimePackageVersions(from: mlx?["versions"])

            let probePassed: Bool
            let compatible: Bool
            let reason: String
            if let interpreterError, !interpreterError.isEmpty {
                probePassed = false
                compatible = false
                reason = boundedRuntimeMessage(interpreterError)
            } else if let mlx {
                guard let available = strictBoolean(mlx["available"]) else {
                    throw DesktopBackendClientError.invalidRuntimeInventory
                }
                guard let exactContract = strictBoolean(mlx["compatible"]) else {
                    throw DesktopBackendClientError.invalidRuntimeInventory
                }
                guard !exactContract || available else {
                    throw DesktopBackendClientError.invalidRuntimeInventory
                }
                probePassed = available
                compatible = exactContract
                if available && exactContract {
                    reason = successfulMLXProbeReason(
                        packageVersions: packageVersions
                    )
                } else if available,
                          let compatibilityReason = try optionalRuntimeString(
                              mlx["compatibility_reason"]
                          ),
                          !compatibilityReason.isEmpty {
                    reason = boundedRuntimeMessage(compatibilityReason)
                } else if let unavailableReason = try optionalRuntimeString(mlx["reason"]),
                          !unavailableReason.isEmpty {
                    reason = boundedRuntimeMessage(unavailableReason)
                } else {
                    reason = "The interpreter did not pass the MLX-LM import probe."
                }
            } else {
                probePassed = false
                compatible = false
                reason = "The interpreter did not return an MLX-LM probe result."
            }

            candidates.append(MLXInterpreterCandidate(
                path: path,
                source: boundedRuntimeMessage(source),
                pythonVersion: pythonVersion.map(boundedRuntimeMessage),
                probePassed: probePassed,
                compatible: compatible,
                reason: reason,
                packageVersions: packageVersions
            ))
        }
        let measuredAvailablePaths = Set(
            candidates.lazy.filter(\.probePassed).map(\.path)
        )
        guard advertisedAvailablePaths == measuredAvailablePaths else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }
        let advertisedCompatiblePaths = try mlxPaths(named: "compatible", from: payload)
        let measuredCompatiblePaths = Set(
            candidates.lazy.filter(\.compatible).map(\.path)
        )
        guard advertisedCompatiblePaths == measuredCompatiblePaths else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }
        return MLXRuntimeInventory(selection: selection, candidates: candidates)
    }

    private static func mlxPaths(
        named field: String,
        from payload: [String: Any]?
    ) throws -> Set<String> {
        guard let pathsByRuntime = payload?[field] as? [String: Any],
              pathsByRuntime.allSatisfy({ key, value in
                  !key.isEmpty
                      && (value as? [Any])?.allSatisfy { $0 is String } == true
              }),
              let rawPaths = pathsByRuntime["mlx-lm"] as? [Any] else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }
        var paths = Set<String>()
        for value in rawPaths {
            guard let path = value as? String,
                  isCanonicalAbsolutePath(path),
                  paths.insert(path).inserted else {
                throw DesktopBackendClientError.invalidRuntimeInventory
            }
        }
        return paths
    }

    private static func isCanonicalAbsolutePath(_ path: String) -> Bool {
        guard path.utf8.count <= 4_096,
              path.first == "/",
              !path.contains("\0") else {
            return false
        }
        return URL(fileURLWithPath: path).standardizedFileURL.path == path
    }

    private static func runtimeUnavailabilityReason(
        for path: String,
        payload: [String: Any]?
    ) -> String {
        guard let interpreters = payload?["interpreters"] as? [Any],
              let interpreter = interpreters.compactMap({ $0 as? [String: Any] }).first(where: {
                  $0["path"] as? String == path
              }) else {
            return "The persisted interpreter is not present in the current runtime inventory."
        }
        if let error = interpreter["error"] as? String, !error.isEmpty {
            return boundedRuntimeMessage(error)
        }
        if let runtimes = interpreter["runtimes"] as? [String: Any],
           let mlx = runtimes["mlx-lm"] as? [String: Any] {
            if let reason = mlx["compatibility_reason"] as? String,
               !reason.isEmpty {
                return boundedRuntimeMessage(reason)
            }
            if let reason = mlx["reason"] as? String,
               !reason.isEmpty {
                return boundedRuntimeMessage(reason)
            }
        }
        return "The persisted interpreter did not pass the exact MLX-LM runtime contract."
    }

    private static func boundedRuntimeMessage(_ message: String) -> String {
        String(message.prefix(500))
    }

    private static func optionalRuntimeString(_ value: Any?) throws -> String? {
        if value == nil || value is NSNull {
            return nil
        }
        guard let string = value as? String else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }
        return string
    }

    private static func optionalPlatformString(_ value: Any?) throws -> String? {
        if value == nil || value is NSNull {
            return nil
        }
        guard let string = value as? String else {
            throw DesktopBackendClientError.invalidPlatformResponse
        }
        return string
    }

    private static func strictBoolean(_ value: Any?) -> Bool? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) == CFBooleanGetTypeID() else {
            return nil
        }
        return number.boolValue
    }

    private static func runtimePackageVersions(
        from value: Any?
    ) throws -> [String: String] {
        if value == nil || value is NSNull {
            return [:]
        }
        guard let rawVersions = value as? [String: Any] else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }
        var versions: [String: String] = [:]
        for (package, rawVersion) in rawVersions {
            if rawVersion is NSNull {
                continue
            }
            guard !package.isEmpty,
                  let version = rawVersion as? String,
                  !version.isEmpty else {
                throw DesktopBackendClientError.invalidRuntimeInventory
            }
            versions[package] = boundedRuntimeMessage(version)
        }
        return versions
    }

    private static func successfulMLXProbeReason(
        packageVersions: [String: String]
    ) -> String {
        let mlxVersion = packageVersions["mlx"]
        let mlxLMVersion = packageVersions["mlx-lm"]
        switch (mlxVersion, mlxLMVersion) {
        case let (.some(mlx), .some(mlxLM)):
            return "MLX \(mlx) and MLX-LM \(mlxLM) passed the exact Aptus runtime contract. Selection revalidates the contract before saving."
        default:
            return "MLX-LM passed the exact Aptus runtime contract. Selection revalidates the contract before saving."
        }
    }

    private static func unsignedInteger(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue >= 0,
              number.doubleValue.rounded(.towardZero) == number.doubleValue,
              number.doubleValue <= Double(UInt64.max) else {
            return nil
        }
        return number.uint64Value
    }

    static func errorMessage(from payload: [String: Any]?) -> String {
        if let details = payload?["details"] as? String, !details.isEmpty {
            return details
        }
        if let error = payload?["error"] as? String, !error.isEmpty {
            return "The runtime configuration was rejected: \(error)."
        }
        if let detail = payload?["detail"] as? [String: Any] {
            if let message = detail["message"] as? String, !message.isEmpty {
                return message
            }
            if let details = detail["details"] as? String, !details.isEmpty {
                return details
            }
            if let error = detail["error"] as? String, !error.isEmpty {
                return "The runtime configuration was rejected: \(error)."
            }
        }
        return "The private local service rejected the runtime configuration."
    }
}
