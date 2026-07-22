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

protocol RuntimeConfiguring: AnyObject {
    func configureRuntime(
        runtimeID: String,
        interpreterPath: String,
        completion: @escaping (Result<RuntimeConfigurationResult, Error>) -> Void
    )
}

protocol RuntimeInventoryLoading: AnyObject {
    func loadPersistedMLXRuntime(
        completion: @escaping (Result<PersistedMLXRuntimeSelection, Error>) -> Void
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
        case .invalidRuntimeInventory:
            "The private local service returned an invalid runtime inventory."
        case .responseTooLarge:
            "The private local service returned an oversized response."
        }
    }
}

enum DesktopBackendEndpointPolicy {
    static let runtimeConfigurationPath = "/api/v1/runtimes/configure"
    static let runtimeInventoryPath = "/api/v1/runtimes"
    static let platformPath = "/api/v1/platform"
    private static let allowedPaths = Set([
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
            guard payload?["status"] as? String == "ok",
                  let responseRuntimeID = payload?["runtime_id"] as? String,
                  let responsePath = payload?["interpreter_path"] as? String,
                  responseRuntimeID == runtimeID,
                  !responsePath.isEmpty else {
                self.complete(
                    .failure(DesktopBackendClientError.invalidResponse),
                    completion: completion
                )
                return
            }
            self.complete(
                .success(RuntimeConfigurationResult(
                    runtimeID: responseRuntimeID,
                    interpreterPath: responsePath
                )),
                completion: completion
            )
        }.resume()
    }

    func loadPersistedMLXRuntime(
        completion: @escaping (Result<PersistedMLXRuntimeSelection, Error>) -> Void
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
                let selection = try Self.persistedMLXRuntimeSelection(from: payload)
                self.complete(.success(selection), completion: completion)
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

    static func platformMemorySnapshot(
        from payload: [String: Any]?
    ) throws -> PlatformMemorySnapshot {
        guard payload?["status"] as? String == "ok",
              let platform = payload?["platform"] as? [String: Any],
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

        let observedAt = (platform["observed_at"] as? String).flatMap {
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
              let selected = payload?["selected"] as? [String: Any],
              let available = payload?["available"] as? [String: Any],
              let rawAvailablePaths = available["mlx-lm"] as? [Any] else {
            throw DesktopBackendClientError.invalidRuntimeInventory
        }

        var availablePaths: [String] = []
        for value in rawAvailablePaths {
            guard let path = value as? String, !path.isEmpty else {
                throw DesktopBackendClientError.invalidRuntimeInventory
            }
            availablePaths.append(path)
        }

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
        if availablePaths.contains(selectedPath) {
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
           let mlx = runtimes["mlx-lm"] as? [String: Any],
           let reason = mlx["reason"] as? String,
           !reason.isEmpty {
            return boundedRuntimeMessage(reason)
        }
        return "The persisted interpreter did not pass the MLX-LM availability probe."
    }

    private static func boundedRuntimeMessage(_ message: String) -> String {
        String(message.prefix(500))
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
