import Foundation
import XCTest
@testable import Aptus

private final class RuntimeConfiguratorStub: RuntimeConfiguring {
    private(set) var runtimeID: String?
    private(set) var interpreterPath: String?
    var result: Result<RuntimeConfigurationResult, Error>

    init(result: Result<RuntimeConfigurationResult, Error>) {
        self.result = result
    }

    func configureRuntime(
        runtimeID: String,
        interpreterPath: String,
        completion: @escaping (Result<RuntimeConfigurationResult, Error>) -> Void
    ) {
        self.runtimeID = runtimeID
        self.interpreterPath = interpreterPath
        completion(result)
    }
}

private final class PlatformSnapshotLoaderStub: PlatformSnapshotLoading {
    private(set) var loadCount = 0
    private var completion: ((Result<PlatformMemorySnapshot, Error>) -> Void)?

    func loadPlatformMemorySnapshot(
        completion: @escaping (Result<PlatformMemorySnapshot, Error>) -> Void
    ) {
        loadCount += 1
        self.completion = completion
    }

    func resolve(_ result: Result<PlatformMemorySnapshot, Error>) {
        completion?(result)
    }
}

private final class RuntimeInventoryLoaderStub: RuntimeInventoryLoading {
    private(set) var loadCount = 0
    private var completion: ((Result<MLXRuntimeInventory, Error>) -> Void)?

    func loadMLXRuntimeInventory(
        completion: @escaping (Result<MLXRuntimeInventory, Error>) -> Void
    ) {
        loadCount += 1
        self.completion = completion
    }

    func resolve(_ result: Result<MLXRuntimeInventory, Error>) {
        completion?(result)
    }
}

final class DesktopBackendClientTests: XCTestCase {
    func testRuntimeConfigurationEndpointIsExactOriginAndAllowlisted() throws {
        let origin = try XCTUnwrap(URL(string: "http://127.0.0.1:49152"))
        let url = try XCTUnwrap(DesktopBackendEndpointPolicy.url(
            for: "/api/v1/runtimes/configure",
            origin: origin
        ))
        XCTAssertEqual(url.absoluteString, "http://127.0.0.1:49152/api/v1/runtimes/configure")
        XCTAssertEqual(DesktopBackendEndpointPolicy.url(
            for: "/api/v1/platform",
            origin: origin
        )?.absoluteString, "http://127.0.0.1:49152/api/v1/platform")
        XCTAssertEqual(DesktopBackendEndpointPolicy.url(
            for: "/api/v1/runtimes",
            origin: origin
        )?.absoluteString, "http://127.0.0.1:49152/api/v1/runtimes")
        XCTAssertNil(DesktopBackendEndpointPolicy.url(
            for: "/api/v1/hardware",
            origin: origin
        ))
        XCTAssertNil(DesktopBackendEndpointPolicy.url(
            for: "/api/v1/runtimes/configure",
            origin: try XCTUnwrap(URL(string: "http://localhost:49152"))
        ))
        XCTAssertNil(DesktopBackendEndpointPolicy.url(
            for: "/api/v1/runtimes/configure",
            origin: try XCTUnwrap(URL(string: "https://127.0.0.1:49152"))
        ))
    }

    func testRuntimeConfigurationRequestUsesPrivateCookieAndExpectedContract() throws {
        let session = BackendSession(
            origin: try XCTUnwrap(URL(string: "http://127.0.0.1:49152")),
            token: "0123456789012345678901234567890123456789",
            version: "0.2.0",
            logFile: URL(fileURLWithPath: "/tmp/aptus-test.log")
        )
        let request = try DesktopBackendRequestFactory.runtimeConfiguration(
            session: session,
            runtimeID: "mlx-lm",
            interpreterPath: "/opt/homebrew/bin/python3"
        )

        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.url?.path, "/api/v1/runtimes/configure")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
        XCTAssertTrue(
            request.value(forHTTPHeaderField: "Cookie")?.contains(
                "aptus_desktop_session=0123456789012345678901234567890123456789"
            ) == true
        )
        let payload = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody))
                as? [String: String]
        )
        XCTAssertEqual(payload, [
            "runtime_id": "mlx-lm",
            "interpreter_path": "/opt/homebrew/bin/python3",
        ])
    }

    func testPlatformSnapshotRequestUsesPrivateCookieAndExpectedContract() throws {
        let session = BackendSession(
            origin: try XCTUnwrap(URL(string: "http://127.0.0.1:49152")),
            token: "0123456789012345678901234567890123456789",
            version: "0.2.0",
            logFile: URL(fileURLWithPath: "/tmp/aptus-test.log")
        )

        let request = try DesktopBackendRequestFactory.platformSnapshot(session: session)

        XCTAssertEqual(request.httpMethod, "GET")
        XCTAssertEqual(request.url?.path, "/api/v1/platform")
        XCTAssertNil(request.httpBody)
        XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "application/json")
        XCTAssertTrue(
            request.value(forHTTPHeaderField: "Cookie")?.contains(
                "aptus_desktop_session=0123456789012345678901234567890123456789"
            ) == true
        )
    }

    func testRuntimeInventoryRequestUsesPrivateCookieAndExpectedContract() throws {
        let session = BackendSession(
            origin: try XCTUnwrap(URL(string: "http://127.0.0.1:49152")),
            token: "0123456789012345678901234567890123456789",
            version: "0.2.0",
            logFile: URL(fileURLWithPath: "/tmp/aptus-test.log")
        )

        let request = try DesktopBackendRequestFactory.runtimeInventory(session: session)

        XCTAssertEqual(request.httpMethod, "GET")
        XCTAssertEqual(request.url?.path, "/api/v1/runtimes")
        XCTAssertNil(request.httpBody)
        XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "application/json")
        XCTAssertTrue(
            request.value(forHTTPHeaderField: "Cookie")?.contains(
                "aptus_desktop_session=0123456789012345678901234567890123456789"
            ) == true
        )
    }

    func testRuntimeInventoryHydratesPersistedAvailableMLXInterpreter() throws {
        let path = "/opt/aptus/mlx/bin/python"
        let selection = try DesktopBackendClient.persistedMLXRuntimeSelection(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": ["mlx-lm": path],
            "available": ["mlx-lm": [path]],
            "compatible": ["mlx-lm": [path]],
            "interpreters": [],
        ])

        XCTAssertEqual(selection, .configured(path: path))
    }

    func testRuntimeInventoryHydratesEveryMLXProbeWithBoundedEvidence() throws {
        let readyPath = "/opt/aptus/mlx/bin/python"
        let missingPath = "/usr/bin/python3"
        let inventory = try DesktopBackendClient.mlxRuntimeInventory(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": ["mlx-lm": readyPath],
            "available": ["mlx-lm": [readyPath]],
            "compatible": ["mlx-lm": [readyPath]],
            "interpreters": [
                [
                    "path": readyPath,
                    "source": "configured:APTUS_MLX_PYTHON",
                    "python_version": "3.12.9",
                    "error": NSNull(),
                    "runtimes": [
                        "mlx-lm": [
                            "available": true,
                            "compatible": true,
                            "versions": ["mlx": "0.31.2", "mlx-lm": "0.31.3"],
                        ],
                    ],
                ],
                [
                    "path": missingPath,
                    "source": "known-path:system",
                    "python_version": "3.9.6",
                    "error": NSNull(),
                    "runtimes": [
                        "mlx-lm": [
                            "available": false,
                            "compatible": false,
                            "reason": "ModuleNotFoundError: No module named 'mlx_lm'",
                        ],
                    ],
                ],
            ],
        ])

        XCTAssertEqual(inventory.selection, .configured(path: readyPath))
        XCTAssertEqual(inventory.candidates.count, 2)
        XCTAssertEqual(inventory.candidates[0], MLXInterpreterCandidate(
            path: readyPath,
            source: "configured:APTUS_MLX_PYTHON",
            pythonVersion: "3.12.9",
            probePassed: true,
            compatible: true,
            reason: "MLX 0.31.2 and MLX-LM 0.31.3 passed the exact Aptus runtime contract. Selection revalidates the contract before saving.",
            packageVersions: ["mlx": "0.31.2", "mlx-lm": "0.31.3"]
        ))
        XCTAssertFalse(inventory.candidates[1].probePassed)
        XCTAssertEqual(
            inventory.candidates[1].reason,
            "ModuleNotFoundError: No module named 'mlx_lm'"
        )
    }

    func testRuntimeInventoryRejectsMalformedInterpreterProbe() {
        XCTAssertThrowsError(try DesktopBackendClient.mlxRuntimeInventory(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": [:],
            "available": ["mlx-lm": []],
            "compatible": ["mlx-lm": []],
            "interpreters": [[
                "path": "/usr/bin/python3",
                "source": "known-path:system",
                "runtimes": ["mlx-lm": ["available": 1]],
            ]],
        ]))
    }

    func testRuntimeInventoryRejectsCompatibleListThatContradictsProbeEvidence() {
        let path = "/opt/aptus/mlx/bin/python"
        XCTAssertThrowsError(try DesktopBackendClient.mlxRuntimeInventory(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": ["mlx-lm": path],
            "available": ["mlx-lm": [path]],
            "compatible": ["mlx-lm": [path]],
            "interpreters": [[
                "path": path,
                "source": "configured:APTUS_MLX_PYTHON",
                "python_version": "3.12.9",
                "error": NSNull(),
                "runtimes": [
                    "mlx-lm": [
                        "available": true,
                        "compatible": false,
                        "versions": ["mlx": "0.31.1", "mlx-lm": "0.31.3"],
                        "compatibility_reason": "The exact dependency contract did not pass.",
                    ],
                ],
            ]],
        ]))
    }

    func testRuntimeInventoryRejectsCompatibleProbeWithoutSuccessfulImport() {
        let path = "/opt/aptus/mlx/bin/python"
        XCTAssertThrowsError(try DesktopBackendClient.mlxRuntimeInventory(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": ["mlx-lm": path],
            "available": ["mlx-lm": []],
            "compatible": ["mlx-lm": [path]],
            "interpreters": [[
                "path": path,
                "source": "configured:APTUS_MLX_PYTHON",
                "python_version": "3.12.9",
                "error": NSNull(),
                "runtimes": [
                    "mlx-lm": [
                        "available": false,
                        "compatible": true,
                        "reason": "MLX-LM import failed.",
                    ],
                ],
            ]],
        ]))
    }

    func testRuntimeInventorySeparatesUnconfiguredUnavailableAndInvalidSelections() throws {
        let path = "/opt/aptus/mlx/bin/python"
        let notConfigured = try DesktopBackendClient.persistedMLXRuntimeSelection(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": [:],
            "available": ["mlx-lm": []],
            "compatible": ["mlx-lm": []],
            "interpreters": [],
        ])
        XCTAssertEqual(notConfigured, .notConfigured)

        let unavailable = try DesktopBackendClient.persistedMLXRuntimeSelection(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": ["mlx-lm": path],
            "available": ["mlx-lm": []],
            "compatible": ["mlx-lm": []],
            "interpreters": [[
                "path": path,
                "runtimes": [
                    "mlx-lm": [
                        "available": false,
                        "compatible": false,
                        "reason": "MLX-LM import failed.",
                    ],
                ],
            ]],
        ])
        XCTAssertEqual(
            unavailable,
            .unavailable(path: path, reason: "MLX-LM import failed.")
        )

        let invalid = try DesktopBackendClient.persistedMLXRuntimeSelection(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": ["mlx-lm": "relative/python"],
            "available": ["mlx-lm": []],
            "compatible": ["mlx-lm": []],
            "interpreters": [],
        ])
        XCTAssertEqual(
            invalid,
            .invalid(
                path: "relative/python",
                reason: "The persisted MLX-LM interpreter path is not a canonical absolute path."
            )
        )
    }

    func testRuntimeInventoryRejectsMalformedCompatibleContract() {
        XCTAssertThrowsError(try DesktopBackendClient.persistedMLXRuntimeSelection(from: [
            "schema_version": "aptus.runtime-inventory.v1",
            "selected": [:],
            "compatible": ["mlx-lm": "not-an-array"],
        ]))
    }

    func testPlatformSnapshotParsesMeasuredMemoryWithoutInventingPilotEvidence() throws {
        let gibibyte: UInt64 = 1_024 * 1_024 * 1_024
        let snapshot = try DesktopBackendClient.platformMemorySnapshot(from: [
            "status": "ok",
            "platform": [
                "unified_memory_bytes": 64 * gibibyte,
                "available_memory_bytes": 23 * gibibyte,
                "memory_free_percent": 36,
                "metal_gpu_core_count": 20,
                "observed_at": "2026-07-22T18:00:00+00:00",
            ],
        ])

        XCTAssertEqual(snapshot.availableMemoryBytes, 23 * gibibyte)
        XCTAssertEqual(snapshot.memoryFreePercent, 36)
        XCTAssertEqual(snapshot.metalGPUCoreCount, 20)
        XCTAssertEqual(snapshot.availableMemoryDescription, "23 GiB")
        XCTAssertEqual(snapshot.memoryFreeDescription, "36% free")
        XCTAssertEqual(snapshot.aptusReserveDescription, "8 GiB minimum")
        XCTAssertEqual(snapshot.pilotPeakDescription, "Not measured")
        XCTAssertNil(snapshot.pilotPeakBytes)
    }

    func testPlatformSnapshotRejectsImpossibleMemoryFacts() {
        let gibibyte: UInt64 = 1_024 * 1_024 * 1_024
        XCTAssertThrowsError(try DesktopBackendClient.platformMemorySnapshot(from: [
            "status": "ok",
            "platform": [
                "unified_memory_bytes": 16 * gibibyte,
                "available_memory_bytes": 20 * gibibyte,
                "memory_free_percent": 101,
            ],
        ]))
    }

    func testShellLoadsPlatformSnapshotWithoutBlockingInitialization() {
        let loader = PlatformSnapshotLoaderStub()
        let model = DesktopShellModel(
            machine: testMachine,
            platformSnapshotLoader: loader
        )
        XCTAssertEqual(loader.loadCount, 1)
        XCTAssertEqual(model.platformMemorySnapshotState, .loading)

        let snapshot = PlatformMemorySnapshot(
            availableMemoryBytes: 12 * 1_024 * 1_024 * 1_024,
            memoryFreePercent: 44,
            metalGPUCoreCount: nil,
            pilotPeakBytes: nil,
            observedAt: nil
        )
        loader.resolve(.success(snapshot))

        XCTAssertEqual(model.platformMemorySnapshotState, .available(snapshot))
    }

    func testShellHydratesPersistedRuntimeWithoutBlockingInitialization() {
        let loader = RuntimeInventoryLoaderStub()
        let platformLoader = PlatformSnapshotLoaderStub()
        let model = DesktopShellModel(
            machine: testMachine,
            runtimeInventoryLoader: loader,
            platformSnapshotLoader: platformLoader
        )

        XCTAssertEqual(loader.loadCount, 1)
        XCTAssertEqual(platformLoader.loadCount, 1)
        XCTAssertEqual(model.mlxRuntimeConfiguration, .loading)
        XCTAssertEqual(model.platformMemorySnapshotState, .loading)

        let candidate = MLXInterpreterCandidate(
            path: "/managed/mlx/python",
            source: "configured:APTUS_MLX_PYTHON",
            pythonVersion: "3.12.9",
            probePassed: true,
            compatible: true,
            reason: "MLX-LM imported successfully.",
            packageVersions: [:]
        )
        loader.resolve(.success(MLXRuntimeInventory(
            selection: .configured(path: "/managed/mlx/python"),
            candidates: [candidate]
        )))

        XCTAssertEqual(
            model.mlxRuntimeConfiguration,
            .configured(path: "/managed/mlx/python")
        )
        XCTAssertNil(model.mlxRuntimeConfigurationErrorMessage)
        XCTAssertEqual(model.mlxInterpreterCandidates, [candidate])
    }

    func testShellPreservesUnavailableAndInvalidPersistedRuntimeStates() {
        let unavailableLoader = RuntimeInventoryLoaderStub()
        let unavailableModel = DesktopShellModel(
            machine: testMachine,
            runtimeInventoryLoader: unavailableLoader
        )
        unavailableLoader.resolve(.success(MLXRuntimeInventory(
            selection: .unavailable(
                path: "/missing/mlx/python",
                reason: "The persisted interpreter is not present."
            ),
            candidates: []
        )))
        XCTAssertEqual(
            unavailableModel.mlxRuntimeConfiguration,
            .unavailable(
                path: "/missing/mlx/python",
                reason: "The persisted interpreter is not present."
            )
        )

        let invalidLoader = RuntimeInventoryLoaderStub()
        let invalidModel = DesktopShellModel(
            machine: testMachine,
            runtimeInventoryLoader: invalidLoader
        )
        invalidLoader.resolve(.success(MLXRuntimeInventory(
            selection: .invalid(
                path: "relative/python",
                reason: "The persisted path is invalid."
            ),
            candidates: []
        )))
        XCTAssertEqual(
            invalidModel.mlxRuntimeConfiguration,
            .invalid(
                path: "relative/python",
                reason: "The persisted path is invalid."
            )
        )
    }

    func testShellReportsInventoryFailureWithoutClaimingNoSelectionExists() {
        let loader = RuntimeInventoryLoaderStub()
        let model = DesktopShellModel(
            machine: testMachine,
            runtimeInventoryLoader: loader
        )

        loader.resolve(.failure(DesktopBackendClientError.invalidRuntimeInventory))

        XCTAssertEqual(
            model.mlxRuntimeConfiguration,
            .inventoryUnavailable(
                message: "The private local service returned an invalid runtime inventory."
            )
        )
    }

    func testRuntimeConfigurationSurfacesBackendValidationDetail() {
        XCTAssertEqual(
            DesktopBackendClient.errorMessage(from: [
                "detail": [
                    "error": "runtime_configuration_invalid",
                    "details": "The selected interpreter does not provide MLX-LM.",
                ],
            ]),
            "The selected interpreter does not provide MLX-LM."
        )
    }

    func testShellPersistsSelectedMLXPythonThroughRuntimeConfigurator() throws {
        let python = "/usr/bin/python3"
        guard FileManager.default.isExecutableFile(atPath: python) else {
            throw XCTSkip("The macOS system Python shim is unavailable.")
        }
        let stub = RuntimeConfiguratorStub(result: .success(RuntimeConfigurationResult(
            runtimeID: "mlx-lm",
            interpreterPath: python
        )))
        let model = DesktopShellModel(
            machine: testMachine,
            runtimeConfigurator: stub
        )

        model.configureMLXInterpreter(at: python)

        XCTAssertEqual(stub.runtimeID, "mlx-lm")
        XCTAssertEqual(stub.interpreterPath, python)
        XCTAssertEqual(model.mlxRuntimeConfiguration, .configured(path: python))
        XCTAssertNil(model.mlxRuntimeConfigurationErrorMessage)
    }

    func testShellRejectsNonExecutableMLXPathBeforeCallingBackend() {
        let stub = RuntimeConfiguratorStub(result: .success(RuntimeConfigurationResult(
            runtimeID: "mlx-lm",
            interpreterPath: "/missing/python"
        )))
        let model = DesktopShellModel(
            machine: testMachine,
            runtimeConfigurator: stub
        )

        model.configureMLXInterpreter(at: "/missing/aptus-python")

        XCTAssertNil(stub.runtimeID)
        XCTAssertEqual(model.mlxRuntimeConfiguration, .notConfigured)
        XCTAssertEqual(
            model.mlxRuntimeConfigurationErrorMessage,
            "Choose an executable Python file."
        )
    }

    func testFailedRuntimeReplacementRetainsHydratedPersistedSelection() throws {
        let python = "/usr/bin/python3"
        guard FileManager.default.isExecutableFile(atPath: python) else {
            throw XCTSkip("The macOS system Python shim is unavailable.")
        }
        let loader = RuntimeInventoryLoaderStub()
        let configurator = RuntimeConfiguratorStub(
            result: .failure(DesktopBackendClientError.transport("test failure"))
        )
        let model = DesktopShellModel(
            machine: testMachine,
            runtimeConfigurator: configurator,
            runtimeInventoryLoader: loader
        )
        loader.resolve(.success(MLXRuntimeInventory(
            selection: .configured(path: "/persisted/mlx/python"),
            candidates: []
        )))

        model.configureMLXInterpreter(at: python)

        XCTAssertEqual(
            model.mlxRuntimeConfiguration,
            .configured(path: "/persisted/mlx/python")
        )
        XCTAssertEqual(
            model.mlxRuntimeConfigurationErrorMessage,
            "The private local service could not be reached: test failure"
        )
    }

    func testModelsViewExposesHiddenEnvironmentPickerAndSinglePrimaryAction() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/DesktopShell.swift")
        let contents = try String(contentsOf: source, encoding: .utf8)
        XCTAssertTrue(contents.contains("panel.showsHiddenFiles = true"))
        XCTAssertTrue(contents.contains("\"Change MLX Python…\""))
        XCTAssertTrue(contents.contains("\"Replace MLX Python…\""))
        XCTAssertTrue(contents.contains("Persisted MLX-LM interpreter not ready"))
        XCTAssertFalse(contents.contains("Persisted MLX-LM interpreter unavailable"))
        XCTAssertTrue(contents.contains("Persisted MLX-LM selection invalid"))
        XCTAssertTrue(contents.contains("runtimeID: \"mlx-lm\""))
        XCTAssertTrue(contents.contains("MLX environment doctor"))
        XCTAssertTrue(contents.contains("Use this Python"))
        XCTAssertTrue(contents.contains("It did not install, upgrade, or change any package."))
        XCTAssertTrue(contents.contains("python3 -m venv /path/to/aptus-mlx-env"))
        XCTAssertTrue(contents.contains("'mlx==0.31.2' 'mlx-lm==0.31.3'"))
    }

    private var testMachine: MachineProfile {
        MachineProfile(
            chipName: "Apple M Test",
            operatingSystemName: "macOS 26.0.0",
            availableProcessorCount: 18,
            physicalMemoryBytes: 64 * 1_024 * 1_024 * 1_024,
            metalDeviceName: "Apple M Test",
            metalRecommendedWorkingSetBytes: nil,
            hasUnifiedMemory: true,
            lmStudioInstalled: false,
            oMLXExecutablePath: nil
        )
    }
}
