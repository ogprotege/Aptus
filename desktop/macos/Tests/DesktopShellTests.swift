import Foundation
import XCTest
@testable import Aptus

final class DesktopShellTests: XCTestCase {
    func testNativeNavigationUsesTheRequiredProductOrder() {
        XCTAssertEqual(
            DesktopDestination.allCases.map(\.title),
            ["Home", "Workbench", "Machine", "Models"]
        )
        XCTAssertEqual(Set(DesktopDestination.allCases.map(\.systemImage)).count, 4)
        XCTAssertTrue(DesktopDestination.allCases.allSatisfy {
            !$0.accessibilitySummary.isEmpty
        })
    }

    func testDeploymentPolicyUsesMacOS15FallbackAndMacOS26Generation() {
        XCTAssertFalse(DesktopDeploymentPolicy.supports(OperatingSystemVersion(
            majorVersion: 14,
            minorVersion: 7,
            patchVersion: 0
        )))
        XCTAssertTrue(DesktopDeploymentPolicy.supports(OperatingSystemVersion(
            majorVersion: 15,
            minorVersion: 0,
            patchVersion: 0
        )))
        XCTAssertEqual(
            DesktopVisualGeneration.resolve(for: OperatingSystemVersion(
                majorVersion: 15,
                minorVersion: 6,
                patchVersion: 0
            )),
            .macOS15Fallback
        )
        XCTAssertEqual(
            DesktopVisualGeneration.resolve(for: OperatingSystemVersion(
                majorVersion: 26,
                minorVersion: 0,
                patchVersion: 0
            )),
            .macOS26
        )
    }

    func testShellModelUsesAndRecoversThePrimaryWorkbenchDestination() {
        var retryCount = 0
        let model = DesktopShellModel(
            machine: MachineProfile(
                chipName: "Apple M Test",
                operatingSystemName: "macOS 26.0.0",
                availableProcessorCount: 18,
                physicalMemoryBytes: 64 * 1_024 * 1_024 * 1_024,
                metalDeviceName: "Apple M Test",
                metalRecommendedWorkingSetBytes: nil,
                hasUnifiedMemory: true,
                lmStudioInstalled: false,
                oMLXExecutablePath: nil
            ),
            retryWorkbench: { retryCount += 1 }
        )
        XCTAssertEqual(model.selection, .workbench)

        model.selection = .machine
        model.openWorkbench()
        XCTAssertEqual(model.selection, .workbench)

        model.reportWorkbenchFailure("Local test failure")
        XCTAssertEqual(model.selection, .workbench)
        XCTAssertEqual(model.workbenchErrorMessage, "Local test failure")

        model.selection = .models
        model.openWorkbench()
        XCTAssertEqual(model.selection, .workbench)
        XCTAssertNil(model.workbenchErrorMessage)
        XCTAssertEqual(retryCount, 1)

        model.openWorkbench()
        XCTAssertEqual(retryCount, 1)
    }

    func testWorkbenchIsEmbeddedAndDuplicateWorkflowDestinationsAreRemoved() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/DesktopShell.swift")
        let contents = try String(contentsOf: source, encoding: .utf8)

        XCTAssertTrue(contents.contains("case .workbench:\n            WorkbenchControllerHost"))
        XCTAssertFalse(contents.contains("WorkbenchSheet"))
        XCTAssertFalse(contents.contains("DataView"))
        XCTAssertFalse(contents.contains("PlansView"))
        XCTAssertFalse(contents.contains("RunsView"))
    }

    func testMachineProfileLabelsMeasuredMemoryWithoutInventingCapacity() {
        let profile = MachineProfile(
            chipName: "Apple M Test",
            operatingSystemName: "macOS 26.0.0",
            availableProcessorCount: 18,
            physicalMemoryBytes: 64 * 1_024 * 1_024 * 1_024,
            metalDeviceName: "Apple M Test",
            metalRecommendedWorkingSetBytes: 48 * 1_024 * 1_024 * 1_024,
            hasUnifiedMemory: true,
            lmStudioInstalled: false,
            oMLXExecutablePath: nil
        )

        XCTAssertEqual(profile.chipName, "Apple M Test")
        XCTAssertEqual(profile.availableProcessorCount, 18)
        XCTAssertFalse(profile.physicalMemoryDescription.isEmpty)
        XCTAssertNotNil(profile.recommendedWorkingSetDescription)
        XCTAssertEqual(
            MachineProfile.gibibyteDescription(bytes: 64 * 1_024 * 1_024 * 1_024),
            "64 GiB"
        )
    }

    func testMachineMemorySnapshotKeepsReserveAndPilotEvidenceDistinct() {
        let snapshot = PlatformMemorySnapshot(
            availableMemoryBytes: nil,
            memoryFreePercent: nil,
            metalGPUCoreCount: nil,
            pilotPeakBytes: nil,
            observedAt: nil
        )

        XCTAssertEqual(snapshot.availableMemoryDescription, "Not reported")
        XCTAssertEqual(snapshot.memoryFreeDescription, "Not reported")
        XCTAssertEqual(snapshot.aptusReserveDescription, "8 GiB minimum")
        XCTAssertEqual(snapshot.pilotPeakDescription, "Not measured")
        XCTAssertEqual(
            PlatformMemorySnapshot.minimumMLXPlanningReserveBytes,
            8 * 1_024 * 1_024 * 1_024
        )
    }

    func testMachineViewShowsTheRequiredUnifiedMemoryLedger() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/DesktopShell.swift")
        let contents = try String(contentsOf: source, encoding: .utf8)

        XCTAssertTrue(contents.contains("Unified memory snapshot"))
        XCTAssertTrue(contents.contains("Installed pool"))
        XCTAssertTrue(contents.contains("Current available headroom"))
        XCTAssertTrue(contents.contains("System memory free"))
        XCTAssertTrue(contents.contains("Aptus reserve"))
        XCTAssertTrue(contents.contains("Pilot peak"))
        XCTAssertTrue(contents.contains("Measured Metal GPU cores"))
        XCTAssertFalse(contents.contains("Neural Engine cores"))
        XCTAssertTrue(contents.contains("No estimate is substituted."))
    }

    func testModelsViewNamesEveryRequiredLocalAndRemoteRuntimeRole() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/DesktopShell.swift")
        let contents = try String(contentsOf: source, encoding: .utf8)

        for runtime in ["MLX-LM", "PyTorch MPS", "LM Studio", "oMLX", "CUDA target"] {
            XCTAssertTrue(contents.contains("title: \"\(runtime)\""))
        }
        XCTAssertTrue(contents.contains("Primary local Aptus training runtime"))
        XCTAssertTrue(contents.contains("Compiler unavailable"))
        XCTAssertTrue(contents.contains("No reported interpreter passed the exact MLX-LM runtime contract"))
        XCTAssertFalse(contents.contains("No reported interpreter passed the MLX-LM import probe"))
    }

    func testCurrentMachineProfileReportsOnlyObservedCoreFacts() {
        let profile = MachineProfiler.current()
        XCTAssertFalse(profile.chipName.isEmpty)
        XCTAssertTrue(profile.operatingSystemName.hasPrefix("macOS "))
        XCTAssertGreaterThan(profile.availableProcessorCount, 0)
        XCTAssertGreaterThan(profile.physicalMemoryBytes, 0)
    }

    func testVisualPolicyHasGuardedTahoeAppearanceAndAccessibleFallback() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let shell = try String(
            contentsOf: directory.appendingPathComponent("Sources/DesktopShell.swift"),
            encoding: .utf8
        )
        XCTAssertTrue(shell.contains("#available(macOS 26.0, *)"))
        XCTAssertTrue(shell.contains(".glassEffect(.regular"))
        XCTAssertTrue(shell.contains(".regularMaterial"))
        XCTAssertTrue(shell.contains("accessibilityReduceTransparency"))
        XCTAssertTrue(shell.contains("Color(nsColor: .windowBackgroundColor)"))
        XCTAssertTrue(shell.contains("Color.orange"))
    }

    func testBrandMarkHasNoBakedIconMaskOrDecorativeAmber() throws {
        let mark = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Resources/AptusMark.svg")
        let contents = try String(contentsOf: mark, encoding: .utf8)
        XCTAssertTrue(contents.contains("Aptus calibrated A mark"))
        XCTAssertFalse(contents.contains("<rect"))
        XCTAssertFalse(contents.contains("rx="))
        XCTAssertFalse(contents.localizedCaseInsensitiveContains("#B76318"))
    }

    func testReadyStateHostsNativeShellInsteadOfWholeWindowWebView() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MainWindowController.swift")
        let contents = try String(contentsOf: source, encoding: .utf8)
        XCTAssertTrue(contents.contains("NSHostingController(rootView: AptusDesktopShellView"))
        XCTAssertTrue(contents.contains("runtimeInventoryLoader: backendClient"))
        XCTAssertFalse(contents.contains("contentViewController = controller"))
        XCTAssertTrue(contents.contains("case let .failed(message):\n            currentSession = nil\n            shellModel = nil"))
    }
}
