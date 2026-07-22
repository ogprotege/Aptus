import AppKit
import Darwin
import Foundation
import Metal

enum DesktopDeploymentPolicy {
    static let fallbackMajorVersion = 15
    static let tahoeMajorVersion = 26

    static func supports(_ version: OperatingSystemVersion) -> Bool {
        version.majorVersion >= fallbackMajorVersion
    }
}

enum DesktopVisualGeneration: Equatable {
    case macOS15Fallback
    case macOS26

    static func resolve(for version: OperatingSystemVersion) -> DesktopVisualGeneration {
        version.majorVersion >= DesktopDeploymentPolicy.tahoeMajorVersion
            ? .macOS26
            : .macOS15Fallback
    }
}

enum DesktopDestination: String, CaseIterable, Hashable, Identifiable {
    case home
    case machine
    case models
    case data
    case plans
    case runs

    var id: String { rawValue }

    var title: String {
        switch self {
        case .home: "Home"
        case .machine: "Machine"
        case .models: "Models"
        case .data: "Data"
        case .plans: "Plans"
        case .runs: "Runs"
        }
    }

    var systemImage: String {
        switch self {
        case .home: "house"
        case .machine: "desktopcomputer"
        case .models: "cube"
        case .data: "doc.text"
        case .plans: "list.bullet.clipboard"
        case .runs: "waveform.path.ecg"
        }
    }

    var accessibilitySummary: String {
        switch self {
        case .home: "Machine summary and recommended next action"
        case .machine: "Detected Apple Silicon capabilities"
        case .models: "Model runtime integrations"
        case .data: "Local dataset preparation"
        case .plans: "Training plan construction"
        case .runs: "Run validation and artifacts"
        }
    }
}

enum DesktopLaunchPolicy {
    static func presentsWorkbenchImmediately(environment: [String: String]) -> Bool {
        guard let path = environment["APTUS_DESKTOP_LAUNCH_PROBE_FILE"] else {
            return false
        }
        return !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

struct MachineProfile: Equatable {
    let chipName: String
    let operatingSystemName: String
    let availableProcessorCount: Int
    let physicalMemoryBytes: UInt64
    let metalDeviceName: String?
    let metalRecommendedWorkingSetBytes: UInt64?
    let hasUnifiedMemory: Bool
    let lmStudioInstalled: Bool
    let oMLXExecutablePath: String?

    var physicalMemoryDescription: String {
        Self.memoryDescription(bytes: physicalMemoryBytes)
    }

    var recommendedWorkingSetDescription: String? {
        guard let metalRecommendedWorkingSetBytes else { return nil }
        return Self.memoryDescription(bytes: metalRecommendedWorkingSetBytes)
    }

    static func memoryDescription(bytes: UInt64) -> String {
        let capped = bytes > UInt64(Int64.max) ? Int64.max : Int64(bytes)
        return ByteCountFormatter.string(fromByteCount: capped, countStyle: .memory)
    }

    static func gibibyteDescription(bytes: UInt64) -> String {
        let gibibytes = Double(bytes) / Double(1_024 * 1_024 * 1_024)
        if abs(gibibytes.rounded() - gibibytes) < 0.05 {
            return "\(Int(gibibytes.rounded())) GiB"
        }
        return String(format: "%.1f GiB", gibibytes)
    }
}

struct PlatformMemorySnapshot: Equatable {
    static let minimumMLXPlanningReserveBytes: UInt64 = 8 * 1_024 * 1_024 * 1_024

    let availableMemoryBytes: UInt64?
    let memoryFreePercent: Int?
    let metalGPUCoreCount: Int?
    let pilotPeakBytes: UInt64?
    let observedAt: String?

    var availableMemoryDescription: String {
        guard let availableMemoryBytes else { return "Not reported" }
        return MachineProfile.gibibyteDescription(bytes: availableMemoryBytes)
    }

    var memoryFreeDescription: String {
        guard let memoryFreePercent else { return "Not reported" }
        return "\(memoryFreePercent)% free"
    }

    var aptusReserveDescription: String {
        "\(MachineProfile.gibibyteDescription(bytes: Self.minimumMLXPlanningReserveBytes)) minimum"
    }

    var pilotPeakDescription: String {
        guard let pilotPeakBytes else { return "Not measured" }
        return MachineProfile.gibibyteDescription(bytes: pilotPeakBytes)
    }
}

enum PlatformMemorySnapshotState: Equatable {
    case loading
    case available(PlatformMemorySnapshot)
    case unavailable

    var snapshot: PlatformMemorySnapshot? {
        guard case let .available(snapshot) = self else { return nil }
        return snapshot
    }

    var unavailableDescription: String {
        switch self {
        case .loading:
            "Measuring…"
        case .available:
            "Not reported"
        case .unavailable:
            "Unavailable"
        }
    }
}

enum MachineProfiler {
    static func current(
        processInfo: ProcessInfo = .processInfo,
        fileManager: FileManager = .default
    ) -> MachineProfile {
        let version = processInfo.operatingSystemVersion
        let operatingSystemName = "macOS \(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
        let metalDevice = MTLCreateSystemDefaultDevice()
        let home = fileManager.homeDirectoryForCurrentUser
        let lmStudioPaths = [
            URL(fileURLWithPath: "/Applications/LM Studio.app", isDirectory: true),
            home.appendingPathComponent("Applications/LM Studio.app", isDirectory: true),
        ]
        let oMLXPaths = [
            URL(fileURLWithPath: "/opt/homebrew/bin/omlx"),
            URL(fileURLWithPath: "/usr/local/bin/omlx"),
            home.appendingPathComponent(".local/bin/omlx"),
        ]

        return MachineProfile(
            chipName: sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon Mac",
            operatingSystemName: operatingSystemName,
            availableProcessorCount: processInfo.activeProcessorCount,
            physicalMemoryBytes: processInfo.physicalMemory,
            metalDeviceName: metalDevice?.name,
            metalRecommendedWorkingSetBytes: metalDevice?.recommendedMaxWorkingSetSize,
            hasUnifiedMemory: metalDevice?.hasUnifiedMemory == true,
            lmStudioInstalled: lmStudioPaths.contains {
                fileManager.fileExists(atPath: $0.path)
            },
            oMLXExecutablePath: oMLXPaths.first {
                fileManager.isExecutableFile(atPath: $0.path)
            }?.path
        )
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 1 else {
            return nil
        }
        var value = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else {
            return nil
        }
        return String(cString: value)
    }
}
