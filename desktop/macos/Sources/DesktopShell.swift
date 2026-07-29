import AppKit
import SwiftUI

enum MLXRuntimeConfigurationState: Equatable {
    case loading
    case notConfigured
    case configuring
    case configured(path: String)
    case unavailable(path: String, reason: String)
    case invalid(path: String?, reason: String)
    case inventoryUnavailable(message: String)
}

final class DesktopShellModel: ObservableObject {
    @Published var selection: DesktopDestination
    @Published var workbenchErrorMessage: String?
    @Published var mlxRuntimeConfiguration: MLXRuntimeConfigurationState
    @Published var mlxInterpreterCandidates: [MLXInterpreterCandidate]
    @Published var mlxRuntimeConfigurationErrorMessage: String?
    @Published var platformMemorySnapshotState: PlatformMemorySnapshotState

    let machine: MachineProfile
    private let retryWorkbench: (() -> Void)?
    private let runtimeConfigurator: RuntimeConfiguring?
    private let runtimeInventoryLoader: RuntimeInventoryLoading?
    private let platformSnapshotLoader: PlatformSnapshotLoading?
    private var workbenchNeedsRetry = false
    private var runtimeStateGeneration = 0

    init(
        machine: MachineProfile = MachineProfiler.current(),
        retryWorkbench: (() -> Void)? = nil,
        runtimeConfigurator: RuntimeConfiguring? = nil,
        runtimeInventoryLoader: RuntimeInventoryLoading? = nil,
        platformSnapshotLoader: PlatformSnapshotLoading? = nil
    ) {
        self.machine = machine
        selection = .workbench
        self.retryWorkbench = retryWorkbench
        self.runtimeConfigurator = runtimeConfigurator
        self.runtimeInventoryLoader = runtimeInventoryLoader
        self.platformSnapshotLoader = platformSnapshotLoader
        mlxRuntimeConfiguration = runtimeInventoryLoader == nil
            ? .notConfigured
            : .loading
        mlxInterpreterCandidates = []
        mlxRuntimeConfigurationErrorMessage = nil
        platformMemorySnapshotState = platformSnapshotLoader == nil
            ? .unavailable
            : .loading
        if runtimeInventoryLoader != nil {
            refreshPersistedMLXRuntime()
        }
        if platformSnapshotLoader != nil {
            refreshPlatformMemorySnapshot()
        }
    }

    func openWorkbench() {
        if workbenchNeedsRetry {
            retryWorkbench?()
            workbenchNeedsRetry = false
        }
        workbenchErrorMessage = nil
        selection = .workbench
    }

    func reportWorkbenchFailure(_ message: String) {
        workbenchNeedsRetry = true
        workbenchErrorMessage = message
    }

    func refreshPlatformMemorySnapshot() {
        guard let platformSnapshotLoader else {
            platformMemorySnapshotState = .unavailable
            return
        }
        platformMemorySnapshotState = .loading
        platformSnapshotLoader.loadPlatformMemorySnapshot { [weak self] result in
            switch result {
            case let .success(snapshot):
                self?.platformMemorySnapshotState = .available(snapshot)
            case .failure:
                self?.platformMemorySnapshotState = .unavailable
            }
        }
    }

    func refreshPersistedMLXRuntime() {
        guard let runtimeInventoryLoader else {
            mlxRuntimeConfiguration = .notConfigured
            return
        }
        runtimeStateGeneration += 1
        let generation = runtimeStateGeneration
        mlxRuntimeConfiguration = .loading
        mlxRuntimeConfigurationErrorMessage = nil
        runtimeInventoryLoader.loadMLXRuntimeInventory { [weak self] result in
            guard let self, self.runtimeStateGeneration == generation else {
                return
            }
            switch result {
            case let .success(inventory):
                self.mlxInterpreterCandidates = inventory.candidates
                switch inventory.selection {
                case .notConfigured:
                    self.mlxRuntimeConfiguration = .notConfigured
                case let .configured(path):
                    self.mlxRuntimeConfiguration = .configured(path: path)
                case let .unavailable(path, reason):
                    self.mlxRuntimeConfiguration = .unavailable(
                        path: path,
                        reason: reason
                    )
                case let .invalid(path, reason):
                    self.mlxRuntimeConfiguration = .invalid(
                        path: path,
                        reason: reason
                    )
                }
            case let .failure(error):
                self.mlxInterpreterCandidates = []
                self.mlxRuntimeConfiguration = .inventoryUnavailable(
                    message: error.localizedDescription
                )
            }
        }
    }

    func chooseMLXPython() {
        let panel = NSOpenPanel()
        panel.title = "Choose the MLX Python interpreter"
        panel.message = "Choose the exact Python executable that imports MLX-LM. Hidden virtual-environment folders are visible."
        panel.prompt = "Choose MLX Python"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.resolvesAliases = true
        panel.showsHiddenFiles = true

        let completion: (NSApplication.ModalResponse) -> Void = { [weak self] response in
            guard response == .OK, let path = panel.url?.standardizedFileURL.path else {
                return
            }
            self?.configureMLXInterpreter(at: path)
        }
        if let window = NSApp.keyWindow ?? NSApp.mainWindow {
            panel.beginSheetModal(for: window, completionHandler: completion)
        } else {
            completion(panel.runModal())
        }
    }

    func configureMLXInterpreter(
        at path: String,
        fileManager: FileManager = .default
    ) {
        guard fileManager.isExecutableFile(atPath: path) else {
            mlxRuntimeConfigurationErrorMessage = "Choose an executable Python file."
            return
        }
        guard let runtimeConfigurator else {
            mlxRuntimeConfigurationErrorMessage = "The private local service is not ready for runtime configuration."
            return
        }
        let priorState = mlxRuntimeConfiguration == .loading
            ? MLXRuntimeConfigurationState.notConfigured
            : mlxRuntimeConfiguration
        runtimeStateGeneration += 1
        let generation = runtimeStateGeneration
        mlxRuntimeConfigurationErrorMessage = nil
        mlxRuntimeConfiguration = .configuring
        runtimeConfigurator.configureRuntime(
            runtimeID: "mlx-lm",
            interpreterPath: path
        ) { [weak self] result in
            guard let self, self.runtimeStateGeneration == generation else {
                return
            }
            switch result {
            case let .success(configuration):
                self.mlxRuntimeConfiguration = .configured(
                    path: configuration.interpreterPath
                )
            case let .failure(error):
                self.mlxRuntimeConfiguration = priorState
                self.mlxRuntimeConfigurationErrorMessage = error.localizedDescription
            }
        }
    }
}

struct AptusDesktopShellView: View {
    @ObservedObject var model: DesktopShellModel
    let workbenchController: WebViewController

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 270)
        } detail: {
            destinationView
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .windowBackgroundColor))
        }
        .navigationTitle(model.selection.title)
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button(action: toggleSidebar) {
                    Label("Toggle Sidebar", systemImage: "sidebar.left")
                }
                .help("Show or hide the Aptus sidebar")
            }
            ToolbarItem(placement: .status) {
                Label("Local service connected", systemImage: "lock.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("Private local service connected")
            }
        }
        .alert(
            "Workbench unavailable",
            isPresented: Binding(
                get: { model.workbenchErrorMessage != nil },
                set: { if !$0 { model.workbenchErrorMessage = nil } }
            )
        ) {
            Button("Retry Workbench") {
                model.openWorkbench()
            }
            Button("Dismiss", role: .cancel) {
                model.workbenchErrorMessage = nil
            }
        } message: {
            Text(model.workbenchErrorMessage ?? "The local workbench could not load.")
        }
    }

    private var sidebar: some View {
        List(DesktopDestination.allCases, selection: $model.selection) { destination in
            Label(destination.title, systemImage: destination.systemImage)
                .tag(destination)
                .accessibilityHint(destination.accessibilitySummary)
        }
        .safeAreaInset(edge: .top) {
            SidebarBrandView()
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
        }
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.machine.chipName)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
                Text("Private and local by default")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
        }
        .listStyle(.sidebar)
    }

    @ViewBuilder
    private var destinationView: some View {
        switch model.selection {
        case .home:
            HomeView(
                machine: model.machine,
                memorySnapshotState: model.platformMemorySnapshotState,
                startPlanning: model.openWorkbench,
                reviewMachine: { model.selection = .machine }
            )
        case .workbench:
            WorkbenchControllerHost(controller: workbenchController)
        case .machine:
            MachineView(
                machine: model.machine,
                memorySnapshotState: model.platformMemorySnapshotState,
                openWorkbench: model.openWorkbench
            )
        case .models:
            ModelsView(
                machine: model.machine,
                runtimeConfiguration: model.mlxRuntimeConfiguration,
                interpreterCandidates: model.mlxInterpreterCandidates,
                runtimeConfigurationErrorMessage: model.mlxRuntimeConfigurationErrorMessage,
                chooseMLXPython: model.chooseMLXPython,
                chooseCandidate: { model.configureMLXInterpreter(at: $0) },
                openWorkbench: model.openWorkbench
            )
        }
    }

    private func toggleSidebar() {
        NSApp.sendAction(
            #selector(NSSplitViewController.toggleSidebar(_:)),
            to: nil,
            from: nil
        )
    }
}

private struct SidebarBrandView: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        HStack(spacing: 10) {
            Image(nsImage: AptusMarkAsset.image(for: colorScheme))
                .resizable()
                .renderingMode(.original)
                .scaledToFit()
                .frame(width: 28, height: 28)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text("Aptus")
                    .font(.headline)
                Text("Apple Silicon workbench")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct HomeView: View {
    let machine: MachineProfile
    let memorySnapshotState: PlatformMemorySnapshotState
    let startPlanning: () -> Void
    let reviewMachine: () -> Void

    var body: some View {
        DesktopScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("THIS MAC")
                        .font(.caption.weight(.semibold))
                        .tracking(1.2)
                        .foregroundStyle(.secondary)
                    Text("Your \(machine.chipName) is ready")
                        .font(.system(size: 34, weight: .semibold, design: .default))
                    Text("\(installedMemoryDescription) unified memory · \(gpuCapacityDescription) · \(machine.availableProcessorCount)-core CPU. Evaluate a model or review live headroom before planning.")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 10) {
                            startPlanningButton
                            reviewMachineButton
                        }
                        VStack(alignment: .leading, spacing: 10) {
                            startPlanningButton
                            reviewMachineButton
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(28)
                .adaptiveHeroSurface()

                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 14) {
                        SummaryMetric(title: "Unified memory", value: installedMemoryDescription)
                        SummaryMetric(title: "Current headroom", value: availableHeadroomDescription)
                        SummaryMetric(title: "Metal GPU", value: gpuCapacityDescription)
                    }
                    VStack(spacing: 14) {
                        SummaryMetric(title: "Unified memory", value: installedMemoryDescription)
                        SummaryMetric(title: "Current headroom", value: availableHeadroomDescription)
                        SummaryMetric(title: "Metal GPU", value: gpuCapacityDescription)
                    }
                }

                InformationCallout(
                    systemImage: "checkmark.shield",
                    title: "Aptus keeps claims measured",
                    message: "Hardware detection establishes capacity. Model fit and training support still require checkpoint, runtime, dataset, and workload validation."
                )
            }
        }
    }

    private var availableHeadroomDescription: String {
        memorySnapshotState.snapshot?.availableMemoryDescription
            ?? memorySnapshotState.unavailableDescription
    }

    private var startPlanningButton: some View {
        Button("Open Workbench", action: startPlanning)
            .controlSize(.large)
            .buttonStyle(.borderedProminent)
            .accessibilityHint("Opens the fine-tuning workflow")
    }

    private var reviewMachineButton: some View {
        Button("Review This Mac", action: reviewMachine)
            .controlSize(.large)
            .buttonStyle(.bordered)
            .accessibilityHint("Opens the detected machine details")
    }

    private var installedMemoryDescription: String {
        MachineProfile.gibibyteDescription(bytes: machine.physicalMemoryBytes)
    }

    private var gpuCapacityDescription: String {
        if let coreCount = memorySnapshotState.snapshot?.metalGPUCoreCount {
            return "\(coreCount)-core Metal GPU"
        }
        return machine.metalDeviceName ?? "Metal not detected"
    }
}

private struct MachineView: View {
    let machine: MachineProfile
    let memorySnapshotState: PlatformMemorySnapshotState
    let openWorkbench: () -> Void

    var body: some View {
        DesktopScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeading(
                    eyebrow: "DETECTED HARDWARE",
                    title: machine.chipName,
                    message: "Use the observed machine profile as the starting point. Aptus does not turn memory capacity into an unsupported model-fit promise."
                )

                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 14) {
                        SummaryMetric(title: "Available CPU cores", value: "\(machine.availableProcessorCount)")
                        SummaryMetric(title: "Operating system", value: machine.operatingSystemName)
                        SummaryMetric(title: "Metal device", value: machine.metalDeviceName ?? "Not detected")
                    }
                    VStack(spacing: 14) {
                        SummaryMetric(title: "Available CPU cores", value: "\(machine.availableProcessorCount)")
                        SummaryMetric(title: "Operating system", value: machine.operatingSystemName)
                        SummaryMetric(title: "Metal device", value: machine.metalDeviceName ?? "Not detected")
                    }
                }

                SolidSection(title: "Unified memory snapshot", systemImage: "memorychip") {
                    VStack(spacing: 0) {
                        MemoryFactRow(
                            title: "Installed pool",
                            value: MachineProfile.gibibyteDescription(
                                bytes: machine.physicalMemoryBytes
                            ),
                            detail: machine.hasUnifiedMemory
                                ? "Shared by the CPU and GPU on this Apple Silicon host."
                                : "Installed memory was measured, but Metal did not report a unified pool."
                        )
                        Divider()
                        MemoryFactRow(
                            title: "Current available headroom",
                            value: availableHeadroomDescription,
                            detail: "A point-in-time conservative estimate from the private local platform probe."
                        )
                        Divider()
                        MemoryFactRow(
                            title: "System memory free",
                            value: memoryFreeDescription,
                            detail: "The system-wide free percentage reported by the macOS memory-pressure utility. It can change while Aptus is open."
                        )
                        Divider()
                        MemoryFactRow(
                            title: "Aptus reserve",
                            value: aptusReserveDescription,
                            detail: "Held outside the MLX planning budget. This is a conservative planning rule, not measured use."
                        )
                        Divider()
                        MemoryFactRow(
                            title: "Pilot peak",
                            value: pilotPeakDescription,
                            detail: "No bound pilot evidence is loaded into this native summary. No estimate is substituted."
                        )
                    }
                    Text("Headroom and free percentage are a snapshot, not a promise that a checkpoint or training plan will fit.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                SolidSection(title: "Native compute paths", systemImage: "cpu") {
                    VStack(spacing: 0) {
                        CapabilityRow(
                            title: "Metal Performance Shaders",
                            status: machine.metalDeviceName == nil ? "Needs verification" : "Metal available",
                            detail: machine.metalDeviceName ?? "No default Metal device was reported.",
                            tone: machine.metalDeviceName == nil ? .warning : .normal
                        )
                        Divider()
                        CapabilityRow(
                            title: "MLX",
                            status: "Native path",
                            detail: "Apple Silicon is compatible. Aptus still verifies the selected runtime and checkpoint before use.",
                            tone: .normal
                        )
                        Divider()
                        CapabilityRow(
                            title: "CUDA",
                            status: "Remote only",
                            detail: "CUDA is not native on Apple Silicon. Use a validated NVIDIA target when a plan requires it.",
                            tone: .warning
                        )
                    }
                }

                DisclosureGroup("Technical details") {
                    VStack(alignment: .leading, spacing: 12) {
                        LabeledContent("Operating system", value: machine.operatingSystemName)
                        LabeledContent("Metal device", value: machine.metalDeviceName ?? "Not detected")
                        if let metalGPUCoreCount = memorySnapshotState.snapshot?.metalGPUCoreCount {
                            LabeledContent(
                                "Measured Metal GPU cores",
                                value: "\(metalGPUCoreCount)"
                            )
                        }
                        LabeledContent(
                            "Metal working-set advisory",
                            value: machine.recommendedWorkingSetDescription ?? "Not reported"
                        )
                        Text("The Metal working-set value is an operating-system advisory. It is not a guarantee that a specific model or training run will fit.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 10)
                }
                .padding(18)
                .solidSurface()

                Button("Open Machine Workbench", action: openWorkbench)
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private var availableHeadroomDescription: String {
        memorySnapshotState.snapshot?.availableMemoryDescription
            ?? memorySnapshotState.unavailableDescription
    }

    private var memoryFreeDescription: String {
        memorySnapshotState.snapshot?.memoryFreeDescription
            ?? memorySnapshotState.unavailableDescription
    }

    private var aptusReserveDescription: String {
        "\(MachineProfile.gibibyteDescription(bytes: PlatformMemorySnapshot.minimumMLXPlanningReserveBytes)) minimum"
    }

    private var pilotPeakDescription: String {
        memorySnapshotState.snapshot?.pilotPeakDescription ?? "Not measured"
    }
}

private struct ModelsView: View {
    let machine: MachineProfile
    let runtimeConfiguration: MLXRuntimeConfigurationState
    let interpreterCandidates: [MLXInterpreterCandidate]
    let runtimeConfigurationErrorMessage: String?
    let chooseMLXPython: () -> Void
    let chooseCandidate: (String) -> Void
    let openWorkbench: () -> Void

    var body: some View {
        DesktopScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeading(
                    eyebrow: "RUNTIME INTEGRATIONS",
                    title: "Models",
                    message: "Choose a model only after Aptus confirms the checkpoint, quantization, context, and runtime path."
                )

                SolidSection(title: "Detected integrations", systemImage: "shippingbox") {
                    VStack(spacing: 0) {
                        CapabilityRow(
                            title: "MLX-LM",
                            status: mlxRuntimeStatus,
                            detail: "Primary local Aptus training runtime for validated LoRA and QLoRA plans on Apple Silicon.",
                            tone: mlxRuntimeTone
                        )
                        Divider()
                        CapabilityRow(
                            title: "PyTorch MPS",
                            status: "Compiler unavailable",
                            detail: "Aptus detects this Metal-backed compatibility runtime, but does not present it as executable without a method compiler and evidence contract.",
                            tone: .warning
                        )
                        Divider()
                        CapabilityRow(
                            title: "LM Studio",
                            status: machine.lmStudioInstalled ? "Installed" : "Not detected",
                            detail: machine.lmStudioInstalled
                                ? "Aptus found the LM Studio application. Model compatibility remains a separate check."
                                : "Install LM Studio only if it belongs in your chosen local inference path.",
                            tone: .normal
                        )
                        Divider()
                        CapabilityRow(
                            title: "oMLX",
                            status: machine.oMLXExecutablePath == nil ? "Not detected" : "Executable found",
                            detail: machine.oMLXExecutablePath ?? "No supported oMLX command location was found.",
                            tone: .normal
                        )
                        Divider()
                        CapabilityRow(
                            title: "CUDA target",
                            status: "External host",
                            detail: "Aptus can compile portable output for a separately validated NVIDIA environment.",
                            tone: .warning
                        )
                    }
                }

                InformationCallout(
                    systemImage: "scope",
                    title: "Compatibility is model-specific",
                    message: "A runtime being installed does not prove that a checkpoint, adapter method, or training configuration is supported."
                )

                runtimeConfigurationStatus
                runtimeConfigurationErrorStatus
                environmentDoctor

                HStack(spacing: 12) {
                    Button(runtimeActionTitle, action: chooseMLXPython)
                        .controlSize(.large)
                        .buttonStyle(.borderedProminent)
                        .disabled(runtimeActionDisabled)
                        .help("Choose the exact Python executable that provides MLX-LM")
                    if runtimeActionDisabled {
                        ProgressView()
                            .controlSize(.small)
                            .accessibilityLabel(
                                runtimeConfiguration == .loading
                                    ? "Checking persisted MLX Python"
                                    : "Validating MLX Python"
                            )
                    }
                    Button("Open Model Workbench", action: openWorkbench)
                        .controlSize(.large)
                        .buttonStyle(.bordered)
                }
            }
        }
    }

    private var environmentDoctor: some View {
        SolidSection(title: "MLX environment doctor", systemImage: "stethoscope") {
            VStack(alignment: .leading, spacing: 14) {
                Text("Aptus asked the private local service to inspect a bounded set of likely Python executables. It did not install, upgrade, or change any package.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if interpreterCandidates.isEmpty {
                    Text(environmentDoctorEmptyMessage)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    VStack(spacing: 0) {
                        ForEach(Array(interpreterCandidates.enumerated()), id: \.element.id) { index, candidate in
                            if index > 0 {
                                Divider()
                            }
                            MLXInterpreterCandidateRow(
                                candidate: candidate,
                                isSelected: selectedRuntimePath == candidate.path,
                                isBusy: runtimeConfiguration == .configuring,
                                select: { chooseCandidate(candidate.path) }
                            )
                        }
                    }
                }

                if shouldShowExternalEnvironmentRecipe {
                    Divider()
                    VStack(alignment: .leading, spacing: 8) {
                        Text("No reported interpreter passed the exact MLX-LM runtime contract")
                            .font(.callout.weight(.semibold))
                        Text("Create a separate environment outside any compiled bundle. Aptus will inspect it only after you select its Python executable.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(Self.externalEnvironmentRecipe)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                Color(nsColor: .textBackgroundColor),
                                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
                            )
                            .accessibilityLabel("MLX environment setup commands")
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var runtimeConfigurationStatus: some View {
        switch runtimeConfiguration {
        case .loading:
            Text("Checking the persisted MLX-LM interpreter with the private local service.")
                .font(.callout)
                .foregroundStyle(.secondary)
        case .notConfigured:
            Text("Finder-launched apps do not inherit your shell runtime variables. Choose the MLX Python executable once so Aptus can validate and persist it.")
                .font(.callout)
                .foregroundStyle(.secondary)
        case .configuring:
            Text("Validating the selected interpreter with the private local service.")
                .font(.callout)
                .foregroundStyle(.secondary)
        case let .configured(path):
            VStack(alignment: .leading, spacing: 5) {
                Label("MLX-LM interpreter configured", systemImage: "checkmark.circle")
                    .font(.callout.weight(.semibold))
                Text(path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        case let .unavailable(path, reason):
            VStack(alignment: .leading, spacing: 5) {
                Label("Persisted MLX-LM interpreter not ready", systemImage: "exclamationmark.triangle")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.orange)
                Text(path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case let .invalid(path, reason):
            VStack(alignment: .leading, spacing: 5) {
                Label("Persisted MLX-LM selection invalid", systemImage: "exclamationmark.triangle")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.orange)
                if let path {
                    Text(path)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Text(reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case let .inventoryUnavailable(message):
            VStack(alignment: .leading, spacing: 5) {
                Label("Runtime status unavailable", systemImage: "exclamationmark.triangle")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.orange)
                Text("Aptus could not verify whether an MLX-LM interpreter is already persisted. The existing selection, if any, was not changed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var runtimeConfigurationErrorStatus: some View {
        if let runtimeConfigurationErrorMessage {
            VStack(alignment: .leading, spacing: 5) {
                Label("MLX-LM selection unchanged", systemImage: "exclamationmark.triangle")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(.orange)
                Text(runtimeConfigurationErrorMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var runtimeActionDisabled: Bool {
        runtimeConfiguration == .loading || runtimeConfiguration == .configuring
    }

    private var selectedRuntimePath: String? {
        switch runtimeConfiguration {
        case let .configured(path), let .unavailable(path, _):
            path
        case let .invalid(path, _):
            path
        case .loading, .notConfigured, .configuring, .inventoryUnavailable:
            nil
        }
    }

    private var environmentDoctorEmptyMessage: String {
        switch runtimeConfiguration {
        case .loading:
            "Checking likely Python interpreters."
        case .inventoryUnavailable:
            "Interpreter evidence is unavailable until the private local service can answer the runtime inventory request."
        default:
            "The local service did not report a likely Python interpreter."
        }
    }

    private var shouldShowExternalEnvironmentRecipe: Bool {
        guard !interpreterCandidates.contains(where: \.compatible) else {
            return false
        }
        switch runtimeConfiguration {
        case .loading, .inventoryUnavailable:
            return false
        default:
            return true
        }
    }

    private static let externalEnvironmentRecipe = """
    python3 -m venv /path/to/aptus-mlx-env
    /path/to/aptus-mlx-env/bin/python -m pip install --upgrade pip
    /path/to/aptus-mlx-env/bin/python -m pip install 'mlx==0.31.2' 'mlx-lm==0.31.3'
    """

    private var mlxRuntimeStatus: String {
        switch runtimeConfiguration {
        case .loading:
            "Checking"
        case .notConfigured:
            "Choose Python"
        case .configuring:
            "Validating"
        case .configured:
            "Ready"
        case .unavailable, .invalid, .inventoryUnavailable:
            "Needs attention"
        }
    }

    private var mlxRuntimeTone: CapabilityTone {
        switch runtimeConfiguration {
        case .configured:
            .normal
        case .loading, .configuring:
            .normal
        case .notConfigured, .unavailable, .invalid, .inventoryUnavailable:
            .warning
        }
    }

    private var runtimeActionTitle: String {
        switch runtimeConfiguration {
        case .loading:
            "Checking MLX Python…"
        case .notConfigured:
            "Choose MLX Python…"
        case .configuring:
            "Validating MLX Python…"
        case .configured:
            "Change MLX Python…"
        case .unavailable, .invalid:
            "Replace MLX Python…"
        case .inventoryUnavailable:
            "Choose or Replace MLX Python…"
        }
    }
}

private struct MLXInterpreterCandidateRow: View {
    let candidate: MLXInterpreterCandidate
    let isSelected: Bool
    let isBusy: Bool
    let select: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Label(
                    candidate.compatible
                        ? "Exact runtime contract passed"
                        : (candidate.probePassed
                            ? "Import passed, versions incompatible"
                            : "Import probe failed"),
                    systemImage: candidate.compatible
                        ? "checkmark.circle.fill"
                        : "exclamationmark.triangle.fill"
                )
                .font(.callout.weight(.semibold))
                .foregroundStyle(candidate.compatible ? Color.green : Color.orange)
                Spacer(minLength: 12)
                if candidate.compatible {
                    Button(isSelected ? "Selected" : "Use this Python", action: select)
                        .buttonStyle(.bordered)
                        .disabled(isSelected || isBusy)
                        .accessibilityHint(
                            "The private local service will validate the exact pinned MLX dependency contract before saving this interpreter."
                        )
                }
            }
            Text(candidate.path)
                .font(.caption.monospaced())
                .textSelection(.enabled)
            HStack(spacing: 8) {
                Text(candidate.source)
                if let pythonVersion = candidate.pythonVersion {
                    Text("Python \(pythonVersion)")
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            Text(candidate.reason)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 12)
    }
}

private struct DesktopScrollView<Content: View>: View {
    @ViewBuilder let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        ScrollView {
            content
                .frame(maxWidth: 980, alignment: .leading)
                .padding(.horizontal, 34)
                .padding(.vertical, 30)
                .frame(maxWidth: .infinity, alignment: .top)
        }
    }
}

private struct PageHeading: View {
    let eyebrow: String
    let title: String
    let message: String

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text(eyebrow)
                .font(.caption.weight(.semibold))
                .tracking(1.2)
                .foregroundStyle(.secondary)
            Text(title)
                .font(.largeTitle.weight(.semibold))
            Text(message)
                .font(.title3)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct SummaryMetric: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title3.weight(.semibold))
                .lineLimit(2)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
        .padding(18)
        .solidSurface()
    }
}

private struct SolidSection<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder let content: Content

    init(
        title: String,
        systemImage: String,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.systemImage = systemImage
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            content
        }
        .padding(20)
        .solidSurface()
    }
}

private struct InformationCallout: View {
    let systemImage: String
    let title: String
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: systemImage)
                .font(.title3)
                .foregroundStyle(.tint)
                .frame(width: 24)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.headline)
                Text(message)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(18)
        .solidSurface()
    }
}

private enum CapabilityTone {
    case normal
    case warning
}

private struct CapabilityRow: View {
    let title: String
    let status: String
    let detail: String
    let tone: CapabilityTone

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.body.weight(.medium))
                Text(detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 18)
            Text(status)
                .font(.caption.weight(.semibold))
                .foregroundStyle(tone == .warning ? Color.orange : Color.secondary)
                .multilineTextAlignment(.trailing)
        }
        .padding(.vertical, 13)
    }
}

private struct MemoryFactRow: View {
    let title: String
    let value: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 18) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.body.weight(.medium))
                Text(detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 18)
            Text(value)
                .font(.body.weight(.semibold))
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
        }
        .padding(.vertical, 13)
    }
}

private struct WorkbenchControllerHost: NSViewControllerRepresentable {
    let controller: WebViewController

    func makeNSViewController(context: Context) -> WebViewController {
        controller
    }

    func updateNSViewController(_ nsViewController: WebViewController, context: Context) {}
}

private extension View {
    func solidSurface() -> some View {
        modifier(SolidSurfaceModifier())
    }

    func adaptiveHeroSurface() -> some View {
        modifier(AdaptiveHeroSurfaceModifier())
    }
}

private struct SolidSurfaceModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.55), lineWidth: 1)
            }
    }
}

private struct AdaptiveHeroSurfaceModifier: ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    @ViewBuilder
    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: 20, style: .continuous)
        if reduceTransparency {
            content
                .background(Color(nsColor: .controlBackgroundColor), in: shape)
                .overlay { shape.stroke(Color(nsColor: .separatorColor), lineWidth: 1) }
        } else if #available(macOS 26.0, *) {
            content.glassEffect(.regular, in: shape)
        } else {
            content
                .background(.regularMaterial, in: shape)
                .overlay { shape.stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 1) }
        }
    }
}
