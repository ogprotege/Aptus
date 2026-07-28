import AppKit
import SwiftUI

final class MainWindowController: NSWindowController, NSWindowDelegate {
    private let backend: BackendController
    private let startupController = StartupViewController()
    private var currentSession: BackendSession?
    private var shellModel: DesktopShellModel?
    var onBackendReady: ((BackendSession) -> Void)?
    var onWorkbenchReady: ((BackendSession) -> Void)?

    init(backend: BackendController = BackendController()) {
        self.backend = backend
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Aptus"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.toolbarStyle = .unified
        window.titlebarSeparatorStyle = .none
        window.minSize = NSSize(width: 1_020, height: 700)
        window.backgroundColor = .windowBackgroundColor
        window.center()
        super.init(window: window)
        window.delegate = self

        startupController.onRetry = { [weak self] in self?.retry() }
        startupController.onShowLog = { [weak self] in self?.showBackendLog() }
        backend.onStateChange = { [weak self] state in self?.apply(state) }
        contentViewController = startupController
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func start() {
        showWindow(nil)
        window?.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
        startupController.showStarting()
        backend.start()
    }

    func stopBackend(completion: ((BackendShutdownResult) -> Void)? = nil) {
        backend.stop(completion: completion)
    }

    func openWorkbench() {
        shellModel?.openWorkbench()
    }

    private func apply(_ state: BackendState) {
        switch state {
        case .starting:
            currentSession = nil
            shellModel = nil
            contentViewController = startupController
            startupController.showStarting()
        case let .ready(session):
            currentSession = session
            let controller = WebViewController(session: session)
            controller.onFatalError = { [weak self] message in self?.showWebFailure(message) }
            controller.onWorkbenchReady = { [weak self] in
                self?.onWorkbenchReady?(session)
            }
            let backendClient = DesktopBackendClient(session: session)
            let model = DesktopShellModel(
                retryWorkbench: { [weak controller] in controller?.retryLoad() },
                runtimeConfigurator: backendClient,
                runtimeInventoryLoader: backendClient,
                platformSnapshotLoader: backendClient
            )
            shellModel = model
            contentViewController = NSHostingController(rootView: AptusDesktopShellView(
                model: model,
                workbenchController: controller
            ))
            onBackendReady?(session)
        case let .failed(message):
            currentSession = nil
            shellModel = nil
            contentViewController = startupController
            startupController.showFailure(message)
        case .stopped, .stopping:
            break
        }
    }

    private func retry() {
        startupController.showStarting()
        backend.restart()
    }

    private func showWebFailure(_ message: String) {
        guard let shellModel else {
            contentViewController = startupController
            startupController.showFailure(message)
            return
        }
        shellModel.reportWorkbenchFailure(message)
    }

    private func showBackendLog() {
        if let logFile = currentSession?.logFile {
            NSWorkspace.shared.activateFileViewerSelecting([logFile])
            return
        }
        do {
            let paths = try ApplicationPaths(sessionIdentifier: "log-reveal")
            if FileManager.default.fileExists(atPath: paths.logFile.path) {
                NSWorkspace.shared.activateFileViewerSelecting([paths.logFile])
            } else {
                NSWorkspace.shared.open(paths.logsDirectory)
            }
        } catch {
            NSSound.beep()
        }
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        NSApplication.shared.terminate(nil)
        return false
    }
}
