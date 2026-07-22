import AppKit

final class ApplicationTerminationCoordinator {
    private(set) var replyPending = false

    func requestTermination(
        stop: (@escaping () -> Void) -> Void,
        reply: @escaping (Bool) -> Void
    ) -> NSApplication.TerminateReply {
        guard !replyPending else { return .terminateLater }
        replyPending = true
        stop { [weak self] in
            guard let self, self.replyPending else { return }
            self.replyPending = false
            reply(true)
        }
        return .terminateLater
    }
}

final class AptusApplication: NSObject, NSApplicationDelegate {
    private var mainWindowController: MainWindowController?
    private let terminationCoordinator = ApplicationTerminationCoordinator()

    func applicationDidFinishLaunching(_ notification: Notification) {
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
            return
        }
        NSApplication.shared.setActivationPolicy(.regular)
        if let mark = Bundle.main.image(forResource: "AptusMark") {
            NSApplication.shared.applicationIconImage = mark
        }
        let controller = makeMainWindowController()
        mainWindowController = controller
        controller.onWorkbenchReady = { [weak self] session in
            self?.completeLaunchProbe(session: session)
        }
        installMainMenu()
        controller.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        if !terminationCoordinator.replyPending {
            mainWindowController?.stopBackend()
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let mainWindowController else { return .terminateNow }
        return terminationCoordinator.requestTermination(
            stop: { completion in mainWindowController.stopBackend(completion: completion) },
            reply: { [weak sender] shouldTerminate in
                sender?.reply(toApplicationShouldTerminate: shouldTerminate)
            }
        )
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func makeMainWindowController() -> MainWindowController {
        let environment = ProcessInfo.processInfo.environment
        guard environment["APTUS_DESKTOP_LAUNCH_PROBE_FILE"] != nil,
              let rootPath = environment["APTUS_DESKTOP_LAUNCH_PROBE_ROOT"],
              !rootPath.isEmpty else {
            return MainWindowController()
        }
        let root = URL(fileURLWithPath: rootPath, isDirectory: true).standardizedFileURL
        do {
            let paths = try ApplicationPaths(
                applicationSupportRoot: root.appendingPathComponent("support", isDirectory: true),
                logsRoot: root.appendingPathComponent("logs", isDirectory: true),
                cachesRoot: root.appendingPathComponent("caches", isDirectory: true),
                sessionIdentifier: "packaged-launch"
            )
            return MainWindowController(backend: BackendController(pathsFactory: { paths }))
        } catch {
            return MainWindowController()
        }
    }

    private func completeLaunchProbe(session: BackendSession) {
        guard let path = ProcessInfo.processInfo.environment["APTUS_DESKTOP_LAUNCH_PROBE_FILE"],
              !path.isEmpty else { return }
        let probeURL = URL(fileURLWithPath: path).standardizedFileURL
        let payload: [String: Any] = [
            "backendReady": true,
            "host": session.origin.host ?? "",
            "port": session.origin.port ?? 0,
            "version": session.version,
            "windowVisible": mainWindowController?.window?.isVisible == true,
            "workbenchLoaded": true,
            "reactReady": true,
            "workbenchMarker": WorkbenchReadyMessagePolicy.marker,
        ]
        do {
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            try FileManager.default.createDirectory(
                at: probeURL.deletingLastPathComponent(),
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try data.write(to: probeURL, options: .atomic)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: probeURL.path
            )
            DispatchQueue.main.async {
                NSApplication.shared.terminate(nil)
            }
        } catch {
            NSSound.beep()
        }
    }

    private func installMainMenu() {
        let mainMenu = NSMenu()
        let appItem = NSMenuItem()
        mainMenu.addItem(appItem)
        let appMenu = NSMenu()
        appItem.submenu = appMenu
        appMenu.addItem(withTitle: "About Aptus", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Hide Aptus", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = appMenu.addItem(withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Aptus", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        let editItem = NSMenuItem()
        mainMenu.addItem(editItem)
        let editMenu = NSMenu(title: "Edit")
        editItem.submenu = editMenu
        editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

        let windowItem = NSMenuItem()
        mainMenu.addItem(windowItem)
        let windowMenu = NSMenu(title: "Window")
        windowItem.submenu = windowMenu
        windowMenu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        NSApplication.shared.windowsMenu = windowMenu
        NSApplication.shared.mainMenu = mainMenu
    }
}
