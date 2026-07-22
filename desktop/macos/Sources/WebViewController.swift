import AppKit
import WebKit

enum DesktopSessionCookie {
    static func make(session: BackendSession) -> HTTPCookie? {
        HTTPCookie(properties: [
            .originURL: session.origin,
            .domain: "127.0.0.1",
            .path: "/",
            .name: "aptus_desktop_session",
            .value: session.token,
            .discard: "TRUE",
            .sameSitePolicy: "Strict",
            .init("HttpOnly"): "TRUE",
        ])
    }
}

enum DesktopNavigationPolicy {
    static func isSameOrigin(_ url: URL, as origin: URL) -> Bool {
        url.scheme == origin.scheme
            && url.host == origin.host
            && url.port == origin.port
    }

    static func isExternalWebURL(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else { return false }
        return scheme == "http" || scheme == "https"
    }

    static func shouldOpenExternalURL(_ url: URL, navigationType: WKNavigationType) -> Bool {
        navigationType == .linkActivated && isExternalWebURL(url)
    }
}

struct WorkbenchReadinessGate {
    private(set) var documentLoaded = false
    private(set) var reactReady = false
    private var reported = false

    mutating func noteDocumentLoaded() -> Bool {
        documentLoaded = true
        return takeFinalReadiness()
    }

    mutating func noteReactReady() -> Bool {
        reactReady = true
        return takeFinalReadiness()
    }

    private mutating func takeFinalReadiness() -> Bool {
        guard documentLoaded, reactReady, !reported else { return false }
        reported = true
        return true
    }
}

final class WebViewController: NSViewController, WKNavigationDelegate, WKUIDelegate {
    private let session: BackendSession
    private var bridge: DesktopBridge?
    private var readinessGate = WorkbenchReadinessGate()
    private(set) var webView: WKWebView!
    var onFatalError: ((String) -> Void)?
    var onWorkbenchReady: (() -> Void)?

    init(session: BackendSession) {
        self.session = session
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.preferences.isTextInteractionEnabled = true
        let contentController = WKUserContentController()
        contentController.addUserScript(WKUserScript(
            source: DesktopBridgeScript.source,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        ))
        configuration.userContentController = contentController

        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsMagnification = true
        webView.underPageBackgroundColor = AptusPalette.cloud
        view = webView
    }

    override func viewDidAppear() {
        super.viewDidAppear()
        guard bridge == nil else { return }
        let bridge = DesktopBridge(window: view.window, expectedOrigin: session.origin)
        bridge.onWorkbenchReady = { [weak self] in
            guard let self, self.readinessGate.noteReactReady() else { return }
            self.onWorkbenchReady?()
        }
        self.bridge = bridge
        webView.configuration.userContentController.add(bridge, name: "aptusDesktop")
        installSessionCookieAndLoad()
    }

    deinit {
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: "aptusDesktop")
    }

    private func installSessionCookieAndLoad() {
        guard let cookie = DesktopSessionCookie.make(session: session) else {
            onFatalError?("Aptus could not create its private desktop session.")
            return
        }
        webView.configuration.websiteDataStore.httpCookieStore.setCookie(cookie) { [weak self] in
            guard let self else { return }
            var request = URLRequest(url: self.session.origin)
            request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
            self.webView.load(request)
        }
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let target = navigationAction.request.url else {
            decisionHandler(.cancel)
            return
        }
        if DesktopNavigationPolicy.isSameOrigin(target, as: session.origin) {
            decisionHandler(.allow)
            return
        }
        if DesktopNavigationPolicy.shouldOpenExternalURL(
            target,
            navigationType: navigationAction.navigationType
        ) {
            NSWorkspace.shared.open(target)
        }
        decisionHandler(.cancel)
    }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
    ) {
        onFatalError?("The local workbench could not load: \(error.localizedDescription)")
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        readinessGate = WorkbenchReadinessGate()
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard let loadedURL = webView.url,
              DesktopNavigationPolicy.isSameOrigin(loadedURL, as: session.origin) else {
            return
        }
        if readinessGate.noteDocumentLoaded() {
            onWorkbenchReady?()
        }
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        webView.reload()
    }

    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        if let url = navigationAction.request.url,
           !DesktopNavigationPolicy.isSameOrigin(url, as: session.origin),
           DesktopNavigationPolicy.shouldOpenExternalURL(
               url,
               navigationType: navigationAction.navigationType
           ) {
            NSWorkspace.shared.open(url)
        }
        return nil
    }
}
