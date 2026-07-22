import AppKit
import UniformTypeIdentifiers
import WebKit

enum DesktopBridgeScript {
    static let source = #"""
    (() => {
      if (window.aptusDesktop) return;
      class AptusDesktopError extends Error {
        constructor(code, message) {
          super(message);
          this.name = "AptusDesktopError";
          this.code = code;
        }
      }
      const defaultRequestTimeoutMilliseconds = 30000;
      const modalRequestTimeoutMilliseconds = 300000;
      const requestTimeout = (method) => (
        method === "pickDataset" || method === "pickOutputDirectory"
          ? modalRequestTimeoutMilliseconds
          : defaultRequestTimeoutMilliseconds
      );
      const pending = new Map();
      let sequence = 0;
      const request = (method, args = {}) => new Promise((resolve, reject) => {
        const id = `aptus-${Date.now()}-${++sequence}`;
        const timeout = setTimeout(() => {
          if (!pending.delete(id)) return;
          reject(new AptusDesktopError(
            "request_timeout",
            "The native Aptus action did not respond in time."
          ));
        }, requestTimeout(method));
        pending.set(id, { resolve, reject, timeout });
        window.webkit.messageHandlers.aptusDesktop.postMessage({ id, method, args });
      });
      Object.defineProperty(window, "__aptusDesktopResolve", {
        value: (id, payload) => {
          const entry = pending.get(id);
          if (!entry) return;
          pending.delete(id);
          clearTimeout(entry.timeout);
          if (payload && payload.ok) entry.resolve(payload.value ?? null);
          else {
            const error = payload?.error;
            const code = typeof error?.code === "string" ? error.code : "native_action_failed";
            const message = typeof error?.message === "string"
              ? error.message
              : "The native Aptus action failed.";
            entry.reject(new AptusDesktopError(code, message));
          }
        },
        configurable: false,
        enumerable: false,
        writable: false,
      });
      Object.defineProperty(window, "aptusDesktop", {
        value: Object.freeze({
          platform: "macos",
          reportWorkbenchReady: () => {
            const marker = document.querySelector(
              '[data-aptus-workbench-ready="aptus-workbench-v1"]'
            );
            if (!marker) {
              return Promise.reject(new Error("The Aptus workbench is not ready."));
            }
            return request("workbenchReady", {
              protocolVersion: 1,
              marker: "aptus-workbench-v1",
            });
          },
          pickDataset: () => request("pickDataset"),
          pickOutputDirectory: () => request("pickOutputDirectory"),
          revealInFinder: (path) => request("revealInFinder", { path }),
        }),
        configurable: false,
        enumerable: true,
        writable: false,
      });
    })();
    """#
}

struct DesktopBridgeRequest {
    let id: String
    let method: String
    let args: [String: Any]

    static func decode(_ body: Any) -> Result<DesktopBridgeRequest, DesktopBridgeProtocolError> {
        guard let body = body as? [String: Any],
              let id = body["id"] as? String,
              !id.isEmpty else {
            return .failure(DesktopBridgeProtocolError(
                id: nil,
                code: "invalid_request",
                message: "The native request must include a nonempty string id."
            ))
        }
        guard let method = body["method"] as? String, !method.isEmpty else {
            return .failure(DesktopBridgeProtocolError(
                id: id,
                code: "invalid_request",
                message: "The native request must include a nonempty string method."
            ))
        }
        let rawArgs = body["args"] ?? [String: Any]()
        guard let args = rawArgs as? [String: Any] else {
            return .failure(DesktopBridgeProtocolError(
                id: id,
                code: "invalid_request",
                message: "The native request args must be an object."
            ))
        }
        return .success(DesktopBridgeRequest(id: id, method: method, args: args))
    }
}

struct DesktopBridgeProtocolError: Error, Equatable {
    let id: String?
    let code: String
    let message: String
}

enum WorkbenchReadyMessagePolicy {
    static let marker = "aptus-workbench-v1"
    static let protocolVersion = 1

    static func allows(_ args: [String: Any]) -> Bool {
        let version = (args["protocolVersion"] as? NSNumber)?.intValue
            ?? args["protocolVersion"] as? Int
        return version == protocolVersion && args["marker"] as? String == marker
    }
}

enum DesktopBridgeMessagePolicy {
    static func allows(
        isMainFrame: Bool,
        scheme: String,
        host: String,
        port: Int,
        expectedOrigin: URL
    ) -> Bool {
        guard isMainFrame,
              expectedOrigin.scheme == "http",
              expectedOrigin.host == "127.0.0.1",
              let expectedPort = expectedOrigin.port else {
            return false
        }
        return scheme.lowercased() == "http"
            && host == "127.0.0.1"
            && port == expectedPort
    }
}

final class DesktopBridge: NSObject, WKScriptMessageHandler {
    private weak var window: NSWindow?
    private let expectedOrigin: URL
    var onWorkbenchReady: (() -> Void)?

    init(window: NSWindow?, expectedOrigin: URL) {
        self.window = window
        self.expectedOrigin = expectedOrigin
    }

    func updateWindow(_ window: NSWindow?) {
        self.window = window
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        let securityOrigin = message.frameInfo.securityOrigin
        guard DesktopBridgeMessagePolicy.allows(
                  isMainFrame: message.frameInfo.isMainFrame,
                  scheme: securityOrigin.protocol,
                  host: securityOrigin.host,
                  port: securityOrigin.port,
                  expectedOrigin: expectedOrigin
              ),
              message.name == "aptusDesktop" else {
            return
        }
        let request: DesktopBridgeRequest
        switch DesktopBridgeRequest.decode(message.body) {
        case let .success(decoded):
            request = decoded
        case let .failure(error):
            if let id = error.id {
                reject(
                    id: id,
                    code: error.code,
                    message: error.message,
                    webView: message.webView
                )
            }
            return
        }
        switch request.method {
        case "workbenchReady":
            guard WorkbenchReadyMessagePolicy.allows(request.args) else {
                reject(
                    id: request.id,
                    code: "invalid_argument",
                    message: "Invalid workbench-ready signal.",
                    webView: message.webView
                )
                return
            }
            onWorkbenchReady?()
            resolve(id: request.id, value: true, webView: message.webView)
        case "pickDataset":
            pickDataset(id: request.id, webView: message.webView)
        case "pickOutputDirectory":
            pickOutputDirectory(id: request.id, webView: message.webView)
        case "revealInFinder":
            guard let path = request.args["path"] as? String, !path.isEmpty else {
                reject(
                    id: request.id,
                    code: "invalid_argument",
                    message: "A path is required.",
                    webView: message.webView
                )
                return
            }
            reveal(path: path, id: request.id, webView: message.webView)
        default:
            reject(
                id: request.id,
                code: "unsupported_action",
                message: "Unsupported native action: \(request.method)",
                webView: message.webView
            )
        }
    }

    private func pickDataset(id: String, webView: WKWebView?) {
        let panel = NSOpenPanel()
        panel.title = "Choose a training dataset"
        panel.message = "Aptus reads the selected file locally. Supported formats are JSON, JSONL, CSV, and text."
        panel.prompt = "Choose Dataset"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.resolvesAliases = true
        panel.allowedContentTypes = ["json", "jsonl", "csv", "txt"].compactMap {
            UTType(filenameExtension: $0)
        }
        present(panel) { response in
            let path = response == .OK ? panel.url?.path : nil
            self.resolve(id: id, value: path, webView: webView)
        }
    }

    private func pickOutputDirectory(id: String, webView: WKWebView?) {
        let panel = NSSavePanel()
        panel.title = "Choose a new bundle directory"
        panel.message = "Name a fresh directory. Aptus will not overwrite an existing nonempty bundle."
        panel.prompt = "Choose Bundle Directory"
        panel.nameFieldLabel = "Bundle name:"
        panel.nameFieldStringValue = "aptus-training-bundle"
        panel.canCreateDirectories = true
        present(panel) { response in
            let path = response == .OK ? panel.url?.path : nil
            self.resolve(id: id, value: path, webView: webView)
        }
    }

    private func reveal(path: String, id: String, webView: WKWebView?) {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            reject(
                id: id,
                code: "artifact_missing",
                message: "The selected Aptus artifact no longer exists.",
                webView: webView
            )
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting([url])
        resolve(id: id, value: true, webView: webView)
    }

    private func present(_ panel: NSSavePanel, completion: @escaping (NSApplication.ModalResponse) -> Void) {
        if let window {
            panel.beginSheetModal(for: window, completionHandler: completion)
        } else {
            completion(panel.runModal())
        }
    }

    private func resolve(id: String, value: Any?, webView: WKWebView?) {
        let payload: [String: Any] = ["ok": true, "value": value ?? NSNull()]
        send(id: id, payload: payload, webView: webView)
    }

    private func reject(id: String, code: String, message: String, webView: WKWebView?) {
        send(
            id: id,
            payload: ["ok": false, "error": ["code": code, "message": message]],
            webView: webView
        )
    }

    private func send(id: String, payload: [String: Any], webView: WKWebView?) {
        guard JSONSerialization.isValidJSONObject([id, payload]),
              let data = try? JSONSerialization.data(withJSONObject: [id, payload]),
              let json = String(data: data, encoding: .utf8) else {
            return
        }
        webView?.evaluateJavaScript("window.__aptusDesktopResolve(...\(json));")
    }
}
