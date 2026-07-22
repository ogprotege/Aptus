import XCTest
@testable import Aptus

final class DesktopBridgeTests: XCTestCase {
    func testInjectedBridgeMatchesBrowserContract() {
        let source = DesktopBridgeScript.source
        XCTAssertTrue(source.contains("platform: \"macos\""))
        XCTAssertTrue(source.contains("reportWorkbenchReady"))
        XCTAssertTrue(source.contains("aptus-workbench-v1"))
        XCTAssertTrue(source.contains("pickDataset"))
        XCTAssertTrue(source.contains("pickOutputDirectory"))
        XCTAssertTrue(source.contains("revealInFinder"))
        XCTAssertTrue(source.contains("__aptusDesktopResolve"))
        XCTAssertTrue(source.contains("window.webkit.messageHandlers.aptusDesktop"))
        XCTAssertTrue(source.contains("defaultRequestTimeoutMilliseconds = 30000"))
        XCTAssertTrue(source.contains("modalRequestTimeoutMilliseconds = 300000"))
        XCTAssertTrue(source.contains("requestTimeout(method)"))
        XCTAssertTrue(source.contains("clearTimeout(entry.timeout)"))
        XCTAssertTrue(source.contains("request_timeout"))
        XCTAssertTrue(source.contains("AptusDesktopError"))
    }

    func testWorkbenchReadySignalRequiresVersionedStableMarker() {
        XCTAssertTrue(WorkbenchReadyMessagePolicy.allows([
            "protocolVersion": 1,
            "marker": "aptus-workbench-v1",
        ]))
        XCTAssertFalse(WorkbenchReadyMessagePolicy.allows([
            "protocolVersion": 2,
            "marker": "aptus-workbench-v1",
        ]))
        XCTAssertFalse(WorkbenchReadyMessagePolicy.allows([
            "protocolVersion": 1,
            "marker": "wrong-marker",
        ]))
        XCTAssertFalse(WorkbenchReadyMessagePolicy.allows([:]))
    }

    func testFinalReadinessRequiresBothDocumentLoadAndReactHandshake() {
        var documentFirst = WorkbenchReadinessGate()
        XCTAssertFalse(documentFirst.noteDocumentLoaded())
        XCTAssertTrue(documentFirst.noteReactReady())
        XCTAssertFalse(documentFirst.noteReactReady())

        var reactFirst = WorkbenchReadinessGate()
        XCTAssertFalse(reactFirst.noteReactReady())
        XCTAssertTrue(reactFirst.noteDocumentLoaded())
        XCTAssertFalse(reactFirst.noteDocumentLoaded())
    }

    func testVectorMarkIsPackagedWithApplication() {
        XCTAssertNotNil(Bundle.main.url(forResource: "AptusMark", withExtension: "svg"))
    }

    func testBridgeAcceptsOnlyTheMainFrameAtTheExactSessionOrigin() throws {
        let origin = try XCTUnwrap(URL(string: "http://127.0.0.1:49152"))
        XCTAssertTrue(DesktopBridgeMessagePolicy.allows(
            isMainFrame: true,
            scheme: "http",
            host: "127.0.0.1",
            port: 49_152,
            expectedOrigin: origin
        ))
        XCTAssertFalse(DesktopBridgeMessagePolicy.allows(
            isMainFrame: false,
            scheme: "http",
            host: "127.0.0.1",
            port: 49_152,
            expectedOrigin: origin
        ))
        XCTAssertFalse(DesktopBridgeMessagePolicy.allows(
            isMainFrame: true,
            scheme: "https",
            host: "127.0.0.1",
            port: 49_152,
            expectedOrigin: origin
        ))
        XCTAssertFalse(DesktopBridgeMessagePolicy.allows(
            isMainFrame: true,
            scheme: "http",
            host: "localhost",
            port: 49_152,
            expectedOrigin: origin
        ))
        XCTAssertFalse(DesktopBridgeMessagePolicy.allows(
            isMainFrame: true,
            scheme: "http",
            host: "127.0.0.1",
            port: 49_153,
            expectedOrigin: origin
        ))
    }

    func testBridgeDecodesACompleteRequest() throws {
        let decoded = try DesktopBridgeRequest.decode([
            "id": "aptus-1",
            "method": "revealInFinder",
            "args": ["path": "/tmp/example"],
        ]).get()

        XCTAssertEqual(decoded.id, "aptus-1")
        XCTAssertEqual(decoded.method, "revealInFinder")
        XCTAssertEqual(decoded.args["path"] as? String, "/tmp/example")
    }

    func testBridgeReturnsCorrelatableTypedErrorForMalformedRequest() {
        let result = DesktopBridgeRequest.decode([
            "id": "aptus-2",
            "method": 42,
            "args": [String: Any](),
        ])

        guard case let .failure(error) = result else {
            return XCTFail("Expected malformed bridge request to fail")
        }
        XCTAssertEqual(error.id, "aptus-2")
        XCTAssertEqual(error.code, "invalid_request")
        XCTAssertFalse(error.message.isEmpty)
    }

    func testBridgeRejectsNonObjectArgsAsInvalidRequest() {
        let result = DesktopBridgeRequest.decode([
            "id": "aptus-3",
            "method": "revealInFinder",
            "args": "not-an-object",
        ])

        guard case let .failure(error) = result else {
            return XCTFail("Expected bridge args validation to fail")
        }
        XCTAssertEqual(error.id, "aptus-3")
        XCTAssertEqual(error.code, "invalid_request")
    }
}
