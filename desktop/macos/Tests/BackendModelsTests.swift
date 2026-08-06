import Foundation
import XCTest
@testable import Aptus

final class BackendModelsTests: XCTestCase {
    func testReadinessAcceptsOnlyExpectedLoopbackOrigin() throws {
        let readiness = BackendReadiness(host: "127.0.0.1", port: 49_321, version: "0.2.0")
        XCTAssertEqual(
            try readiness.validatedOrigin(expectedVersion: "0.2.0").absoluteString,
            "http://127.0.0.1:49321"
        )
    }

    func testReadinessRejectsNonLoopbackHost() {
        let readiness = BackendReadiness(host: "0.0.0.0", port: 8_787, version: "0.2.0")
        XCTAssertThrowsError(try readiness.validatedOrigin(expectedVersion: "0.2.0"))
    }

    func testReadinessRejectsInvalidPort() {
        let readiness = BackendReadiness(host: "127.0.0.1", port: 0, version: "0.2.0")
        XCTAssertThrowsError(try readiness.validatedOrigin(expectedVersion: "0.2.0"))
    }

    func testReadinessRejectsMismatchedApplicationVersion() {
        let readiness = BackendReadiness(host: "127.0.0.1", port: 49_321, version: "0.1.0")
        XCTAssertThrowsError(try readiness.validatedOrigin(expectedVersion: "0.2.0"))
    }

    func testHealthResponseAcceptsCurrentContractAndUnknownProperties() throws {
        let data = Data(#"{"status":"ok","version":"0.2.0","api_contract_version":"aptus.api.v1","future":"allowed"}"#.utf8)
        let response = try JSONDecoder().decode(BackendHealthResponse.self, from: data)

        XCTAssertNoThrow(try response.validate(expectedVersion: "0.2.0"))
        XCTAssertEqual(response.status, "ok")
        XCTAssertEqual(response.apiContractVersion, "aptus.api.v1")
    }

    func testHealthResponseRejectsUnknownStatusContractAndVersion() throws {
        let invalidResponses = [
            BackendHealthResponse(
                status: "starting",
                version: "0.2.0",
                apiContractVersion: "aptus.api.v1"
            ),
            BackendHealthResponse(
                status: "ok",
                version: "0.2.0",
                apiContractVersion: "aptus.api.v2"
            ),
            BackendHealthResponse(
                status: "ok",
                version: "0.3.0",
                apiContractVersion: "aptus.api.v1"
            ),
        ]
        for response in invalidResponses {
            XCTAssertThrowsError(try response.validate(expectedVersion: "0.2.0"))
        }

        let missingContract = Data(#"{"status":"ok","version":"0.2.0"}"#.utf8)
        XCTAssertThrowsError(
            try JSONDecoder().decode(BackendHealthResponse.self, from: missingContract)
        )
    }

    func testSessionTokenHasAtLeastThirtyTwoRandomBytes() throws {
        let first = try SessionToken.generate()
        let second = try SessionToken.generate()
        XCTAssertNotEqual(first, second)
        XCTAssertGreaterThanOrEqual(Data(base64Encoded: first)?.count ?? 0, 32)
    }

    func testExplicitBackendOverrideWins() {
        let result = BackendExecutableResolver.resolve(
            environment: ["APTUS_DESKTOP_BACKEND": "/bin/echo"],
            bundle: .main
        )
        XCTAssertEqual(result, BackendExecutable(url: URL(fileURLWithPath: "/bin/echo"), leadingArguments: []))
    }

    func testDesktopCookieIsPrivateAndBoundToLoopback() throws {
        let session = BackendSession(
            origin: try XCTUnwrap(URL(string: "http://127.0.0.1:49152")),
            token: String(repeating: "a", count: 44),
            version: "0.2.0",
            logFile: URL(fileURLWithPath: "/tmp/aptus.log")
        )
        let cookie = try XCTUnwrap(DesktopSessionCookie.make(session: session))
        XCTAssertEqual(cookie.name, "aptus_desktop_session")
        XCTAssertEqual(cookie.domain, "127.0.0.1")
        XCTAssertEqual(cookie.path, "/")
        XCTAssertTrue(cookie.isHTTPOnly)
        XCTAssertTrue(cookie.isSessionOnly)
        XCTAssertNil(cookie.expiresDate)
    }

    func testDesktopNavigationPolicyRestrictsExternalSchemes() throws {
        let origin = try XCTUnwrap(URL(string: "http://127.0.0.1:49152"))
        XCTAssertTrue(DesktopNavigationPolicy.isSameOrigin(
            try XCTUnwrap(URL(string: "http://127.0.0.1:49152/workbench")),
            as: origin
        ))
        XCTAssertFalse(DesktopNavigationPolicy.isSameOrigin(
            try XCTUnwrap(URL(string: "https://127.0.0.1:49152/workbench")),
            as: origin
        ))
        XCTAssertTrue(DesktopNavigationPolicy.isExternalWebURL(
            try XCTUnwrap(URL(string: "https://aptus.example/docs"))
        ))
        XCTAssertTrue(DesktopNavigationPolicy.shouldOpenExternalURL(
            try XCTUnwrap(URL(string: "https://aptus.example/docs")),
            navigationType: .linkActivated
        ))
        XCTAssertFalse(DesktopNavigationPolicy.shouldOpenExternalURL(
            try XCTUnwrap(URL(string: "https://aptus.example/docs")),
            navigationType: .other
        ))
        XCTAssertFalse(DesktopNavigationPolicy.isExternalWebURL(
            try XCTUnwrap(URL(string: "file:///tmp/private"))
        ))
        XCTAssertFalse(DesktopNavigationPolicy.isExternalWebURL(
            try XCTUnwrap(URL(string: "aptus-dangerous://payload"))
        ))
    }
}
