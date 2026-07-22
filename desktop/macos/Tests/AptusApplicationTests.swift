import AppKit
import XCTest
@testable import Aptus

final class AptusApplicationTests: XCTestCase {
    func testTerminationCoordinatorWaitsForBackendStopBeforeReplying() {
        let coordinator = ApplicationTerminationCoordinator()
        var stopCompletion: (() -> Void)?
        var duplicateStopRequested = false
        let replied = expectation(description: "Application termination reply sent")

        let result = coordinator.requestTermination(
            stop: { stopCompletion = $0 },
            reply: { shouldTerminate in
                XCTAssertTrue(shouldTerminate)
                replied.fulfill()
            }
        )
        XCTAssertEqual(result, .terminateLater)
        XCTAssertTrue(coordinator.replyPending)

        let duplicateResult = coordinator.requestTermination(
            stop: { _ in duplicateStopRequested = true },
            reply: { _ in XCTFail("A duplicate termination request must not reply.") }
        )
        XCTAssertEqual(duplicateResult, .terminateLater)
        XCTAssertFalse(duplicateStopRequested)

        stopCompletion?()
        wait(for: [replied], timeout: 1)
        XCTAssertFalse(coordinator.replyPending)
    }
}
