import AppKit
import XCTest
@testable import Aptus

final class AptusApplicationTests: XCTestCase {
    func testTerminationCoordinatorWaitsForBackendStopBeforeReplying() {
        let coordinator = ApplicationTerminationCoordinator()
        var stopCompletion: ((BackendShutdownResult) -> Void)?
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

        stopCompletion?(.success)
        wait(for: [replied], timeout: 1)
        XCTAssertFalse(coordinator.replyPending)
    }

    func testTerminationCoordinatorRefusesQuitWhenBackendStopFails() {
        let coordinator = ApplicationTerminationCoordinator()
        var stopCompletion: ((BackendShutdownResult) -> Void)?
        let replied = expectation(description: "Application termination refusal sent")
        let survivor = BackendProcessObservation(
            identity: BackendProcessIdentity(pid: 42, startSeconds: 10, startMicroseconds: 20),
            parentPID: 1,
            state: .sleeping
        )
        let failure = BackendShutdownFailure(
            rootPID: 42,
            activeProcesses: [survivor],
            rootProcessRunning: true,
            signalAttempts: [],
            terminationHandlerObserved: false
        )

        let result = coordinator.requestTermination(
            stop: { stopCompletion = $0 },
            reply: { shouldTerminate in
                XCTAssertFalse(shouldTerminate)
                replied.fulfill()
            }
        )

        XCTAssertEqual(result, .terminateLater)
        stopCompletion?(.failure(failure))
        wait(for: [replied], timeout: 1)
        XCTAssertFalse(coordinator.replyPending)
    }
}
