import XCTest

final class TileBrandTests: XCTestCase {
    func testIconRendererOwnsTheTileMaskAndReversedGlyph() throws {
        let renderer = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/render_icon.swift")
        let contents = try String(contentsOf: renderer, encoding: .utf8)

        XCTAssertTrue(contents.contains("static let inset: CGFloat = 72"))
        XCTAssertTrue(contents.contains("static let cornerRadius: CGFloat = 132"))
        XCTAssertTrue(contents.contains("srgbRed: 0x0C / 255"))
        XCTAssertTrue(contents.contains("bounds.fill(using: .sourceAtop)"))
        XCTAssertTrue(contents.contains("NSColor.white.setFill()"))
        XCTAssertTrue(contents.contains("[PREVIEW.png]"))
    }

    func testNativeBrandSurfacesApplyTheTileOutsideTheSourceGlyph() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let shell = try String(
            contentsOf: directory.appendingPathComponent("Sources/DesktopShell.swift"),
            encoding: .utf8
        )
        let startup = try String(
            contentsOf: directory.appendingPathComponent("Sources/StartupViewController.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(shell.contains("RoundedRectangle(cornerRadius: 3.6, style: .circular)"))
        XCTAssertTrue(shell.contains("AptusPalette.brandTeal"))
        XCTAssertTrue(shell.contains(".foregroundStyle(.white)"))
        XCTAssertTrue(startup.contains("markBackground.layer?.cornerRadius = 18.3"))
        XCTAssertTrue(startup.contains("mark.contentTintColor = .white"))
        XCTAssertTrue(startup.contains("AptusPalette.brandTeal.cgColor"))
    }
}
