import XCTest

final class AptusBrandTests: XCTestCase {
    func testIconRendererPreservesNightColorsAndUsesSmallOpticalMaster() throws {
        let renderer = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/render_icon.swift")
        let contents = try String(contentsOf: renderer, encoding: .utf8)

        XCTAssertTrue(contents.contains("private enum AppIconSurface"))
        XCTAssertTrue(contents.contains("static let inset: CGFloat = 72"))
        XCTAssertTrue(contents.contains("static let cornerRadius: CGFloat = 132"))
        XCTAssertTrue(contents.contains("srgbRed: 0x17 / 255"))
        XCTAssertTrue(contents.contains("green: 0x25 / 255"))
        XCTAssertTrue(contents.contains("blue: 0x2B / 255"))
        XCTAssertTrue(contents.contains("usesSmallMaster: Bool"))
        XCTAssertTrue(contents.contains("item.usesSmallMaster ? smallSource : source"))
        XCTAssertFalse(contents.contains("sourceAtop"))
        XCTAssertFalse(contents.contains("NSColor.white.setFill()"))
    }

    func testRefinedMarkIsCenteredRoundedAndTwoColor() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let day = try String(
            contentsOf: directory.appendingPathComponent("Resources/AptusMark.svg"),
            encoding: .utf8
        )
        let night = try String(
            contentsOf: directory.appendingPathComponent("Resources/AptusMarkNight.svg"),
            encoding: .utf8
        )
        let small = try String(
            contentsOf: directory.appendingPathComponent("Resources/AptusMarkNightSmall.svg"),
            encoding: .utf8
        )

        let geometry = "M226 806 L460 244 C470 217 489 202 512 202 C535 202 554 217 564 244 L798 806"
        XCTAssertTrue(day.contains(geometry))
        XCTAssertTrue(day.contains("M256 608 H768"))
        XCTAssertTrue(day.contains("stroke=\"#20343B\" stroke-width=\"88\""))
        XCTAssertTrue(day.contains("stroke=\"#0C6E77\" stroke-width=\"64\""))
        XCTAssertTrue(night.contains("stroke=\"#EDF3F4\" stroke-width=\"88\""))
        XCTAssertTrue(night.contains("stroke=\"#72D0D4\" stroke-width=\"64\""))
        XCTAssertTrue(small.contains("stroke=\"#EDF3F4\" stroke-width=\"96\""))
        XCTAssertFalse(day.contains("<rect"))
        XCTAssertFalse(night.contains("<rect"))
    }

    func testNativeBrandSurfacesUseAdaptiveFullColorArtwork() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let asset = try String(
            contentsOf: directory.appendingPathComponent("Sources/AptusMarkAsset.swift"),
            encoding: .utf8
        )
        let shell = try String(
            contentsOf: directory.appendingPathComponent("Sources/DesktopShell.swift"),
            encoding: .utf8
        )
        let startup = try String(
            contentsOf: directory.appendingPathComponent("Sources/StartupViewController.swift"),
            encoding: .utf8
        )

        XCTAssertTrue(asset.contains("lightImage = load(named: \"AptusMark\")"))
        XCTAssertTrue(asset.contains("nightImage = load(named: \"AptusMarkNight\")"))
        XCTAssertTrue(asset.contains("copy.isTemplate = false"))
        XCTAssertTrue(shell.contains("@Environment(\\.colorScheme)"))
        XCTAssertTrue(shell.contains(".renderingMode(.original)"))
        XCTAssertTrue(startup.contains("markImageView?.image = AptusMarkAsset.image"))
        XCTAssertFalse(shell.contains("AptusPalette.brandTeal"))
        XCTAssertFalse(startup.contains("AptusPalette.brandTeal"))
        XCTAssertFalse(startup.contains("contentTintColor = .white"))
    }
}
