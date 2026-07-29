import AppKit
import SwiftUI

enum AptusMarkAsset {
    private static let lightImage = load(named: "AptusMark")
    private static let nightImage = load(named: "AptusMarkNight")

    static func image(for colorScheme: ColorScheme) -> NSImage {
        colorScheme == .dark ? nightImage : lightImage
    }

    static func image(for appearance: NSAppearance) -> NSImage {
        let match = appearance.bestMatch(from: [.darkAqua, .aqua])
        return match == .darkAqua ? nightImage : lightImage
    }

    private static func load(named resourceName: String) -> NSImage {
        let source = Bundle.main.image(forResource: resourceName)
            ?? NSImage(systemSymbolName: "a.circle", accessibilityDescription: "Aptus")
            ?? NSImage(size: NSSize(width: 32, height: 32))
        guard let copy = source.copy() as? NSImage else { return source }
        copy.isTemplate = false
        copy.accessibilityDescription = "Aptus calibrated A"
        return copy
    }
}
