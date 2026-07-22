import AppKit

enum AptusPalette {
    static let cloud = NSColor(srgbRed: 0xF3 / 255, green: 0xF7 / 255, blue: 0xF8 / 255, alpha: 1)
    static let porcelain = NSColor.white
    static let graphite = NSColor(srgbRed: 0x17 / 255, green: 0x25 / 255, blue: 0x2B / 255, alpha: 1)
    static let circuitTeal = NSColor(srgbRed: 0x0B / 255, green: 0x66 / 255, blue: 0x70 / 255, alpha: 1)
    static let calibrationAmber = NSColor(srgbRed: 0xB7 / 255, green: 0x63 / 255, blue: 0x18 / 255, alpha: 1)
    static let faultRed = NSColor(srgbRed: 0xA4 / 255, green: 0x3A / 255, blue: 0x32 / 255, alpha: 1)
    static let mutedGraphite = graphite.withAlphaComponent(0.68)
    static let hairline = graphite.withAlphaComponent(0.14)
}
