#!/usr/bin/env swift

import AppKit
import Foundation

enum IconError: LocalizedError {
    case usage
    case sourceUnreadable(String)
    case bitmapCreation(Int)
    case pngEncoding(String)
    case iconutil(Int32)

    var errorDescription: String? {
        switch self {
        case .usage:
            return "Usage: render_icon.swift SOURCE.svg OUTPUT.icns"
        case let .sourceUnreadable(path):
            return "The SVG icon could not be read: \(path)"
        case let .bitmapCreation(size):
            return "A \(size)-pixel icon bitmap could not be created."
        case let .pngEncoding(name):
            return "The icon PNG could not be encoded: \(name)"
        case let .iconutil(status):
            return "iconutil failed with status \(status)."
        }
    }
}

func render(source: NSImage, pixels: Int, output: URL) throws {
    guard let representation = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixels,
        pixelsHigh: pixels,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        throw IconError.bitmapCreation(pixels)
    }
    representation.size = NSSize(width: pixels, height: pixels)
    NSGraphicsContext.saveGraphicsState()
    guard let context = NSGraphicsContext(bitmapImageRep: representation) else {
        NSGraphicsContext.restoreGraphicsState()
        throw IconError.bitmapCreation(pixels)
    }
    NSGraphicsContext.current = context
    context.imageInterpolation = .high
    NSColor(srgbRed: 0xEA / 255, green: 0xF3 / 255, blue: 0xF4 / 255, alpha: 1).setFill()
    NSRect(x: 0, y: 0, width: pixels, height: pixels).fill()
    source.draw(
        in: NSRect(x: 0, y: 0, width: pixels, height: pixels),
        from: .zero,
        operation: .sourceOver,
        fraction: 1,
        respectFlipped: true,
        hints: [.interpolation: NSImageInterpolation.high]
    )
    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    guard let data = representation.representation(using: .png, properties: [:]) else {
        throw IconError.pngEncoding(output.lastPathComponent)
    }
    try data.write(to: output, options: .atomic)
}

do {
    guard CommandLine.arguments.count == 3 else { throw IconError.usage }
    let sourceURL = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
    guard let source = NSImage(contentsOf: sourceURL) else {
        throw IconError.sourceUnreadable(sourceURL.path)
    }

    let manager = FileManager.default
    let iconset = outputURL.deletingPathExtension().appendingPathExtension("iconset")
    if manager.fileExists(atPath: iconset.path) {
        try manager.removeItem(at: iconset)
    }
    try manager.createDirectory(at: iconset, withIntermediateDirectories: true)
    let outputs: [(name: String, pixels: Int)] = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1_024),
    ]
    for item in outputs {
        try render(
            source: source,
            pixels: item.pixels,
            output: iconset.appendingPathComponent(item.name)
        )
    }

    if manager.fileExists(atPath: outputURL.path) {
        try manager.removeItem(at: outputURL)
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
    process.arguments = ["--convert", "icns", "--output", outputURL.path, iconset.path]
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw IconError.iconutil(process.terminationStatus)
    }
    try manager.removeItem(at: iconset)
} catch {
    FileHandle.standardError.write(Data("Aptus icon error: \(error.localizedDescription)\n".utf8))
    exit(1)
}
