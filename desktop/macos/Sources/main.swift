import AppKit

let application = NSApplication.shared
let applicationDelegate = AptusApplication()
application.delegate = applicationDelegate
application.run()
withExtendedLifetime(applicationDelegate) {}
