import AppKit

final class StartupViewController: NSViewController {
    var onRetry: (() -> Void)?
    var onShowLog: (() -> Void)?

    private let statusLabel = NSTextField(labelWithString: "Starting Aptus")
    private let detailLabel = NSTextField(wrappingLabelWithString: "Preparing the private local planning service.")
    private let progress = NSProgressIndicator()
    private let statusRule = NSView()
    private let retryButton = NSButton(title: "Retry", target: nil, action: nil)
    private let logButton = NSButton(title: "Show Backend Log", target: nil, action: nil)

    override func loadView() {
        let root = NSView()
        root.wantsLayer = true
        root.layer?.backgroundColor = AptusPalette.cloud.cgColor

        let mark = NSImageView()
        mark.image = Bundle.main.image(forResource: "AptusMark")
        mark.imageScaling = .scaleProportionallyUpOrDown
        mark.translatesAutoresizingMaskIntoConstraints = false

        let eyebrow = NSTextField(labelWithString: "LOCAL FINE-TUNING WORKBENCH")
        eyebrow.font = .systemFont(ofSize: 11, weight: .semibold)
        eyebrow.textColor = AptusPalette.circuitTeal
        eyebrow.alignment = .left

        statusLabel.font = .systemFont(ofSize: 28, weight: .semibold)
        statusLabel.textColor = AptusPalette.graphite
        detailLabel.font = .systemFont(ofSize: 14, weight: .regular)
        detailLabel.textColor = AptusPalette.mutedGraphite
        detailLabel.maximumNumberOfLines = 3

        progress.style = .spinning
        progress.controlSize = .small
        progress.startAnimation(nil)

        statusRule.wantsLayer = true
        statusRule.layer?.backgroundColor = AptusPalette.circuitTeal.cgColor
        statusRule.translatesAutoresizingMaskIntoConstraints = false

        retryButton.bezelStyle = .rounded
        retryButton.target = self
        retryButton.action = #selector(retry)
        retryButton.isHidden = true
        logButton.bezelStyle = .rounded
        logButton.target = self
        logButton.action = #selector(showLog)
        logButton.isHidden = true

        let statusRow = NSStackView(views: [progress, detailLabel])
        statusRow.orientation = .horizontal
        statusRow.alignment = .centerY
        statusRow.spacing = 10

        let actions = NSStackView(views: [retryButton, logButton])
        actions.orientation = .horizontal
        actions.spacing = 8

        let copy = NSStackView(views: [eyebrow, statusLabel, statusRow, actions])
        copy.orientation = .vertical
        copy.alignment = .leading
        copy.spacing = 12
        copy.setCustomSpacing(7, after: eyebrow)
        copy.setCustomSpacing(18, after: statusLabel)
        copy.translatesAutoresizingMaskIntoConstraints = false

        let card = NSView()
        card.wantsLayer = true
        card.layer?.backgroundColor = AptusPalette.porcelain.cgColor
        card.layer?.cornerRadius = 18
        card.layer?.borderColor = AptusPalette.hairline.cgColor
        card.layer?.borderWidth = 1
        card.translatesAutoresizingMaskIntoConstraints = false
        card.addSubview(mark)
        card.addSubview(statusRule)
        card.addSubview(copy)
        root.addSubview(card)

        NSLayoutConstraint.activate([
            card.centerXAnchor.constraint(equalTo: root.centerXAnchor),
            card.centerYAnchor.constraint(equalTo: root.centerYAnchor),
            card.widthAnchor.constraint(equalToConstant: 620),
            card.heightAnchor.constraint(equalToConstant: 270),

            mark.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 42),
            mark.centerYAnchor.constraint(equalTo: card.centerYAnchor),
            mark.widthAnchor.constraint(equalToConstant: 142),
            mark.heightAnchor.constraint(equalToConstant: 142),

            statusRule.leadingAnchor.constraint(equalTo: mark.trailingAnchor, constant: 34),
            statusRule.centerYAnchor.constraint(equalTo: card.centerYAnchor),
            statusRule.widthAnchor.constraint(equalToConstant: 3),
            statusRule.heightAnchor.constraint(equalToConstant: 142),

            copy.leadingAnchor.constraint(equalTo: statusRule.trailingAnchor, constant: 28),
            copy.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -38),
            copy.centerYAnchor.constraint(equalTo: card.centerYAnchor),
        ])
        view = root
    }

    func showStarting() {
        _ = view
        statusLabel.stringValue = "Starting Aptus"
        detailLabel.stringValue = "Preparing the private local planning service."
        statusRule.layer?.backgroundColor = AptusPalette.circuitTeal.cgColor
        progress.isHidden = false
        progress.startAnimation(nil)
        retryButton.isHidden = true
        logButton.isHidden = true
    }

    func showFailure(_ message: String) {
        _ = view
        statusLabel.stringValue = "Aptus could not start"
        detailLabel.stringValue = message
        statusRule.layer?.backgroundColor = AptusPalette.faultRed.cgColor
        progress.stopAnimation(nil)
        progress.isHidden = true
        retryButton.isHidden = false
        logButton.isHidden = false
    }

    @objc private func retry() {
        onRetry?()
    }

    @objc private func showLog() {
        onShowLog?()
    }
}
