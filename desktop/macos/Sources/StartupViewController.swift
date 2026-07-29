import AppKit

private final class AppearanceTrackingView: NSView {
    var onAppearanceChange: (() -> Void)?

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        onAppearanceChange?()
    }
}

final class StartupViewController: NSViewController {
    var onRetry: (() -> Void)?
    var onShowLog: (() -> Void)?

    private let statusLabel = NSTextField(labelWithString: "Starting Aptus")
    private let detailLabel = NSTextField(wrappingLabelWithString: "Preparing the private local planning service.")
    private let progress = NSProgressIndicator()
    private let statusRule = NSView()
    private let retryButton = NSButton(title: "Retry", target: nil, action: nil)
    private let logButton = NSButton(title: "Show Backend Log", target: nil, action: nil)
    private weak var rootSurface: NSView?
    private weak var cardSurface: NSView?
    private weak var markSurface: NSView?
    private var statusColor = AptusPalette.circuitTeal

    override func loadView() {
        let root = AppearanceTrackingView()
        root.wantsLayer = true
        root.onAppearanceChange = { [weak self] in self?.updateLayerColors() }
        rootSurface = root

        let markCanvas = NSView()
        markCanvas.translatesAutoresizingMaskIntoConstraints = false

        let markBackground = NSView()
        markBackground.wantsLayer = true
        markBackground.layer?.cornerRadius = 18.3
        markBackground.translatesAutoresizingMaskIntoConstraints = false
        markSurface = markBackground

        let mark = NSImageView()
        if let source = Bundle.main.image(forResource: "AptusMark"),
           let image = source.copy() as? NSImage {
            image.isTemplate = true
            mark.image = image
        }
        mark.contentTintColor = .white
        mark.imageScaling = .scaleProportionallyUpOrDown
        mark.translatesAutoresizingMaskIntoConstraints = false
        markCanvas.addSubview(markBackground)
        markCanvas.addSubview(mark)

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
        card.layer?.cornerRadius = 18
        card.layer?.borderWidth = 1
        cardSurface = card
        card.translatesAutoresizingMaskIntoConstraints = false
        card.addSubview(markCanvas)
        card.addSubview(statusRule)
        card.addSubview(copy)
        root.addSubview(card)

        NSLayoutConstraint.activate([
            card.centerXAnchor.constraint(equalTo: root.centerXAnchor),
            card.centerYAnchor.constraint(equalTo: root.centerYAnchor),
            card.widthAnchor.constraint(equalToConstant: 620),
            card.heightAnchor.constraint(equalToConstant: 270),

            markCanvas.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 42),
            markCanvas.centerYAnchor.constraint(equalTo: card.centerYAnchor),
            markCanvas.widthAnchor.constraint(equalToConstant: 142),
            markCanvas.heightAnchor.constraint(equalToConstant: 142),

            markBackground.leadingAnchor.constraint(equalTo: markCanvas.leadingAnchor, constant: 10),
            markBackground.trailingAnchor.constraint(equalTo: markCanvas.trailingAnchor, constant: -10),
            markBackground.topAnchor.constraint(equalTo: markCanvas.topAnchor, constant: 10),
            markBackground.bottomAnchor.constraint(equalTo: markCanvas.bottomAnchor, constant: -10),

            mark.leadingAnchor.constraint(equalTo: markCanvas.leadingAnchor),
            mark.trailingAnchor.constraint(equalTo: markCanvas.trailingAnchor),
            mark.topAnchor.constraint(equalTo: markCanvas.topAnchor),
            mark.bottomAnchor.constraint(equalTo: markCanvas.bottomAnchor),

            statusRule.leadingAnchor.constraint(equalTo: markCanvas.trailingAnchor, constant: 34),
            statusRule.centerYAnchor.constraint(equalTo: card.centerYAnchor),
            statusRule.widthAnchor.constraint(equalToConstant: 3),
            statusRule.heightAnchor.constraint(equalToConstant: 142),

            copy.leadingAnchor.constraint(equalTo: statusRule.trailingAnchor, constant: 28),
            copy.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -38),
            copy.centerYAnchor.constraint(equalTo: card.centerYAnchor),
        ])
        view = root
        updateLayerColors()
    }

    func showStarting() {
        _ = view
        statusLabel.stringValue = "Starting Aptus"
        detailLabel.stringValue = "Preparing the private local planning service."
        statusColor = AptusPalette.circuitTeal
        updateLayerColors()
        progress.isHidden = false
        progress.startAnimation(nil)
        retryButton.isHidden = true
        logButton.isHidden = true
    }

    func showFailure(_ message: String) {
        _ = view
        statusLabel.stringValue = "Aptus could not start"
        detailLabel.stringValue = message
        statusColor = AptusPalette.faultRed
        updateLayerColors()
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

    private func updateLayerColors() {
        guard isViewLoaded else { return }
        view.effectiveAppearance.performAsCurrentDrawingAppearance {
            rootSurface?.layer?.backgroundColor = AptusPalette.cloud.cgColor
            cardSurface?.layer?.backgroundColor = AptusPalette.porcelain.cgColor
            cardSurface?.layer?.borderColor = AptusPalette.hairline.cgColor
            markSurface?.layer?.backgroundColor = AptusPalette.brandTeal.cgColor
            statusRule.layer?.backgroundColor = statusColor.cgColor
        }
    }
}
