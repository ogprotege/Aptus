# Aptus Legacy Provenance Report

> **Documentation status:** Archived evidence
>
> **Applies to:** Dated forensic review of the removed legacy `HyperTune/` tree
>
> **Last reviewed:** 2026-08-06
>
> **Next scheduled review:** 2027-08-06, or when provenance or reproduction paths change
>
> **Historical warning:** The body below is preserved point-in-time evidence,
> not a current dependency or legal determination. Start with the
> [audit index](README.md) or [current capabilities](../../product/current-capabilities.md).

Date: 2026-07-21

This report covers provenance-sensitive material recovered from the legacy
`HyperTune/` folder. It does not provide legal advice. Machine-readable claims
and source links are in `claims-and-provenance.jsonl`.

## Determinations

### Vendored PyReFT code

The files under `HyperTune/PyReft-Repo/` identify themselves as Stanford NLP
PyReFT and link to `stanfordnlp/pyreft`. The exact copied revision, file-level
modifications, and chain of custody are not recorded. The current upstream
repository uses Apache License 2.0:

- Upstream project: https://github.com/stanfordnlp/pyreft
- Upstream license: https://github.com/stanfordnlp/pyreft/blob/main/LICENSE

The current upstream license requires recipients to receive the license, modified files
to carry change notices, source distributions to retain applicable copyright
and attribution notices, and distributions to preserve upstream NOTICE
attributions when a NOTICE file exists.

This verifies the current upstream license, not that every vendored file is
covered by that revision or unchanged from it. The legacy folder does not
preserve a complete upstream license/NOTICE beside
the vendored code. Its own `HyperTune/LICENSE` is only a two-line “MIT” label,
while other legacy prose calls the system proprietary. These statements cannot
establish ownership or relicense third-party code.

Disposition: archive the vendored tree with provenance warnings. Aptus should
prefer a pinned upstream dependency or a clean integration rather than copying
the source into its product tree.

### Node dependency identity

Registry metadata checks on 2026-07-21 produced:

- `mcp-server@^0.1.0`: npm returned E404; the declared range is not
  resolvable.
- `huggingface-api@^0.3.0`: npm returned E404; the declared range is not
  resolvable.
- `@modelcontextprotocol/sdk`: the official package exists; npm reported
  version `1.29.0`, MIT license, and repository
  `modelcontextprotocol/typescript-sdk`.

This proves that `HyperTune/package.json` cannot produce its declared Node
environment. It does not by itself prove that every historical package name or
API was fabricated; it proves that the declared dependency graph is unavailable
now.

### ReFT claims

The ReFT paper (arXiv:2404.03592) reports that LoReFT interventions were
15–65 times more parameter-efficient than LoRA across the paper’s evaluated
benchmarks. The same paper limits the explored model set primarily to the
LLaMA family, describes ReFT as hyperparameter-sensitive, and identifies the
large search space as unresolved.

Disposition: preserve the claim only as a cited research result. Do not turn it
into a universal Aptus performance promise or treat the legacy task table as
ground truth.

### QLoRA claims

The QLoRA paper (arXiv:2305.14314) supports 4-bit NF4 storage, commonly BF16
computation, double quantization, and paged optimizers as concrete memory-saving
techniques. It reports double quantization saving about 0.37 bits per parameter
and demonstrates 65B fine-tuning on a single 48 GB GPU under its experimental
configuration.

The paper does not validate the legacy project’s simple “parameter count divided
by eight” estimator for arbitrary model families, sequence lengths, kernels,
optimizers, or devices.

Disposition: retain QLoRA choices as strategy candidates. Replace fixed memory
ratios with versioned, component-level estimates and calibration data.

### Closed-model specifications

`HyperTune-NEW_stuff_05-16-25/manual-models.json` assigns parameter counts to
closed GPT, Claude, and Gemini systems without primary-source citations.

Disposition: archive as historical brainstorming and mark DO NOT SHIP. Aptus
must not calculate hardware plans from undisclosed model internals.

## Required controls for future Aptus work

1. Every imported algorithm, table, dataset, and code block needs a source,
   license, confidence level, and scope statement.
2. Research-derived defaults must be labeled as initial priors until Aptus
   reproduces or calibrates them.
3. Generated dependency locks must record registry integrity hashes.
4. Closed-model support must use provider-published capabilities rather than
   guessed architecture or parameter data.
5. Product copy must distinguish “paper reports,” “legacy heuristic,”
   “Aptus-measured,” and “guaranteed.”
