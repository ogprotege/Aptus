# Aptus Legacy Recovery Audit

> **Documentation status:** Archived evidence
>
> **Applies to:** Dated forensic snapshot of the removed legacy `HyperTune/` tree
>
> **Last reviewed:** 2026-07-22
>
> **Next scheduled review:** 2027-07-22, or when provenance or reproduction paths change
>
> Preserve the historical reports and machine-readable records as one evidence
> bundle. They do not describe current Aptus behavior. Use the
> [historical index](../../archive/index.md) and
> [current capabilities](../../product/current-capabilities.md) for orientation.

This directory contains the preserved evidence and conclusions from the
forensic review of the former legacy `HyperTune/` source folder. HyperTune is
the historical source name; Aptus is the product name. The working legacy
folder was removed after extraction, and the user retains a separate backup.

## Start here

1. `executive-summary.md` — direct verdict and recommended next step.
2. `hidden-gems.md` — ranked ideas and implementation seams worth recovering.
3. `failure-and-risk-register.md` — correctness, security, provenance, and
   overclaim findings.
4. `architecture-options.md` — candidate Aptus architectures for a separate
   design phase.
5. `static-typescript.md` and `static-python.md` — language-specific evidence.
6. `provenance-report.md` — third-party, dependency, and research-claim checks.
7. `sandbox-summary.md` — bounded dynamic verification and blocked gates.

## Machine-readable evidence

- `baseline-manifest.json` and `inventory.jsonl`
- `duplicate-clusters.json` and `version-families.json`
- `reference-map.json`
- `secret-scan.json`
- `claims-and-provenance.jsonl`
- `sandbox-results.jsonl`
- `classification.jsonl` and `classification-summary.json`
- `generated-bundle-manifest.json`

## Snapshot totals

- 228 artifacts, 1,879,017 bytes
- 38 exact duplicate clusters covering 98 files
- 30 normalized version families
- 23 empty files
- 3 Python parse failures
- 40 missing relative imports
- 35 ADAPT, 126 ARCHIVE, 67 DISCARD, 0 KEEP

## Reproduce the local evidence

From the Aptus repository root:

```bash
python3 -m tools.aptus_audit.generate \
  /path/to/local/legacy-copy \
  docs/audits/aptus-legacy
python3 -m unittest discover -s tests/tools -v
```

The static generator is fail-closed and transactionally replaces the complete
machine-generated bundle. The recorded dynamic checks are historical evidence.
To rerun them, temporarily restore the backup at the repository-root path
`HyperTune/`, review the explicit subprocess plan, and run
`python3 -m tools.aptus_audit.run_checks --allow-host-subprocesses` only inside
an externally sandboxed environment. The runner does not enforce OS-level
isolation itself. Registry results and compatibility failures can change over
time; JSONL records include timestamps, commands, inherited environment-key
names, output hashes, and bounded previews.

No classification authorizes deletion. Extraction, cleanup, and Aptus product
implementation require separate approval.
