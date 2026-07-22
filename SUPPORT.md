# Support

> **Status:** Active | **Authority:** Repository support policy | **Applies to:** Aptus 0.2 | **Audience:** Users and contributors | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when support channels change

Aptus 0.2 is an engineering preview. Support begins with reproducible evidence
and the current documented boundary.

## Before reporting a problem

1. Read [current capabilities](docs/product/current-capabilities.md) and
   [troubleshooting](docs/guides/troubleshooting.md).
2. Confirm the exact Aptus commit and Python version.
3. Record the command, exit status, validation state, finding codes, and job ID.
4. Include the plan and candidate IDs, but redact local paths when necessary.
5. State whether the facts were measured, provider-declared, user-attested, or
   inferred.
6. State the operating system, backend, device names, and package environment.

Do not attach tokens, private datasets, model weights, checkpoints, caches,
unredacted logs, or proprietary provider responses.

## Where to report

- Use the [repository issue forms](https://github.com/ogprotege/Aptus/issues/new/choose)
  for reproducible, non-sensitive defects and documentation errors.
- Use a pull request for a tested correction that follows
  [CONTRIBUTING.md](CONTRIBUTING.md).
- Use the private security path described in [SECURITY.md](SECURITY.md) for a
  vulnerability or any report that would expose sensitive details.

## What support cannot establish

A discussion or issue response cannot authorize a run, grant model or dataset
rights, prove hardware fit, or certify model quality. Those conclusions require
the exact contracts and evidence described in the documentation.

## Related documentation

- [Documentation index](docs/index.md)
- [Operator checklist](docs/operations/operator-checklist.md)
- [Error and finding codes](docs/reference/error-codes.md)
