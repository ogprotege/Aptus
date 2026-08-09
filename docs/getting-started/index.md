# Getting Started

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** First-time users | **Last reviewed:** 2026-08-04 | **Review by:** 2026-10-22 or when onboarding changes

Choose the smallest path that proves what you need.

## Recommended order

1. [Choose your Aptus path](choose-your-path.md) to separate local planning from
   CUDA execution.
2. [Install Aptus](install.md) and verify the CLI.
3. Run the [first planning-only tutorial](first-plan.md). It is executable on
   this Mac and starts no training.
4. Use the [CUDA target-host quickstart](quickstart.md) only after replacing all
   model and hardware facts.

## Evidence checkpoints

| Checkpoint | Safe conclusion |
| --- | --- |
| Dataset profile | The local source was parsed and measured as recorded |
| Plan | Aptus compared its bounded candidate set against entered facts and wrote an `aptus.training-plan.v6` bound to one policy-snapshot digest |
| Package-free static pass | The exact `aptus.bundle.v3` frozen snapshot, decision parity, identities, and structure passed; current host-policy currency was not established |
| Installed-host static pass | The same bundle checks passed and the snapshot matched the installed host registry at validation time |
| Pilot pass | The exact real-model path completed the bounded target-host pilot |
| Measured run pass | The parent verified the bound full run and export structure |

None of these checkpoints alone proves task quality. A v4-or-earlier plan, or a
coherent v5 plan whose policy semantics or snapshot digest is no longer current,
must be replanned from preserved facts rather than edited in place.

## Related documentation

- [Documentation home](../index.md)
- [Current capabilities](../product/current-capabilities.md)
- [Troubleshooting](../guides/troubleshooting.md)
- [Operator checklist](../operations/operator-checklist.md)
