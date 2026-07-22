# Getting Started

> **Status:** Active | **Authority:** Documentation navigation | **Applies to:** Aptus 0.2 | **Audience:** First-time users | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when onboarding changes

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
| Plan | Aptus compared its bounded candidate set against entered facts |
| Static pass | The exact compiled bundle passed structural checks |
| Pilot pass | The exact real-model path completed the bounded target-host pilot |
| Measured run pass | The parent verified the bound full run and export structure |

None of these checkpoints alone proves task quality.

## Related documentation

- [Documentation home](../index.md)
- [Current capabilities](../product/current-capabilities.md)
- [Troubleshooting](../guides/troubleshooting.md)
- [Operator checklist](../operations/operator-checklist.md)
