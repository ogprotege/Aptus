# Aptus MoE Product Surface Design Plan

**Last Updated:** 2026-07-27

## Subject and Job

The subject is evidence-backed MoE fine-tuning on Apple Silicon. The audience is a practitioner who must decide whether a large sparse model fits and exactly what Aptus will train. The screen's job is to distinguish resident scale, active compute, and adapter scope before planning.

## Compact Token System

- Cloud `#f3f7f8`: application field
- Porcelain `#ffffff`: evidence panels
- Graphite `#17252b`: primary information
- Circuit teal `#0b6670`: verified topology and supported route
- Calibration amber `#884500`: derived facts and pilot-required limits
- Fault red `#a43a32`: unsupported or mismatched topology

Typography remains Aptus-specific: Familjen Grotesk for headings, Atkinson Hyperlegible for working text, and IBM Plex Mono for model identifiers, counts, and bindings.

## Layout

The existing Model panel gains one compact topology instrument below provider inspection.

```text
+------------------------------------------------------+
| Mixture of experts                EXACT FAMILY       |
|                                                      |
| 30.53B resident  ==============================      |
|  3.35B active    ===                                 |
|                                                      |
| [128 experts] [8 active/token] [48 sparse layers]   |
| Adapter scope: attention q/k/v/o only                |
+------------------------------------------------------+
```

The long rail encodes total resident parameters. The shorter nested rail encodes the topology-derived active subset. The ratio is data, not decoration.

## Signature

The expert-topology rail is the sole new signature element. It makes the central MoE distinction visible without suggesting that inactive experts disappear from memory.

## Self-Critique and Revision

A separate dashboard, network animation, or new accent palette would make the feature feel detached from Aptus and overstate routing as a live measurement. The revision keeps the feature inside evidence intake, uses existing tokens, and limits motion to none. The rail is static because it represents configuration, not observed token routing.

The first sketch risked making "active" look like free memory. The final copy pairs the rail with an explicit resident-memory sentence and labels active parameters as topology-derived.

## Build Constraints

- Preserve keyboard focus and current responsive stacking.
- Use semantic `dl` data and an accessible text summary before visual rails.
- Do not rely on color to distinguish total from active.
- Keep reduced-motion behavior unchanged by adding no animation.
- Show the instrument only when a complete, exact MoE topology is present.
