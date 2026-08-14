## Summary

Describe the user-visible outcome and the contract that changed.

## Contract and evidence

- [ ] I identified the implementation authority and every affected consumer.
- [ ] I recorded whether schema, formula, identity, state, or bundle versions change.
- [ ] Unsupported and unknown cases remain fail-closed with explicit reasons.
- [ ] Estimates, measurements, structural checks, and quality claims remain distinct.

## Mission sustain (M9 — every PR)

- [ ] This change makes a false “yes” less likely, or does not change that risk.
- [ ] Claim language still matches evidence; no claim expanded without a packet.
- [ ] No rejection was hidden.
- [ ] No method was added without a compiler and gates.
- [ ] Stop list still holds: no undocumented universal recommendation, silent install, MLX resume, unauthenticated multi-user jobs, or ladder-bypassing agent train.

## Verification

List exact commands and results. Include target hardware, model revision, and
dataset digest only when real runtime evidence was collected.

- [ ] Python tests and static checks pass.
- [ ] Web tests, typecheck, and production build pass when applicable.
- [ ] A fresh generated bundle was reviewed when compiler output changed.
- [ ] Wheel and installed-package smoke checks pass when packaging changed.
- [ ] Required CUDA pilots were run, or missing target-host evidence is stated.

## Documentation

- [ ] User guides, reference, architecture, operations, and claim language were reviewed.
- [ ] CLI help, API shapes, web copy, and generated bundle guidance match behavior.
- [ ] Local links, anchors, metadata, and the documentation health/debt records pass.

## Sensitive material

- [ ] This change contains no token, private dataset, cache, checkpoint, model weight, raw job state, or personal machine artifact.
