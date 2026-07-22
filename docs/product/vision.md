# Product Vision

Fine-tuning setup consumes time because model, data, hardware, objective,
precision, quantization, memory, distribution, dependencies, and artifacts must
agree. Aptus makes those decisions explicit and testable.

The intended product accepts pinned facts, compares a declared strategy catalog,
shows assumptions and resource tradeoffs, and emits an executable bundle. It
then converts uncertainty into evidence through ordered checks on the target
host.

## Product principles

- Facts carry provenance.
- Unsupported paths stay visible and fail closed.
- Estimates remain separate from measurements.
- Recommendations name their candidate set and objective.
- Generated artifacts are reviewable and identity-bound.
- Runtime evidence must belong to the exact plan, model, data, environment, and
  hardware.
- Full runs never overwrite earlier output.
- A child process cannot certify its own successful completion.
- Quality requires an explicit evaluation contract.

## End state

V0.2 already has a versioned method-descriptor registry. It separates four
selectable, gated executable identities from four experimental and three
research-only identities. Registry presence alone never grants planner or
compiler access.

The longer-term system can extend that registry with method-specific estimators,
compilers, checkpoint and restart state, export and reload contracts, and real
pilot evidence. It can also add calibrated priors, evaluation targets,
additional execution backends, cloud runners, provider integrations, and
controlled automation interfaces. Each addition must enter through an explicit
contract and its own evidence gate.

V0.2 is the local CUDA core. It does not yet implement those later integrations
and has not completed release CUDA evidence.
