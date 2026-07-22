# Aptus Documentation

Aptus v0.2 plans, compiles, validates, and locally runs a bounded set of
fine-tuning strategies. It is an engineering preview, not a released training
service.

## Start here

- [Install](getting-started/install.md)
- [Quickstart](getting-started/quickstart.md)
- [Current capabilities](product/current-capabilities.md)
- [User workflows](product/user-workflows.md)
- [Troubleshooting](guides/troubleshooting.md)

## Use Aptus

- [Model, dataset, and hardware facts](guides/model-dataset-hardware.md)
- [Compare plans](guides/compare-plans.md)
- [Compile, validate, and run](guides/compile-validate-run.md)
- [Recovery and the resume boundary](guides/resume-recover.md)

## Understand the system

- [System architecture](architecture/system.md)
- [Artifact compiler](architecture/artifact-compiler.md)
- [Execution orchestrator](architecture/execution-orchestrator.md)
- [Security boundaries](architecture/security-boundaries.md)
- [UI and UX contract](product/ui-ux.md)
- [Product vision](product/vision.md)
- [Claim language](product/claim-language.md)

## Reference

- [CLI](reference/cli.md)
- [API](reference/api.md)
- [Plan schema](reference/plan-schema.md)
- [Bundle manifest](reference/bundle-manifest.md)
- [Capability matrix](reference/capability-matrix.md)
- [Validation states](reference/validation-states.md)
- [Run states](reference/run-states.md)
- [Error codes](reference/error-codes.md)
- [Glossary](reference/glossary.md)

## Operations

- [Release gates](operations/release-gates.md)
- [Security policy](../SECURITY.md)
- [Contributing](../CONTRIBUTING.md)
- [Roadmap](../ROADMAP.md)

## Evidence notice

The checked-in test suite does not replace a target-host pilot. No real CUDA
pilot has been completed on the current development Mac. Current support claims
must remain conditional until the release gates are satisfied.
