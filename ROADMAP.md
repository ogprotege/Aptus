# Roadmap

The roadmap separates the executable v0.2 contract from future work. An item on
this page is not a supported capability until code, tests, documentation, and
target-host evidence all agree.

## v0.2 stabilization

- Complete a real CUDA run for every claimed executable method and placement.
- Record clean-environment dependency installation on each supported path.
- Prove managed cancellation, stale-owner recovery, global-lease behavior, and
  crash-safe completion promotion on the target operating systems.
- Verify structural exports by loading the pinned base model and adapter where
  applicable.
- Run browser accessibility and responsive checks against the packaged web app.
- Publish a reproducible release evidence record.

Full-parameter FSDP remains unsupported during v0.2. LoRA FSDP remains
conditional until its runtime gate is complete.

## Planner depth

- Version the method catalog independently from planner code.
- Separate training objective, parameterization, recipe modifiers, optimizer,
  precision, quantization, and distribution as explicit planning axes.
- Add calibrated device and model-family priors without replacing measured
  pilot evidence.
- Add target wall-time and budget constraints with honest abstention.
- Add richer dataset quality, contamination, and task-shape diagnostics.

## Evaluation and export contracts

- Define evaluation datasets, metrics, thresholds, and baseline comparisons as
  first-class target facts.
- Bind evaluation results to the exact run and exported artifact.
- Define exporter interfaces for adapters, merged models, and deployment
  packages.
- Add semantic load and inference checks beyond the current structural file-tree
  verification.

## Execution and recovery

- Design a full checkpoint manifest that binds model, optimizer, scheduler,
  scaler, RNG, dataloader progress, environment, plan, and distributed topology.
- Enable full-run resume only after that manifest survives interruption tests.
- Add an explicit deep re-verification command for historical artifacts.
- Add retention and cleanup policies for unique no-clobber runs and caches.

## Additional platforms and integrations

- Evaluate ROCm, then MPS, under separate capability and pilot gates.
- Add cloud runner and provider adapters behind explicit credentials and cost
  boundaries.
- Add MCP and external automation adapters after the local authorization model
  is defined.
- Add experiment trackers as optional sinks, never as the source of truth for
  local completion.

## Non-goals

Aptus will not claim universal strategy optimality, guaranteed fit, guaranteed
quality, or automatic permission to train a model or dataset.
